import logging
from copy import deepcopy
from typing import List, Callable, Union, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed

from elphick.mass_composition import MassComposition


class Node:
    def __init__(self, operation: Callable[..., Union[MassComposition, Tuple[MassComposition, MassComposition]]],
                 dependencies: List[str], kwargs: dict = None):
        self.operation = operation
        self.dependencies = dependencies
        self.kwargs = kwargs if kwargs is not None else {}


class DAG:
    def __init__(self, name: str = 'DAG', n_jobs=-1):
        self.name = name
        self.n_jobs = n_jobs  # Number of workers for parallel execution
        self.graph = nx.DiGraph()
        self.results = {}  # Store the results of node executions

    @property
    def mass_compositions(self):
        mass_compositions = {}
        for node in self.graph.nodes:
            if node in self.results:
                result = self.results[node]
                if isinstance(result, MassComposition):
                    mass_compositions[node] = result
                elif isinstance(result, tuple) and all(isinstance(r, MassComposition) for r in result):
                    for r in result:
                        mass_compositions[r.name] = r
            else:
                # If the node is not in the results dictionary, it is a leaf node
                # Retrieve its result from the MassComposition objects associated with the node in the graph
                mc = self.graph.nodes[node]['mc']
                mass_compositions[node] = mc
        return mass_compositions

    @property
    def all_nodes_(self):
        """Identify all nodes in the DAG."""
        return list(self.graph.nodes)

    @staticmethod
    def input(mc: MassComposition) -> MassComposition:
        return mc

    @staticmethod
    def output(inputs: Union[MassComposition, List[MassComposition,]], name: str) -> MassComposition:
        res: Optional[MassComposition] = None
        if isinstance(inputs, MassComposition):
            if inputs.name == name:
                res = inputs
        else:
            inputs = [item for sublist in inputs for item in sublist]  # flatten the list.
            for mc in inputs:
                if mc.name == name:
                    return mc
        return res

    def add_node(self, name: str, operation: Callable[..., MassComposition], dependencies: List[str] = None,
                 kwargs: dict = None, defined: bool = True, mc: MassComposition = None) -> 'DAG':
        dependencies = dependencies if dependencies is not None else []
        self.graph.add_node(name, operation=operation, kwargs=kwargs, defined=defined, name=name, mc=mc,
                            dependencies=dependencies)
        for dependency in dependencies:
            self.graph.add_edge(dependency, name)

        # Validate the uniqueness of the step names and the MassComposition names
        self._validate_unique_names()
        return self

    def topological_sort(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def run(self, mcs: dict):
        logging.info("Running the DAG")  # Log the node that is being executed

        # Set the 'mc' attribute of the nodes to the corresponding MassComposition objects from the mcs dictionary
        for node, mc in mcs.items():
            if node in self.graph.nodes:
                self.graph.nodes[node]['mc'] = mc

        # Add nodes to the graph in the order they appear in the mcs dictionary
        for node in mcs.keys():
            self.add_node(node, self.graph.nodes[node]['operation'], self.graph.nodes[node]['dependencies'],
                          kwargs=self.graph.nodes[node]['kwargs'], defined=self.graph.nodes[node]['defined'],
                          mc=self.graph.nodes[node]['mc'])

        executed_nodes = set()  # Keep track of nodes that have been executed

        while len(executed_nodes) < len(self.graph):
            # Find nodes with no predecessors that haven't been executed yet
            ready_nodes = [node for node in self.graph.nodes if
                           all(pred in executed_nodes for pred in self.graph.predecessors(node)) and
                           node not in executed_nodes]

            logging.info(f"Ready nodes: {ready_nodes}")
            logging.info(f"Executed nodes: {list(executed_nodes)}")

            if not ready_nodes:
                unexecuted_nodes = set(self.graph.nodes) - executed_nodes
                for node in unexecuted_nodes:
                    predecessors = list(self.graph.predecessors(node))
                    logging.info(f"Node {node} is waiting for {predecessors}")

            # Create a job for each ready node
            jobs = [delayed(self.execute_node)(node, mcs) for node in ready_nodes]

            # Execute the jobs in parallel
            if jobs:
                results, _ = zip(*Parallel(n_jobs=self.n_jobs)(jobs))
            else:
                results = []

            # Update self.results and executed_nodes with the returned value of each job
            for i, result in enumerate(results):
                if result is not None:
                    if isinstance(result, tuple):
                        for r in result:
                            if isinstance(r, MassComposition):
                                self.results[r.name] = r
                    else:
                        self.results[ready_nodes[i]] = result
                executed_nodes.add(ready_nodes[i])

        self._update_mc_nodes()

    def execute_node(self, node: str, mcs: dict):

        logging.info(f"Executing node {node}")  # Log the node that is being executed
        operation = self.graph.nodes[node]['operation']
        kwargs = self.graph.nodes[node]['kwargs']
        defined = self.graph.nodes[node]['defined']

        # Check if the node is in the mcs dictionary
        if node in mcs:
            mc = mcs[node]
            # Check if kwargs is not None before passing it to the operation
            result = operation(mc, **kwargs) if kwargs is not None else operation(mc)
        else:
            # If the node is not in the mcs dictionary, it means that it needs to be created inside the DAG
            # In this case, execute the operation with the results of its dependencies as inputs
            # Check if the results of the predecessors are available
            if all(dependency in self.results for dependency in self.graph.predecessors(node)):
                inputs = [self.results[dependency] for dependency in self.graph.predecessors(node)]
                # Check if kwargs is not None before passing it to the operation
                if operation == DAG.output:
                    result = operation(inputs, name=node)
                else:
                    result = operation(*inputs, **kwargs) if kwargs is not None else operation(*inputs)
            else:
                logging.info(
                    f"Waiting for predecessors of node {node}")  # Log the node that is waiting for its predecessors
                return None, []

        # If the node has successors and is defined, store the result of the node execution
        if list(self.graph.successors(node)) and defined:
            self.results[node] = result
            logging.info(f"Stored result for node {node}")  # Log the node for which a result was stored

        # If the result is a tuple, do not add new nodes to the graph
        # Instead, return the MassComposition objects in the tuple
        if isinstance(result, tuple):
            return result, []
        # Set the 'mc' attribute of the edges to the result after the worker returns
        elif operation == DAG.output:
            self.graph.edges[list(self.graph.predecessors(node))[0], node]['mc'] = result
        else:
            for successor in self.graph.successors(node):
                self.graph.edges[node, successor]['mc'] = result

        return result, []

    def plot(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1,
                font_size=10, font_weight='bold', width=2)
        plt.show()

    def _validate_unique_names(self):
        # Validate the uniqueness of the step names
        node_names = [name for name in self.graph.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Step names are not unique within the DAG.")

        # Validate the uniqueness of the MassComposition names
        mc_names = [data['mc'].name for _, data in self.graph.nodes(data=True) if
                    'mc' in data and data['mc'] is not None]
        duplicates = [name for name in mc_names if mc_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"MassComposition names are not unique within the DAG. Duplicates: {set(duplicates)}")

    def _update_mc_nodes(self):
        for edge in self.graph.edges:
            mc = self.graph.edges[edge]['mc']
            src_node, dst_node = edge
            mc.nodes = [src_node, dst_node]
