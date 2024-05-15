import logging
from typing import List, Callable, Union, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed

from elphick.mass_composition import MassComposition

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, operation: Callable[..., Union[MassComposition, Tuple[MassComposition, MassComposition]]],
                 dependencies: List[str], kwargs: dict = None):
        self.operation = operation
        self.dependencies = dependencies
        self.kwargs = kwargs if kwargs is not None else {}


class DAG:
    def __init__(self, name: str = 'DAG', n_jobs=-1):
        self.name = name
        self.streams = {}  # Map stream names to MassComposition objects
        self.stream_parent_node = {}  # Map stream names to node names
        self.n_jobs = n_jobs  # Number of workers for parallel execution
        self.graph = nx.DiGraph()
        self.node_executed = {}  # Store the execution state of nodes

    @property
    def mass_compositions(self) -> dict[str, MassComposition]:
        """
        Retrieves all the MassComposition objects associated with the nodes in the DAG.

        This property iterates over all the nodes in the DAG, checks if the node has been executed,
        and if so, retrieves the result of the node execution. If the node has not been executed,
        it retrieves the MassComposition object associated with the node in the graph.

        Returns:
            dict[str, MassComposition]: A dictionary where the keys are node names and the values
            are MassComposition objects associated with those nodes.
        """
        mass_compositions = {}
        for node in self.graph.nodes:
            if node in self.node_executed:
                result = self.node_executed[node]
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
    def output(mc: MassComposition) -> MassComposition:
        return mc

    def _finalize(self):
        """
        Final checks before execution.
        """
        orphan_nodes = [node for node, degree in self.graph.degree() if degree == 0]
        if orphan_nodes:
            raise ValueError(f"Orphan nodes: {orphan_nodes} exist.  Please check your configuration.")

        # Validate unique names
        self._validate_unique_names()

    def add_input(self, name: str):
        self.graph.add_node(name, operation=DAG.input, kwargs=None, defined=True, name=name, mc=None,
                            dependencies=[])
        # Update the stream_to_node mapping
        self.stream_parent_node[name] = name

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

    def add_step(self, name: str, operation: Callable, streams: List[str], kwargs: dict = None, defined: bool = True):
        # Determine dependencies from the input streams
        dependencies = [self.stream_parent_node[stream] for stream in streams]
        self.graph.add_node(name, operation=operation, dependencies=dependencies, kwargs=kwargs, defined=defined)
        for stream in streams:
            self.graph.add_edge(self.stream_parent_node[stream], name, name=stream)
        if kwargs is not None:
            for key, value in kwargs.items():
                if key in ['name', 'name_1', 'name_2']:
                    self.stream_parent_node[value] = name

    def add_output(self, name: str, stream: str):
        parent_node = self.stream_parent_node.get(stream)
        if parent_node is None:
            raise ValueError(f"No parent node found for stream {stream}")
        self.graph.add_node(name, operation=DAG.output, dependencies=[stream], kwargs=None, defined=True, name=name)
        self.graph.add_edge(parent_node, name)

    def _topological_sort(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def run(self, mcs: dict):
        """
        Executes the Directed Acyclic Graph (DAG).

        This method takes a dictionary of MassComposition objects as input and executes the operations defined in
        the DAG.
        The execution starts from the input nodes and proceeds in a topological order, ensuring that each node
        is executed only after all its predecessors have been executed. The results of the node executions are
        stored in the `self.streams` dictionary.

        Parameters:
        mcs (dict): A dictionary mapping node names to MassComposition objects. These are the initial MassComposition objects
                    for the input nodes of the DAG.

        Returns:
        None
        """
        logging.info("Running the DAG")  # Log the node that is being executed
        self._finalize()

        # Initialize the execution state of all nodes to False
        for node in self.graph.nodes:
            self.node_executed[node] = False

        executed_nodes = set()  # Keep track of nodes that have been executed

        while len(executed_nodes) < len(self.graph):
            # Find nodes with no predecessors that haven't been executed yet
            ready_nodes = [node for node in self.graph.nodes if
                           all(pred in self.streams for pred in self.graph.predecessors(node)) and
                           node not in executed_nodes]

            logger.info(f"Ready nodes: {ready_nodes}")
            logger.info(f"Executed nodes: {list(executed_nodes)}")
            logger.info(f"Result streams: {self.streams}")

            if not ready_nodes:
                unexecuted_nodes = set(self.graph.nodes) - executed_nodes
                for node in unexecuted_nodes:
                    predecessors = list(self.graph.predecessors(node))
                    logger.info(f"Node {node} is waiting for {predecessors}")

            # Create a job for each ready node
            jobs = [delayed(self.execute_node)(node, mcs, executed_nodes) for node in ready_nodes]

            # Execute the jobs in parallel
            if jobs:
                results = Parallel(n_jobs=self.n_jobs)(jobs)
                # Filter out None values
                results = [result for result in results if result is not None]
                if results:
                    results, _ = zip(*results)
            else:
                results = []

            # Update self.results and executed_nodes with the returned value of each job
            for i, result in enumerate(results):
                executed_nodes.add(ready_nodes[i])

        # self._update_mc_nodes()

        # assign the streams to the edges
        for edge in self.graph.edges:
            src_node, dst_node = edge
            self.graph.edges[edge]['mc'] = self.streams[self.stream_parent_node[src_node]]

    def execute_node(self, node: str, mcs: dict, executed_nodes: set) -> Optional[Union[MassComposition, Tuple[MassComposition, ...]]]:
        """
        Executes a node in the DAG.

        This method takes a node and a dictionary of MassComposition objects. It executes the operation associated with the
        node using the MassComposition objects as inputs. If the node has successors and is defined, the result of the node
        execution is stored.

        Parameters:
        node (str): The name of the node to be executed.
        mcs (dict): A dictionary mapping node names to MassComposition objects.

        Returns:
        Union[MassComposition, Tuple[MassComposition, ...]]: The result of the node execution, or None if the node is waiting for its predecessors.
        """
        logger.info(f"Executing node {node}")  # Log the node that is being executed
        operation = self.graph.nodes[node]['operation']
        kwargs = self.graph.nodes[node]['kwargs']
        defined = self.graph.nodes[node]['defined']

        logger.info(f"State of self.streams before executing node {node}: {self.streams}")

        # Log the predecessors of the node
        predecessors = list(self.graph.predecessors(node))
        logger.info(f"Predecessors of node {node}: {predecessors}")

        try:
            # Check if the node is in the mcs dictionary
            if node in mcs:
                mc = mcs[node]
                # Check if kwargs is not None before passing it to the operation
                result = operation(mc, **kwargs) if kwargs is not None else operation(mc)
            else:
                # If the node is not in the mcs dictionary, it means that it needs to be created inside the DAG
                # In this case, execute the operation with the results of its dependencies as inputs
                # Check if the results of the predecessors are available
                if all(dependency in self.streams for dependency in self.graph.predecessors(node)):
                    inputs = [self.streams[dependency] for dependency in self.graph.predecessors(node)]
                    # If only one input stream is provided, retrieve the corresponding MassComposition object
                    if len(inputs) == 1:
                        inputs = inputs[0]
                        # Check if kwargs is not None before passing it to the operation
                        result = operation(inputs, **kwargs) if kwargs is not None else operation(inputs)
                    else:
                        # Ensure inputs is always an iterable
                        if isinstance(inputs, MassComposition):
                            inputs = [inputs]
                        # Check if kwargs is not None before passing it to the operation
                        result = operation(*inputs, **kwargs) if kwargs is not None else operation(*inputs)
                else:
                    logger.info(
                        f"Waiting for predecessors of node {node}")  # Log the node that is waiting for its predecessors
                    return None
        except AttributeError as e:
            logger.error(f"Error while executing node {node}: {e}")
            raise

        # If the node has successors and is defined, store the result of the node execution
        if list(self.graph.successors(node)) and defined:
            if isinstance(result, tuple):
                for mc in result:
                    self.streams[mc.name] = mc
                    logger.info(f"Stored results for stream {mc.name}")  # Log the node for which a result was stored
            else:
                self.streams[node] = result
                logger.info(f"Stored results for stream {node}")  # Log the node for which a result was stored

        # After executing the operation, update the execution state of the node
        executed_nodes.add(node)
        self.node_executed[node] = True  # Update the execution state in the self.node_executed dictionary

        # Log the state of the self.streams dictionary
        logger.info(f"State of self.streams after executing node {node}: {self.streams}")

        # Ensure the result is always a tuple
        if isinstance(result, MassComposition):
            return (result, None)

        return result

    def plot(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1,
                font_size=10, font_weight='bold', width=2)
        plt.show()

    def _validate_unique_names(self):
        """
        Validates the uniqueness of the step names and MassComposition names in the DAG.     Returns:
        """
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

    # def _update_mc_nodes(self):
    #     """
    #     Sets the nodes properties on the MassComposition objects to be consistent with the
    #     "soon-to-be" MCNetwork object.
    #     Returns:
    #
    #     """
    #     for edge in self.graph.edges:
    #         mc = self.graph.edges[edge]['mc']
    #         src_node, dst_node = edge
    #         mc.nodes = [src_node, dst_node]
