import logging
from typing import List, Callable, Union, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed

from elphick.mass_composition import Stream, MassComposition

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, operation: Callable[..., Union[Stream, Tuple[Stream, Stream]]],
                 dependencies: List[str], kwargs: dict = None):
        self.operation = operation
        self.dependencies = dependencies
        self.kwargs = kwargs if kwargs is not None else {}


class DAG:
    def __init__(self, name: str = 'DAG', n_jobs=-1):
        self.name = name
        self.stream_parent_node = {}  # Map stream names to node names
        self.n_jobs = n_jobs  # Number of workers for parallel execution
        self.graph = nx.DiGraph()
        self.node_executed = {}  # Store the execution state of nodes

    # @property
    # def mass_compositions(self) -> dict[str, Stream]:
    #     """
    #     Retrieves all the Stream objects associated with the nodes in the DAG.
    #
    #     This property iterates over all the nodes in the DAG, checks if the node has been executed,
    #     and if so, retrieves the result of the node execution. If the node has not been executed,
    #     it retrieves the Stream object associated with the node in the graph.
    #
    #     Returns:
    #         dict[str, Stream]: A dictionary where the keys are node names and the values
    #         are Stream objects associated with those nodes.
    #     """
    #     mass_compositions = {}
    #     for node in self.graph.nodes:
    #         if node in self.node_executed:
    #             result = self.node_executed[node]
    #             if isinstance(result, Stream):
    #                 mass_compositions[node] = result
    #             elif isinstance(result, tuple) and all(isinstance(r, Stream) for r in result):
    #                 for r in result:
    #                     mass_compositions[r.name] = r
    #         else:
    #             # If the node is not in the results dictionary, it is a leaf node
    #             # Retrieve its result from the Stream objects associated with the node in the graph
    #             mc = self.graph.nodes[node]['mc']
    #             mass_compositions[node] = mc
    #     return mass_compositions

    @property
    def streams(self):
        """
        Retrieves all the Stream objects associated with the edges in the DAG.

        This property iterates over all the edges in the DAG and retrieves the Stream object associated with each edge.

        Returns:
            dict[str, Stream]: A dictionary where the keys are edge names and the values
            are Stream objects associated with those edges.
        """
        streams = {}
        for edge in self.graph.edges:
            strm = self.graph.edges[edge].get('mc')
            if strm is not None:
                streams[strm.name] = strm
        return streams

    @property
    def all_nodes_(self):
        """Identify all nodes in the DAG."""
        return list(self.graph.nodes)

    @staticmethod
    def input(strm: Stream) -> Stream:
        return strm

    @staticmethod
    def output(strm: Stream) -> Stream:
        return strm

    def add_input(self, name: str):
        self.graph.add_node(name, operation=DAG.input, kwargs=None, defined=True, name=name, strm=None,
                            dependencies=[])
        # Update the stream_to_node mapping
        self.stream_parent_node[name] = name

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

    def run(self, input_streams: dict):
        """
        Executes the Directed Acyclic Graph (DAG).

        This method takes a dictionary of Stream objects as input and executes the operations defined in
        the DAG.
        The execution starts from the input nodes and proceeds in a topological order, ensuring that each node
        is executed only after all its predecessors have been executed. The results of the node executions are
        stored in the `self.streams` dictionary.

        Parameters:
        input_streams (dict): A dictionary mapping node names to Stream objects. These are the initial Stream objects
                    for the input nodes of the DAG.

        Returns:
        None
        """
        logger.info("Running the DAG")  # Log the node that is being executed
        self._finalize()

        # Initialize the execution state of all nodes to False
        for node in self.graph.nodes:
            self.node_executed[node] = False

        executed_nodes = set()  # Keep track of nodes that have been executed

        while len(executed_nodes) < len(self.graph):
            # Find nodes with no predecessors that haven't been executed yet
            ready_nodes = [node for node in self.graph.nodes if
                           all(pred in executed_nodes for pred in self.graph.predecessors(node)) and
                           node not in executed_nodes]

            logger.debug(f"Ready nodes: {ready_nodes}")
            logger.debug(f"Executed nodes: {list(executed_nodes)}")
            logger.debug(f"Result streams: {self.streams}")

            if not ready_nodes:
                unexecuted_nodes = set(self.graph.nodes) - executed_nodes
                for node in unexecuted_nodes:
                    predecessors = list(self.graph.predecessors(node))
                    logger.debug(f"Node {node} is waiting for {predecessors}")

            # Create a job for each ready node
            jobs = [delayed(self.execute_node)(node, input_streams, executed_nodes) for node in ready_nodes]

            # Execute the jobs in parallel
            if jobs:
                results = Parallel(n_jobs=self.n_jobs)(jobs)
                # Filter out None values
                results = [result for result in results if result is not None]
            else:
                results = []

            # Update executed_nodes with the returned value of each job
            for i, result in enumerate(results):
                executed_nodes.add(ready_nodes[i])

    def execute_node(self, node: str, strms: dict, executed_nodes: set) -> Optional[Union[Stream, Tuple[Stream, ...]]]:
        """
        Executes a node in the DAG.

        This method takes a node and a dictionary of Stream objects. It executes the operation associated with the
        node using the Stream objects as inputs. If the node has successors and is defined, the result of the node
        execution is stored in the edges of the graph.

        Parameters:
        node (str): The name of the node to be executed.
        strms (dict): A dictionary mapping node names to Stream objects.

        Returns:
        Union[Stream, Tuple[Stream, ...]]: The result of the node execution, or None if the node is waiting for its predecessors.
        """
        logger.info(f"Executing node {node}")  # Log the node that is being executed
        operation = self.graph.nodes[node]['operation']
        kwargs = self.graph.nodes[node]['kwargs']
        defined = self.graph.nodes[node]['defined']

        logger.debug(f"State of self.streams before executing node {node}: {self.streams}")

        # Log the predecessors of the node
        predecessors = list(self.graph.predecessors(node))
        logger.debug(f"Predecessors of node {node}: {predecessors}")

        try:
            # Check if the node is an input node
            if operation == DAG.input:
                strm: Union[Stream, MassComposition] = strms[node]
                if isinstance(strm, MassComposition):
                    strm = Stream(mc=strm)
                result = operation(strm)

            # Check if the node is an output node
            elif operation == DAG.output:
                # Retrieve the Stream object from the edge between the output node and its predecessor
                predecessor = list(self.graph.predecessors(node))[0]
                strm = self.graph.edges[(predecessor, node)].get('mc')
                result = operation(strm)

            # If not an input or output, then it is a step node
            else:
                # If the node is not in the strms dictionary, it means that it needs to be created inside the DAG
                # In this case, execute the operation with the results of its dependencies as inputs
                # Check if the results of the predecessors are available

                if all(self.node_executed[dependency] for dependency in self.graph.predecessors(node)):
                    inputs = [self.graph.get_edge_data(*edge)['mc'] for edge in self.graph.in_edges(node)]
                    # If only one input stream is provided, retrieve the corresponding Stream object
                    if len(inputs) == 1:
                        inputs = inputs[0]
                        # Check if kwargs is not None before passing it to the operation
                        result = operation(inputs, **kwargs) if kwargs is not None else operation(inputs)
                    else:
                        # Ensure inputs is always an iterable
                        if isinstance(inputs, Stream):
                            inputs = [inputs]
                        # Check if kwargs is not None before passing it to the operation
                        result = operation(*inputs, **kwargs) if kwargs is not None else operation(*inputs)
                else:
                    logger.debug(f"Waiting for predecessors of node {node}")
                    return None

        except AttributeError as e:
            logger.error(f"Error while executing node {node}: {e}")
            raise

        # If the node has successors and is defined, store the result of the node execution in the edges of the graph
        if list(self.graph.successors(node)) and defined:
            if isinstance(result, tuple):
                for i, strm in enumerate(result):
                    self.graph.edges[(node, list(self.graph.successors(node))[i])]['mc'] = strm
                    logger.debug(f"Stored results for stream {strm.name}")  # Log the node for which a result was stored
            else:
                for successor in self.graph.successors(node):
                    self.graph.edges[(node, successor)]['mc'] = result
                logger.debug(f"Stored results for stream {node}")  # Log the node for which a result was stored

        # After executing the operation, update the execution state of the node
        executed_nodes.add(node)
        self.node_executed[node] = True  # Update the execution state in the self.node_executed dictionary

        # Log the state of the self.streams dictionary
        logger.debug(f"State of self.streams after executing node {node}: {self.streams}")

        # Ensure the result is always a tuple
        if isinstance(result, Stream):
            return (result, None)

        return result

    def plot(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1,
                font_size=10, font_weight='bold', width=2)
        plt.show()

    def _finalize(self):
        """
        Final checks before execution.
        """
        orphan_nodes = [node for node, degree in self.graph.degree() if degree == 0]
        if orphan_nodes:
            raise ValueError(f"Orphan nodes: {orphan_nodes} exist.  Please check your configuration.")

        # Validate unique names
        self._validate_unique_names()

    def _validate_unique_names(self):
        """
        Validates the uniqueness of the step names and Stream names in the DAG.
        """
        # Validate the uniqueness of the step names
        node_names = [name for name in self.graph.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Step names are not unique within the DAG.")

        # Validate the uniqueness of the Stream names
        strm_names = [data['mc'].name for _, data in self.graph.nodes(data=True) if
                      'mc' in data and data['mc'] is not None]
        duplicates = [name for name in strm_names if strm_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Stream names are not unique within the DAG. Duplicates: {set(duplicates)}")
