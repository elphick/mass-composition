import logging
from typing import List, Callable, Union, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

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

    def add_input(self, name: str) -> 'DAG':
        self.graph.add_node(name, operation=DAG.input, kwargs=None, defined=True, name=name, strm=None,
                            dependencies=[])
        # Update the stream_to_node mapping
        self.stream_parent_node[name] = name
        return self

    def add_step(self, name: str, operation: Callable, streams: List[str], kwargs: dict = None,
                 defined: bool = True) -> 'DAG':
        if name in self.all_nodes_:
            raise ValueError(f"A step with the name '{name}' already exists.")
        # Determine dependencies from the input streams
        dependencies = [self.stream_parent_node[stream] for stream in streams]
        self.graph.add_node(name, operation=operation, dependencies=dependencies, kwargs=kwargs, defined=defined)
        for stream in streams:
            # Only add the edge if no edge with the same name attribute already exists in the graph
            if not any(data.get('name') == stream for _, _, data in self.graph.edges(data=True)):
                self.graph.add_edge(self.stream_parent_node[stream], name, name=stream)
            else:
                logger.error(f"Edge with name {stream} already exists in the graph.")
                raise KeyError(f"Edge with name {stream} already exists in the graph.")
        if kwargs is not None:
            for key, value in kwargs.items():
                if key in ['name', 'name_1', 'name_2']:
                    self.stream_parent_node[value] = name
        return self

    def add_output(self, name: str, stream: str) -> 'DAG':
        parent_node = self.stream_parent_node.get(stream)
        if parent_node is None:
            logger.error(f"No parent node found for stream {stream}")
            raise ValueError(f"No parent node found for stream {stream}")
        self.graph.add_node(name, operation=DAG.output, dependencies=[stream], kwargs=None, defined=True, name=name)
        self.graph.add_edge(parent_node, name, name=stream)
        return self

    def _topological_sort(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def run(self, input_streams: dict, progress_bar: bool = True):
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
        progress_bar (bool): If True, a progress bar is displayed during the execution of the DAG.

        Returns:
        None
        """
        logger.info("Preparing the DAG")
        self._finalize()
        logger.info("Executing the DAG")

        # Initialize the execution state of all nodes to False
        for node in self.graph.nodes:
            self.node_executed[node] = False

        if progress_bar:
            # Initialise a progressbar that will count up to the number of nodes in the graph
            pbar = tqdm(total=len(self.graph.nodes), desc="Executing nodes", unit="node")

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
            jobs = [delayed(self.execute_node)(node, input_streams) for node in ready_nodes]

            # Execute the jobs in parallel
            if jobs:
                results = Parallel(n_jobs=self.n_jobs)(jobs)
                # Filter out None values
                results = [result for result in results if result is not None]
            else:
                results = []

            # Update executed_nodes and self.graph.edges with the returned values
            for node, result, updated_edges in results:
                executed_nodes.add(node)
                self.node_executed[node] = True
                for edge, strm in updated_edges.items():
                    logger.debug(f"Updating edge {edge} with stream {strm.name}")
                    self.graph.edges[edge]['mc'] = strm
                if progress_bar:
                    # update the progress bar by one step
                    pbar.set_postfix_str(f"Processed node: {node}")
                    pbar.update()
        if progress_bar:
            pbar.close()  # Close the progress bar
        logger.debug(f"DAG execution complete for the nodes: {executed_nodes}")

    def execute_node(self, node: str, strms: dict):
        """
        Executes a node in the DAG.

        This method takes a node and a dictionary of Stream objects. It executes the operation associated with the
        node using the Stream objects as inputs. If the node has successors and is defined, the result of the node
        execution is stored in the edges of the graph. Otherwise, the result is returned as is.

        Parameters:
        node (str): The name of the node to be executed.
        strms (dict): A dictionary mapping node names to Stream objects.

        Returns:
        Union[Stream, Tuple[Stream, ...]]: The result of the node execution, or None if the node is waiting for its predecessors.
        """
        logger.debug(f"Executing node {node}")  # Log the node that is being executed
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
                    if operation == Stream.add:
                        # If the operation is Stream.add, then the inputs should be passed as a tuple
                        inputs = tuple(inputs)  # supports > 2 streams
                        # don't unpack inputs for Stream.add, we unpack on the other side to iterate the sum.
                        result = operation(inputs[0], inputs[1:], **kwargs) if kwargs is not None else operation(
                            inputs[0], inputs[1:])
                    else:
                        # Check if kwargs is not None before passing it to the operation
                        result = operation(*inputs, **kwargs) if kwargs is not None else operation(*inputs)

        except AttributeError as e:
            logger.error(f"Error while executing node {node}: {e}")
            raise

        # Return the node, result, and the updated edges
        updated_edges = {}
        if list(self.graph.successors(node)) and defined:  # If the node has successors and is defined, so not outputs.
            if isinstance(result, tuple):  # Multiple outputs
                # Assign each output stream to the corresponding successor
                for strm in result:
                    # Look up the edge by the name of the output stream
                    edge = next(
                        ((n1, n2) for n1, n2, data in self.graph.edges(data=True) if data.get('name') == strm.name),
                        None)
                    if edge is not None:
                        updated_edges[edge] = strm
                    else:
                        raise ValueError(f"Edge not found for stream {strm.name}")
            else:  # Single outputs
                for successor in self.graph.successors(node):
                    updated_edges[(node, successor)] = result

        # Check the type of the result
        if isinstance(result, MassComposition):
            # If the result is a Stream object, return it as is
            return node, result, updated_edges
        else:
            # If the result is a generator or a tuple of Stream objects, convert it to a list
            return node, list(result), updated_edges

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
