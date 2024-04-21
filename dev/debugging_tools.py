import networkx as nx


def pretty_print_graph(graph: nx.Graph):
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"  Name: {node}, Type: {type(node)}, Attributes: {data}")

    print("\nEdges:")
    for (node1, node2, data) in graph.edges(data=True):
        print(f"  Edge: ({node1}, {node2}), Type: {type((node1, node2))}, Attributes: {data}")