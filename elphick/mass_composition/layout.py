from typing import Dict

import networkx as nx
import numpy as np
from networkx import DiGraph, multipartite_layout


def digraph_linear_layout(g, orientation: str = "vertical", scale: float = -1.0):
    """Position nodes of a digraph in layers of straight lines.

    Parameters
    ----------
    g : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    orientation : string (default='vertical')

    scale : number (default: 1)
        Scale factor for positions.


    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = digraph_linear_layout(g)

    Notes
    -----
    Intended for use with DiGraphs with a single degree 1 node with an out-edge

    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    g.nodes[0]['_dist'] = 0
    for x_dist in range(1, len(g.nodes) + 1):
        # get the x position from the tree "depth" / distance from the source node.
        nodes_at_x_dist: dict = nx.descendants_at_distance(g, 0, x_dist)
        if not nodes_at_x_dist:
            break
        else:
            # add the distance to the graph node to enable calling the multipartite layout
            for node in nodes_at_x_dist:
                g.nodes[node]['_dist'] = x_dist

    if orientation == 'vertical':
        orientation = 'horizontal'
    elif orientation == 'horizontal':
        orientation = 'vertical'
        scale = -scale
    else:
        raise ValueError("orientation argument not in 'vertical'|'horizontal'")

    pos = multipartite_layout(g, subset_key="_dist", align=orientation, scale=scale)

    for node in g.nodes:
        g.nodes[node].pop('_dist')

    return pos
