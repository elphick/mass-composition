from typing import Dict, List, Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from elphick.mass_composition import MassComposition
import networkx as nx
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap

from elphick.mass_composition.mc_node import MCNode, NodeType


class MCNetwork(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    @classmethod
    def from_streams(cls, streams: List[MassComposition], name: Optional[str] = 'Flowsheet') -> 'MCNetwork':
        bunch_of_edges: List = []
        for stream in streams:
            if stream.data.nodes is None:
                raise KeyError(f'Stream {stream.name} does not have the node property set')
            nodes = stream.data.nodes

            # add the objects to the edges
            bunch_of_edges.append((nodes[0], nodes[1], {'mc': stream}))

        graph = cls(name=name)
        graph.add_edges_from(bunch_of_edges)
        d_node_objects: Dict = {}
        for node in graph.nodes:
            d_node_objects[node] = MCNode(node_id=int(node))

        nx.set_node_attributes(graph, d_node_objects, 'mc')

        for node in graph.nodes:
            d_node_objects[node].inputs = [graph.get_edge_data(e[0], e[1])['mc'] for e in graph.in_edges(node)]
            d_node_objects[node].outputs = [graph.get_edge_data(e[0], e[1])['mc'] for e in graph.out_edges(node)]

        # nx.set_node_attributes(graph, d_node_balance, 'data')
        # bal_vals: List = [n['data'].balanced for n in graph.nodes if n is not None]
        # graph['balanced'] = all(bal_vals)
        return graph

    @property
    def balanced(self) -> bool:
        bal_vals: List = [self.nodes[n]['mc'].balanced for n in self.nodes]
        bal_vals = [bv for bv in bal_vals if bv is not None]
        return all(bal_vals)

    def report(self) -> pd.DataFrame:
        """Summary Report

        Returns:

        """
        chunks: List[pd.DataFrame] = []
        for n, nbrs in self.adj.items():
            for nbr, eattr in nbrs.items():
                chunks.append(eattr['mc'].aggregate().assign(stream=eattr['mc'].name))
        rpt: pd.DataFrame = pd.concat(chunks, axis='index').set_index('stream')
        return rpt

    def get_node_input_outputs(self, node) -> Tuple:
        in_edges = self.in_edges(node)
        in_mc = [self.get_edge_data(oe[0], oe[1])['mc'] for oe in in_edges]
        out_edges = self.out_edges(node)
        out_mc = [self.get_edge_data(oe[0], oe[1])['mc'] for oe in out_edges]
        return in_mc, out_mc

    def plot_network(self) -> plt.Figure:

        hf, ax = plt.subplots()
        # TODO: add muti-partite layout to provide left to right layout
        pos = nx.spring_layout(self, seed=1234)

        edge_labels: Dict = {}
        edge_colors: List = []
        node_colors: List = []

        for node1, node2, data in self.edges(data=True):
            edge_labels[(node1, node2)] = data['mc'].name
            # TODO: add colors by edge balance status, once defined
            edge_colors.append('gray')

        for n in self.nodes:
            if self.nodes[n]['mc'].node_type == NodeType.BALANCE:
                if self.nodes[n]['mc'].balanced:
                    node_colors.append('green')
                else:
                    node_colors.append('red')
            else:
                node_colors.append('gray')

        nx.draw(self, pos=pos, ax=ax, with_labels=True, font_weight='bold',
                node_color=node_colors, edge_color=edge_colors)

        nx.draw_networkx_edge_labels(self, pos=pos, ax=ax, edge_labels=edge_labels, font_color='black')

        ax.set_title(f"{self.name}\nBalanced: {self.balanced}")
        return hf

    def plot_sankey(self,
                    width_var: str = 'mass_wet',
                    color_var: Optional[str] = None,
                    edge_colormap: Optional[str] = 'viridis'
                    ) -> Figure:

        rpt: pd.DataFrame = self.report()

        if color_var is not None:
            import seaborn as sns
            cmap = sns.color_palette(edge_colormap, as_cmap=True)
            rpt: pd.DataFrame = self.report()
            v_min = float(rpt[color_var].min())
            v_max = float(rpt[color_var].max())

        if isinstance(list(self.nodes)[0], int):
            labels = [str(n) for n in list(self.nodes)]
        else:
            labels = list(self.nodes)

        # run the report for the hover data
        d_custom_data: Dict = self._rpt_to_html(df=rpt)

        source: List = []
        target: List = []
        value: List = []
        edge_custom_data = []
        edge_color: List = []
        edge_labels: List = []
        for u, v, data in self.edges(data=True):
            edge_labels.append(data['mc'].name)
            source.append(u)
            target.append(v)
            value.append(float(data['mc'].aggregate()[width_var]))
            edge_custom_data.append(d_custom_data[data['mc'].name])

            if color_var is not None:
                val: float = float(data['mc'].aggregate()[color_var])
                str_color: str = f'rgba({self._color_from_float(v_min, v_max, val, cmap)})'.replace('[', '').replace(
                    ']', '')
                edge_color.append(str_color)
            else:
                edge_color: Optional[str] = None

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="blue",
                customdata=labels
            ),
            link=dict(
                source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=target,
                value=value,
                color=edge_color,
                label=edge_labels,  # over-written by hover template
                customdata=edge_custom_data,
                hovertemplate='<b><i>%{label}</i></b><br />Source: %{source.customdata}<br />'
                              'Target: %{target.customdata}<br />%{customdata}'
            ))])

        fig.update_layout(title_text=self.name, font_size=10)
        return fig

    @staticmethod
    def _rpt_to_html(df: pd.DataFrame) -> Dict:
        custom_data: Dict = {}
        for i, row in df.iterrows():
            str_data: str = '<br />'
            for k, v in dict(row).items():
                str_data += f'{k}: {round(v, 2)}<br />'
            custom_data[i] = str_data
        return custom_data

    @staticmethod
    def _color_from_float(vmin: float, vmax: float, val: float, cmap: ListedColormap) -> Tuple[
        float, float, float]:
        color_index: int = int((val - vmin) / ((vmax - vmin) / 256.0))
        color_index = min(max(0, color_index), 255)
        return cmap.colors[color_index]
