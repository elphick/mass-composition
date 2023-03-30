from typing import Dict, List, Optional, Tuple, Iterable, Set, Union

import pandas as pd
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

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

    def plot(self) -> plt.Figure:
        """Plot the network with matplotlib

        Returns:

        """

        hf, ax = plt.subplots()
        # TODO: add multi-partite layout to provide left to right layout
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

    def plot_network(self) -> go.Figure:
        """Plot the network with plotly

        Returns:

        """
        pos = nx.spring_layout(self, seed=1234)

        edge_trace, node_trace = self._get_scatter_node_edges(pos)

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=self.name,
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def _get_scatter_node_edges(self, pos):
        edge_x = []
        edge_y = []
        for edge in self.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')
        node_x = []
        node_y = []
        for node in self.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                size=20,
                line_width=2))
        return edge_trace, node_trace

    def table_plot(self,
                   named_plots: Union[Tuple, List] = ('sankey', 'table'),
                   sankey_width_var: str = 'mass_wet',
                   sankey_color_var: Optional[str] = None,
                   sankey_edge_colormap: Optional[str] = 'viridis'
                   ) -> Figure:
        """Plot multiple components

        Args:
            named_plots: list of any two ['table', 'sankey', 'network']

        Returns:

        """
        valid_plots: Set[str] = {'table', 'sankey', 'network'}
        invalid_plots: Set[str] = set(named_plots).difference(valid_plots)

        if len(list(named_plots)) != 2:
            raise ValueError('Only two named plots are supported')

        if len(invalid_plots) > 0:
            raise ValueError(f'The supplied named_plots are not in {valid_plots}')

        name_map: Dict = {'table': 'table', 'sankey': 'sankey', 'network': 'xy'}

        fig = make_subplots(rows=1, cols=2,
                            print_grid=False,
                            column_widths=[0.6, 0.4],
                            specs=[[{"type": name_map[named_plots[0]]},
                                    {"type": name_map[named_plots[1]]}]])

        for i, plot in enumerate(named_plots):
            if plot == 'sankey':
                d_sankey: Dict = self._generate_sankey_args(sankey_color_var,
                                                            sankey_edge_colormap,
                                                            sankey_width_var)
                node, link = self._get_sankey_node_link_dicts(d_sankey)
                fig.add_trace(go.Sankey(node=node, link=link), row=1, col=1 + i)

            elif plot == 'table':
                df = self.report().reset_index()
                fig.add_table(
                    header=dict(values=list(df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=df.transpose().values.tolist(),
                               fill_color='lavender',
                               align='left'),
                    row=1, col=1 + i)

            elif plot == 'network':

                pos = nx.spring_layout(self, seed=1234)

                edge_trace, node_trace = self._get_scatter_node_edges(pos)

                fig.add_traces(data=[edge_trace, node_trace])

        fig.update_layout(title_text=self.name, font_size=10, showlegend=False, hovermode='closest',
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        return fig

    def plot_sankey(self,
                    width_var: str = 'mass_wet',
                    color_var: Optional[str] = None,
                    edge_colormap: Optional[str] = 'viridis'
                    ) -> Figure:
        d_sankey: Dict = self._generate_sankey_args(color_var,
                                                    edge_colormap,
                                                    width_var)
        node, link = self._get_sankey_node_link_dicts(d_sankey)
        fig = go.Figure(data=[go.Sankey(node=node, link=link)])
        fig.update_layout(title_text=self.name, font_size=10)
        return fig

    def _generate_sankey_args(self, color_var, edge_colormap, width_var):
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

        d_sankey: Dict = {'edge_color': edge_color,
                          'edge_custom_data': edge_custom_data,
                          'edge_labels': edge_labels,
                          'labels': labels,
                          'source': source,
                          'target': target,
                          'value': value}

        return d_sankey

    @staticmethod
    def _get_sankey_node_link_dicts(d_sankey: Dict):
        node: Dict = dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=d_sankey['labels'],
            color="blue",
            customdata=d_sankey['labels']
        )
        link: Dict = dict(
            source=d_sankey['source'],  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target=d_sankey['target'],
            value=d_sankey['value'],
            color=d_sankey['edge_color'],
            label=d_sankey['edge_labels'],  # over-written by hover template
            customdata=d_sankey['edge_custom_data'],
            hovertemplate='<b><i>%{label}</i></b><br />Source: %{source.customdata}<br />'
                          'Target: %{target.customdata}<br />%{customdata}'
        )
        return node, link

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
