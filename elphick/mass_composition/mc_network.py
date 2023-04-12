from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns

from plotly.subplots import make_subplots

from elphick.mass_composition import MassComposition
from elphick.mass_composition.mc_node import MCNode, NodeType


class MCNetwork(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    @classmethod
    def from_streams(cls, streams: List[MassComposition], name: Optional[str] = 'Flowsheet') -> 'MCNetwork':
        """Class method from a list of objects

        Args:
            streams: List of MassComposition objects
            name: name of the network

        Returns:

        """
        bunch_of_edges: List = []
        for stream in streams:
            if stream.nodes is None:
                raise KeyError(f'Stream {stream.name} does not have the node property set')
            nodes = stream.nodes

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

        graph = nx.convert_node_labels_to_integers(graph)
        # update the temporary nodes on the mc object property to match the renumbered integers
        for node1, node2, data in graph.edges(data=True):
            data['mc'].nodes = [node1, node2]

        return graph

    @property
    def balanced(self) -> bool:
        bal_vals: List = [self.nodes[n]['mc'].balanced for n in self.nodes]
        bal_vals = [bv for bv in bal_vals if bv is not None]
        return all(bal_vals)

    def get_edge_by_name(self, name: str) -> MassComposition:
        """Get the MC object from the network by its name

        Args:
            name: The string name of the MassComposition object stored on an edge in the network.

        Returns:

        """

        res: Optional[MassComposition] = None
        for u, v, a in self.edges(data=True):
            if a['mc'].name == name:
                res = a['mc']

        if not res:
            raise ValueError(f"The specified name: {name} is not found on the network.")

        return res

    def report(self) -> pd.DataFrame:
        """Summary Report

        Total Mass and weight averaged composition
        Returns:

        """
        chunks: List[pd.DataFrame] = []
        for n, nbrs in self.adj.items():
            for nbr, eattr in nbrs.items():
                chunks.append(eattr['mc'].aggregate().assign(name=eattr['mc'].name))
        rpt: pd.DataFrame = pd.concat(chunks, axis='index').set_index('name')
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
        title = f"{self.name}<br>Balanced: {self.balanced}"

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=title,
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        ),
                        )
        return fig

    def plot_sankey(self,
                    width_var: str = 'mass_wet',
                    color_var: Optional[str] = None,
                    edge_colormap: Optional[str] = 'copper_r',
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    ) -> go.Figure:
        """Plot the Network as a sankey

        Args:
            width_var: The variable that determines the sankey width
            color_var: The optional variable that determines the sankey edge color
            edge_colormap: The optional colormap.  Used with color_var.
            vmin: The value that maps to the minimum color
            vmax: The value that maps to the maximum color

        Returns:

        """
        d_sankey: Dict = self._generate_sankey_args(color_var, edge_colormap, width_var, vmin, vmax)
        node, link = self._get_sankey_node_link_dicts(d_sankey)
        fig = go.Figure(data=[go.Sankey(node=node, link=link)])
        title = f"{self.name}<br>Balanced: {self.balanced}"
        fig.update_layout(title_text=title, font_size=10)
        return fig

    def table_plot(self,
                   plot_type: str = 'sankey',
                   table_pos: str = 'left',
                   table_area: float = 0.4,
                   table_header_color: str = 'cornflowerblue',
                   table_odd_color: str = 'whitesmoke',
                   table_even_color: str = 'lightgray',
                   sankey_width_var: str = 'mass_wet',
                   sankey_color_var: Optional[str] = None,
                   sankey_edge_colormap: Optional[str] = 'copper_r',
                   sankey_vmin: Optional[float] = None,
                   sankey_vmax: Optional[float] = None
                   ) -> go.Figure:
        """Plot with table of edge averages

        Args:
            plot_type: The type of plot ['sankey', 'network']
            table_pos: Position of the table ['left', 'right', 'top', 'bottom']
            table_area: The proportion of width or height to allocate to the table [0, 1]
            table_header_color: Color of the table header
            table_odd_color: Color of the odd table rows
            table_even_color: Color of the even table rows
            sankey_width_var: If plot_type is sankey, the variable that determines the sankey width
            sankey_color_var: If plot_type is sankey, the optional variable that determines the sankey edge color
            sankey_edge_colormap: If plot_type is sankey, the optional colormap.  Used with sankey_color_var.
            sankey_vmin: The value that maps to the minimum color
            sankey_vmax: The value that maps to the maximum color

        Returns:

        """

        valid_plot_types: List[str] = ['sankey', 'network']
        if plot_type not in valid_plot_types:
            raise ValueError(f'The supplied plot_type is not in {valid_plot_types}')

        valid_table_pos: List[str] = ['top', 'bottom', 'left', 'right']
        if table_pos not in valid_table_pos:
            raise ValueError(f'The supplied table_pos is not in {valid_table_pos}')

        d_subplot, d_table, d_plot = self._get_position_kwargs(table_pos, table_area, plot_type)

        fig = make_subplots(**d_subplot, print_grid=False)

        df = self.report().reset_index()
        fmt: List[str] = ["%s"] + [".1f", ".1f"] + [".2f"] * (len(df.columns) - 3)
        column_widths = [2] + [1] * (len(df.columns) - 1)

        fig.add_table(
            header=dict(values=list(df.columns),
                        fill_color=table_header_color,
                        align='center',
                        font=dict(color='black', size=12)),
            columnwidth=column_widths,
            cells=dict(values=df.transpose().values.tolist(),
                       align='left', format=fmt,
                       fill_color=[[table_odd_color if i % 2 == 0 else table_even_color for i in range(len(df))] * len(
                           df.columns)]),
            **d_table)

        if plot_type == 'sankey':
            d_sankey: Dict = self._generate_sankey_args(sankey_color_var,
                                                        sankey_edge_colormap,
                                                        sankey_width_var,
                                                        sankey_vmin,
                                                        sankey_vmax)
            node, link = self._get_sankey_node_link_dicts(d_sankey)
            fig.add_trace(go.Sankey(node=node, link=link), **d_plot)

        elif plot_type == 'network':
            pos = nx.spring_layout(self, seed=1234)

            edge_trace, node_trace = self._get_scatter_node_edges(pos)
            fig.add_traces(data=[edge_trace, node_trace], **d_plot)

            fig.update_layout(showlegend=False, hovermode='closest',
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)'
                              )

        title = f"{self.name}<br>Balanced: {self.balanced}"
        fig.update_layout(title_text=title, font_size=12)

        return fig

    @staticmethod
    def _get_position_kwargs(table_pos, table_area, plot_type):
        """Helper to manage location dependencies

        Args:
            table_pos: position of the table: left|right|top|bottom
            table_width: fraction of the plot to assign to the table [0, 1]

        Returns:

        """
        name_type_map: Dict = {'sankey': 'sankey', 'network': 'xy'}
        specs = [[{"type": 'table'}, {"type": name_type_map[plot_type]}]]

        widths: Optional[List[float]] = [table_area, 1.0 - table_area]
        subplot_kwargs: Dict = {'rows': 1, 'cols': 2, 'specs': specs}
        table_kwargs: Dict = {'row': 1, 'col': 1}
        plot_kwargs: Dict = {'row': 1, 'col': 2}

        if table_pos == 'left':
            subplot_kwargs['column_widths'] = widths
        elif table_pos == 'right':
            subplot_kwargs['column_widths'] = widths[::-1]
            subplot_kwargs['specs'] = [[{"type": name_type_map[plot_type]}, {"type": 'table'}]]
            table_kwargs['col'] = 2
            plot_kwargs['col'] = 1
        else:
            subplot_kwargs['rows'] = 2
            subplot_kwargs['cols'] = 1
            table_kwargs['col'] = 1
            plot_kwargs['col'] = 1
            if table_pos == 'top':
                subplot_kwargs['row_heights'] = widths
                subplot_kwargs['specs'] = [[{"type": 'table'}], [{"type": name_type_map[plot_type]}]]
                table_kwargs['row'] = 1
                plot_kwargs['row'] = 2
            elif table_pos == 'bottom':
                subplot_kwargs['row_heights'] = widths[::-1]
                subplot_kwargs['specs'] = [[{"type": name_type_map[plot_type]}], [{"type": 'table'}]]
                table_kwargs['row'] = 2
                plot_kwargs['row'] = 1

        if plot_type == 'network':  # different arguments for different plots
            plot_kwargs = {f'{k}s': v for k, v in plot_kwargs.items()}

        return subplot_kwargs, table_kwargs, plot_kwargs

    def _generate_sankey_args(self, color_var, edge_colormap, width_var, v_min, v_max):
        rpt: pd.DataFrame = self.report()
        if color_var is not None:
            cmap = sns.color_palette(edge_colormap, as_cmap=True)
            rpt: pd.DataFrame = self.report()
            if not v_min:
                v_min = np.floor(rpt[color_var].min())
            if not v_max:
                v_max = np.ceil(rpt[color_var].max())
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
        node_colors: List = []

        for n in self.nodes:
            if self.nodes[n]['mc'].node_type == NodeType.BALANCE:
                if self.nodes[n]['mc'].balanced:
                    node_colors.append('green')
                else:
                    node_colors.append('red')
            else:
                node_colors.append('blue')

        for u, v, data in self.edges(data=True):
            edge_labels.append(data['mc'].name)
            source.append(u)
            target.append(v)
            value.append(float(data['mc'].aggregate()[width_var]))
            edge_custom_data.append(d_custom_data[data['mc'].name])

            if color_var is not None:
                val: float = float(data['mc'].aggregate()[color_var])
                str_color: str = f'rgba{self._color_from_float(v_min, v_max, val, cmap)}'
                edge_color.append(str_color)
            else:
                edge_color: Optional[str] = None

        d_sankey: Dict = {'node_color': node_colors,
                          'edge_color': edge_color,
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
            color=d_sankey['node_color'],
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
    def _color_from_float(vmin: float, vmax: float, val: float,
                          cmap: Union[ListedColormap, LinearSegmentedColormap]) -> Tuple[float, float, float]:
        if isinstance(cmap, ListedColormap):
            color_index: int = int((val - vmin) / ((vmax - vmin) / 256.0))
            color_index = min(max(0, color_index), 255)
            color_rgba = tuple(cmap.colors[color_index])
        elif isinstance(cmap, LinearSegmentedColormap):
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            r, g, b, a = m.to_rgba(val, bytes=True)
            color_rgba = int(r), int(g), int(b), int(a)
        else:
            NotImplementedError("Unrecognised colormap type")

        return color_rgba

    def query(self, mc_name: str, queries: Dict) -> 'MCNetwork':
        """Query/filter across the network

        The queries provided will be applied to the MassComposition object in the network with the mc_name.
        The indexes for that result are then used to filter the other edges of the network.

        Args:
            mc_name: The name of the MassComposition object in the network to which the first filter to be applied.
            queries: The query or queries to apply to the object with mc_name.

        Returns:

        """

        mc_obj_ref: MassComposition = self.get_edge_by_name(mc_name).query(queries=queries)
        # TODO: This construct limits us to filtering along a single dimension only
        coord: str = list(queries.keys())[0]
        index = mc_obj_ref.data[coord]

        # iterate through all other objects on the edges and filter them to the same indexes
        mc_objects: List[MassComposition] = []
        for u, v, a in self.edges(data=True):
            if a['mc'].name == mc_name:
                mc_objects.append(mc_obj_ref)
            else:
                mc_obj: MassComposition = self.get_edge_by_name(a['mc'].name)
                mc_obj._data = mc_obj._data.sel({coord: index.values})
                mc_objects.append(mc_obj)

        res: MCNetwork = MCNetwork.from_streams(mc_objects)

        return res
