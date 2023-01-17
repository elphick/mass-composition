from typing import List, Optional, Dict, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from stream import Stream


class Flowsheet(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    @classmethod
    def from_streams(cls, streams: List[Stream], name: Optional[str] = 'Flowsheet') -> 'Flowsheet':
        bunch_of_edges: List = []
        for stream in streams:
            if stream.in_node is None:
                raise KeyError(f'Stream {stream.name} does not have the in_node property set')
            if stream.out_node is None:
                raise KeyError(f'Stream {stream.name} does not have the out_node property set')

            # add the objects to the edges
            bunch_of_edges.append((stream.in_node, stream.out_node, {'mc': stream}))

        graph = cls(name=name)
        graph.add_edges_from(bunch_of_edges)

        return graph

    def report(self) -> pd.DataFrame:
        """Summary Report

        Returns:

        """
        chunks: List[pd.DataFrame] = []
        for n, nbrs in self.adj.items():
            for nbr, eattr in nbrs.items():
                chunks.append(eattr['mc'].aggregate().to_dataframe().assign(stream=eattr['mc'].name))
        rpt: pd.DataFrame = pd.concat(chunks, axis='index').set_index('stream')
        return rpt

    def plot_sankey(self,
                    width_var: str = 'mass_wet',
                    color_var: Optional[str] = None,
                    edge_colormap: Optional[str] = 'viridis'
                    ):

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
        fig.show()

    @staticmethod
    def _rpt_to_html(df: pd.DataFrame) -> Dict:
        custom_data: Dict = {}
        for i, row in df.iterrows():
            str_data: str = '<br />'
            for k, v in dict(row).items():
                str_data += f'{k}: {round(v, 2)}<br />'
            custom_data[i] = str_data
        return custom_data

    def _color_from_float(self, vmin: float, vmax: float, val: float, cmap: ListedColormap) -> Tuple[
        float, float, float]:
        color_index: int = int((val - vmin) / ((vmax - vmin) / 256.0))
        color_index = min(max(0, color_index), 255)
        return cmap.colors[color_index]


if __name__ == '__main__':
    df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
    print(df_data.shape)
    print(df_data.head())

    obj_stream: Stream = Stream.from_dataframe(df_data, name='stream 1')
    obj_stream.in_node = 0
    obj_stream.out_node = 1
    print(obj_stream)

    print(obj_stream.aggregate())

    stream_1, stream_2 = obj_stream.split(fraction=0.6)
    stream_1.in_node = 1
    stream_1.out_node = 2
    stream_2.in_node = 1
    stream_2.out_node = 3

    stream_list = [obj_stream, stream_1, stream_2]

    obj_fs: Flowsheet = Flowsheet.from_streams(name='Test Flowsheet', streams=stream_list)

    nx.draw(obj_fs, with_labels=True, font_weight='bold')
    plt.show()

    df_report: pd.DataFrame = obj_fs.report()
    print(df_report)

    obj_fs.plot_sankey()

    print('done')
