from typing import List, Optional

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

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
        chunks: List[pd.DataFrame] = []
        for n, nbrs in self.adj.items():
            for nbr, eattr in nbrs.items():
                chunks.append(eattr['mc'].aggregate().to_dataframe().assign(stream=eattr['mc'].name))
        rpt: pd.DataFrame = pd.concat(chunks, axis='index').set_index('stream')
        return rpt


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

    print('done')
