from copy import deepcopy
from functools import partial
from typing import Dict

import pandas as pd

from elphick.mass_composition.mc_node import MCNode
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import perfect
# noinspection PyUnresolvedReferences
from test.fixtures import demo_data, size_assay_data, demo_size_network
from elphick.mass_composition import MassComposition


def test_sankey_plot(demo_size_network):
    mcn: MCNetwork = demo_size_network
    # test both types of colormaps
    fig = mcn.plot_sankey(color_var='Fe', edge_colormap='copper_r', vmin=50, vmax=70)
    fig
    fig2 = mcn.plot_sankey(color_var='Fe', edge_colormap='viridis')
    fig2


def test_table_plot(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    fig = mcn.table_plot()
    fig

    fig = mcn.table_plot(plot_type='network', table_pos='right', table_area=0.3)
    fig

    fig = mcn.table_plot(plot_type='network', table_pos='top', table_area=0.3)
    fig

    fig = mcn.table_plot(table_pos='bottom', table_area=0.2)
    fig


def test_to_dataframe(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    df_res: pd.DataFrame = mcn.to_dataframe()

    d_expected: Dict = {'mass_wet': {(0, 'Feed'): 100.0, (1, 'Feed'): 90.0, (2, 'Feed'): 110.0,
                                     (0, '(0.4 * Feed)'): 40.0, (1, '(0.4 * Feed)'): 36.0,
                                     (2, '(0.4 * Feed)'): 44.0, (0, '(0.6 * Feed)'): 60.0,
                                     (1, '(0.6 * Feed)'): 54.0, (2, '(0.6 * Feed)'): 66.0},
                        'mass_dry': {(0, 'Feed'): 90.0, (1, 'Feed'): 80.0, (2, 'Feed'): 90.0,
                                     (0, '(0.4 * Feed)'): 36.0, (1, '(0.4 * Feed)'): 32.0,
                                     (2, '(0.4 * Feed)'): 36.0, (0, '(0.6 * Feed)'): 54.0,
                                     (1, '(0.6 * Feed)'): 48.0, (2, '(0.6 * Feed)'): 54.0},
                        'H2O': {(0, 'Feed'): 10.0, (1, 'Feed'): 11.11111111111111,
                                (2, 'Feed'): 18.181818181818183, (0, '(0.4 * Feed)'): 10.0,
                                (1, '(0.4 * Feed)'): 11.11111111111111, (2, '(0.4 * Feed)'): 18.181818181818183,
                                (0, '(0.6 * Feed)'): 10.0, (1, '(0.6 * Feed)'): 11.11111111111111,
                                (2, '(0.6 * Feed)'): 18.181818181818183},
                        'Fe': {(0, 'Feed'): 57.0, (1, 'Feed'): 59.0, (2, 'Feed'): 61.0, (0, '(0.4 * Feed)'): 57.0,
                               (1, '(0.4 * Feed)'): 59.0, (2, '(0.4 * Feed)'): 61.0, (0, '(0.6 * Feed)'): 57.0,
                               (1, '(0.6 * Feed)'): 59.0, (2, '(0.6 * Feed)'): 61.0},
                        'SiO2': {(0, 'Feed'): 5.2, (1, 'Feed'): 3.1, (2, 'Feed'): 2.2, (0, '(0.4 * Feed)'): 5.2,
                                 (1, '(0.4 * Feed)'): 3.1, (2, '(0.4 * Feed)'): 2.2, (0, '(0.6 * Feed)'): 5.2,
                                 (1, '(0.6 * Feed)'): 3.1, (2, '(0.6 * Feed)'): 2.2},
                        'Al2O3': {(0, 'Feed'): 3.0, (1, 'Feed'): 1.7, (2, 'Feed'): 0.9,
                                  (0, '(0.4 * Feed)'): 3.0, (1, '(0.4 * Feed)'): 1.7, (2, '(0.4 * Feed)'): 0.9,
                                  (0, '(0.6 * Feed)'): 3.0, (1, '(0.6 * Feed)'): 1.7, (2, '(0.6 * Feed)'): 0.9},
                        'LOI': {(0, 'Feed'): 5.0, (1, 'Feed'): 4.0, (2, 'Feed'): 3.0, (0, '(0.4 * Feed)'): 5.0,
                                (1, '(0.4 * Feed)'): 4.0, (2, '(0.4 * Feed)'): 3.0, (0, '(0.6 * Feed)'): 5.0,
                                (1, '(0.6 * Feed)'): 4.0, (2, '(0.6 * Feed)'): 3.0},
                        'group': {(0, 'Feed'): 'grp_1', (1, 'Feed'): 'grp_1', (2, 'Feed'): 'grp_2',
                                  (0, '(0.4 * Feed)'): 'grp_1', (1, '(0.4 * Feed)'): 'grp_1',
                                  (2, '(0.4 * Feed)'): 'grp_2', (0, '(0.6 * Feed)'): 'grp_1',
                                  (1, '(0.6 * Feed)'): 'grp_1', (2, '(0.6 * Feed)'): 'grp_2'}}

    df_expected: pd.DataFrame = pd.DataFrame.from_dict(d_expected)
    df_expected.index.names = ['index', 'name']
    pd.testing.assert_frame_equal(df_expected, df_res)


def test_from_dataframe_tall(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)
    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    df_tall: pd.DataFrame = mcn.to_dataframe()

    mcn_res: MCNetwork = MCNetwork().from_dataframe(df=df_tall,
                                                    mc_name_col='name')
    df_res: pd.DataFrame = mcn_res.to_dataframe()

    pd.testing.assert_frame_equal(df_tall, df_res)


def test_from_dataframe_wide(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)
    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    df_wide: pd.DataFrame = mcn.to_dataframe().unstack('name')
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide.columns = ['_'.join(col) for col in df_wide.columns.values]

    mcn_res: MCNetwork = MCNetwork().from_dataframe(df=df_wide, mc_name_col=None)

    df_test: pd.DataFrame = mcn.to_dataframe()
    df_res: pd.DataFrame = mcn_res.to_dataframe()
    # indexes are in a different order...  So be it.
    assert set(df_res.index) == set(df_test.index)
    df_res = df_res.loc[df_test.index, :]

    pd.testing.assert_frame_equal(df_test, df_res)


def test_streams_to_dict(demo_size_network):
    mcn: MCNetwork = demo_size_network
    streams: Dict[str, MassComposition] = mcn.streams_to_dict()
    for k, v in streams.items():
        assert isinstance(v, MassComposition)
        assert k == v.name


def test_nodes_to_dict(demo_size_network):
    mcn: MCNetwork = demo_size_network
    nodes: Dict[int: MCNode] = mcn.nodes_to_dict()
    assert list(nodes.keys()) == [0, 1, 2, 3]


def test_set_node_names(demo_size_network):
    mcn: MCNetwork = demo_size_network
    mcn.set_node_names(node_names={0: 'new_feed_name'})
    assert mcn.graph.nodes[0]['mc'].node_name == 'new_feed_name'


def test_set_stream_data(demo_size_network):
    mcn: MCNetwork = demo_size_network
    streams_original: Dict = deepcopy(mcn.streams_to_dict())
    coarse: MassComposition = deepcopy(mcn.get_edge_by_name('coarse'))
    coarse.name = 'coarse_2'
    mcn.set_stream_data(stream_data={'fine': coarse})
    streams_modified: Dict = mcn.streams_to_dict()

    assert list(streams_modified.keys()) == ['size sample', 'coarse', 'coarse_2']
    df1: pd.DataFrame = mcn.get_edge_by_name('coarse').data.to_dataframe()
    df2: pd.DataFrame = mcn.get_edge_by_name('coarse_2').data.to_dataframe()
    pd.testing.assert_frame_equal(df1, df2)
