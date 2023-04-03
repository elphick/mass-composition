from functools import partial

import pandas as pd

from elphick.mass_composition.mc_network import MCNetwork
from elphick.mass_composition.utils.partition import perfect
# noinspection PyUnresolvedReferences
from test.data.fixtures import demo_data, size_assay_data
from elphick.mass_composition import MassComposition


def test_sankey_plot(size_assay_data):
    df_data: pd.DataFrame = size_assay_data
    df_data.rename(columns={'mass_pct': 'mass_dry'}, inplace=True)
    mc_size: MassComposition = MassComposition(df_data, name='size sample')
    partition = partial(perfect, d50=150, dim='size')
    mc_coarse, mc_fine = mc_size.partition(definition=partition)
    mc_coarse.name = 'coarse'
    mc_fine.name = 'fine'

    mcn: MCNetwork = MCNetwork().from_streams([mc_size, mc_coarse, mc_fine])
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
