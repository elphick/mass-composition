import numpy as np
import pandas as pd
import xarray.tests
import xarray as xr

from elphick.mass_composition.mc_network import MCNetwork
# noinspection PyUnresolvedReferences
from test.fixtures import demo_data
from elphick.mass_composition import MassComposition


def test_query(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='demo')
    obj_1: MassComposition = obj_mc.query({'index': 'Fe>58'})

    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [90., 110.],
                                                        'mass_dry': [80., 90.],
                                                        'H2O': [11.11111, 18.181818],
                                                        'Fe': [59., 61.],
                                                        'SiO2': [3.1, 2.2],
                                                        'Al2O3': [1.7, 0.9],
                                                        'LOI': [4.0, 3.0],
                                                        'group': ['grp_1', 'grp_2'],
                                                        'index': [1, 2]})
    df_expected.set_index('index', inplace=True)

    pd.testing.assert_frame_equal(df_expected, obj_1.data.to_dataframe())


def test_query_network(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='demo')
    obj_one, obj_two = obj_mc.split(fraction=0.6, name_1='one', name_2='two')
    mcn: MCNetwork = MCNetwork.from_streams([obj_mc, obj_one, obj_two], name='Network')

    df_report: pd.DataFrame = mcn.query(mc_name='demo', queries={'index': 'Fe>58'}).report()

    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [200., 120.,  80.],
                                                        'mass_dry': [170., 102.,  68.],
                                                        'H2O': [15., 15., 15.],
                                                        'Fe': [60.05882353, 60.05882353, 60.05882353],
                                                        'SiO2': [2.62352941, 2.62352941, 2.62352941],
                                                        'Al2O3': [1.27647059, 1.27647059, 1.27647059],
                                                        'LOI': [3.47058824, 3.47058824, 3.47058824],
                                                        'name': ['demo', 'one', 'two']})
    df_expected.set_index('name', inplace=True)

    pd.testing.assert_frame_equal(df_expected, df_report)

