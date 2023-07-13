import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
from test.fixtures import demo_data
from elphick.mass_composition import MassComposition
import xarray as xr


def test_constrain_clip_mass_tuple(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(clip_mass=(80, 90))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [90., 90., 90.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [0., 11.11111, 0.],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped)


def test_constrain_clip_mass_dict(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(clip_mass={'mass_wet': (0., 95.)})
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [95., 90., 95.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [5.2631, 11.1111, 5.2631],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped, atol=1.0e-04)


def test_constrain_clip_composition_tuple(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(clip_composition=(0., 60.))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [100., 90., 110.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [10., 11.11111, 18.181818],
                                                        'Fe': [57., 59., 60.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped)


def test_constrain_clip_composition_dict(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(clip_composition={'Fe': (0., np.inf), 'Al2O3': (1.0, 100.)})
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [100., 90., 110.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [10., 11.1111, 18.181818],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 1.0],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped, atol=1.0e-04)


def test_constrain_relative_mass_tuple(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(relative_mass=(0.0, 0.1), other=obj_mc.add(obj_mc, name='feed'))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [20., 18., 22.],
                                                        'mass_dry': [18., 16., 18.],
                                                        'H2O': [10., 11.11111, 18.181818],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped)


def test_constrain_relative_mass_dict(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(relative_mass={'mass_dry': (0., 0.45)},
                                                other=obj_mc.add(obj_mc, name='feed'))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [100., 90., 110.],
                                                        'mass_dry': [81.0, 72.0, 81.0],
                                                        'H2O': [19.0, 20.0, 26.36363636363636],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped, atol=1.0e-04)


def test_constrain_relative_composition_tuple(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(relative_composition=(0.0, 0.1), other=obj_mc.add(obj_mc, name='feed'))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [100., 90., 110.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [10., 11.1111, 18.181818],
                                                        'Fe': [10.26, 9.44, 10.98],
                                                        'SiO2': [0.9359999999999999, 0.496, 0.3960000000000001],
                                                        'Al2O3': [0.5400000000000001, 0.272, 0.16200000000000003],
                                                        'LOI': [0.9, 0.6400000000000001, 0.5400000000000001],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped)


def test_constrain_relative_composition_dict(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test_constrain')
    obj_mc2: MassComposition = obj_mc.constrain(clip_composition={'Fe': (0., np.inf), 'Al2O3': (1.0, 100.)}, other=obj_mc.add(obj_mc, name='feed'))
    df_clipped: pd.DataFrame = obj_mc2.data.to_dataframe()
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [100., 90., 110.],
                                                        'mass_dry': [90., 80., 90.],
                                                        'H2O': [10., 11.1111, 18.181818],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 1.0],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, df_clipped, atol=1.0e-04)
