import pandas as pd
import xarray as xr

from elphick.mass_composition import MassComposition
# noinspection PyUnresolvedReferences
from test.fixtures import demo_data, size_assay_data, demo_size_network, script_loc


def test_compare_single(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4, name_1='Split-1', name_2='Split-2')

    d_expected = {'Split-1_mass_wet_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_mass_dry_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_H2O_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_Fe_rec_Feed': {0: 0.4, 1: 0.39999999999999997, 2: 0.4},
                  'Split-1_SiO2_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.39999999999999997},
                  'Split-1_Al2O3_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.39999999999999997},
                  'Split-1_LOI_rec_Feed': {0: 0.4, 1: 0.39999999999999997, 2: 0.4}}
    df_expected: pd.DataFrame = pd.DataFrame.from_dict(d_expected)

    # xarray
    ds_res: xr.Dataset = obj_mc_1.compare(obj_mc, comparisons='recovery', as_dataframe=False)
    df_expected.index.names = ['index']
    pd.testing.assert_frame_equal(df_expected, ds_res.to_dataframe())

    # pandas
    df_res: pd.DataFrame = obj_mc_1.compare(obj_mc, comparisons='recovery', as_dataframe=True)
    df_expected.index.names = ['index']
    pd.testing.assert_frame_equal(df_expected, df_res)


def test_compare_multiple(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4, name_1='Split-1', name_2='Split-2')

    ds_res: xr.Dataset = obj_mc_1.compare(obj_mc, comparisons=['recovery', 'difference'], as_dataframe=False)
    d_expected = {'Split-1_mass_wet_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_mass_dry_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_H2O_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.4},
                  'Split-1_Fe_rec_Feed': {0: 0.4, 1: 0.39999999999999997, 2: 0.4},
                  'Split-1_SiO2_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.39999999999999997},
                  'Split-1_Al2O3_rec_Feed': {0: 0.4, 1: 0.4, 2: 0.39999999999999997},
                  'Split-1_LOI_rec_Feed': {0: 0.4, 1: 0.39999999999999997, 2: 0.4},
                  'Split-1_mass_wet_diff_Feed': {0: -60.0, 1: -54.0, 2: -66.0},
                  'Split-1_mass_dry_diff_Feed': {0: -54.0, 1: -48.0, 2: -54.0},
                  'Split-1_H2O_diff_Feed': {0: 0.0, 1: 0.0, 2: 0.0},
                  'Split-1_Fe_diff_Feed': {0: 0.0, 1: 0.0, 2: 0.0},
                  'Split-1_SiO2_diff_Feed': {0: 0.0, 1: 0.0, 2: 0.0},
                  'Split-1_Al2O3_diff_Feed': {0: 0.0, 1: 0.0, 2: 0.0},
                  'Split-1_LOI_diff_Feed': {0: 0.0, 1: 0.0, 2: 0.0}}
    df_expected: pd.DataFrame = pd.DataFrame.from_dict(d_expected)

    # xarray
    ds_res: xr.Dataset = obj_mc_1.compare(obj_mc, comparisons=['recovery', 'difference'], as_dataframe=False)
    df_expected.index.names = ['index']
    pd.testing.assert_frame_equal(df_expected, ds_res.to_dataframe())

    # pandas
    df_res: pd.DataFrame = obj_mc_1.compare(obj_mc, comparisons=['recovery', 'difference'], as_dataframe=True)
    df_expected.index.names = ['index']
    pd.testing.assert_frame_equal(df_expected, df_res)
