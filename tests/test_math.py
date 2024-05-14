import pandas as pd
import xarray.tests
import xarray as xr

# noinspection PyUnresolvedReferences
from tests.fixtures import demo_data
from elphick.mass_composition import MassComposition


def test_add_xr(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='test_math')
    xr_ds: xr.Dataset = obj_mc._data
    xr_ds_split, xr_ds_comp = xr_ds.mc.split(fraction=0.1)

    # Add the split and complement parts using the mc.add method
    xr_ds_sum: xr.Dataset = xr_ds_split.mc.add(xr_ds_comp)
    # Confirm the sum of the splits is materially equivalent to the starting object.

    xarray.tests.assert_allclose(xr_ds, xr_ds_sum)


def test_add_mc(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='test_math')
    obj_mc_split, obj_mc_comp = obj_mc.split(fraction=0.1)

    # Add the split and complement parts using the mc.add method
    obj_mc_sum: MassComposition = obj_mc_split.add(obj_mc_comp)

    # Confirm the sum of the splits is materially equivalent to the starting object.
    xarray.tests.assert_allclose(obj_mc.data, obj_mc_sum.data)


def test_add_to_self(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='test_math')

    # Add the split and complement parts using the mc.add method
    obj_mc_sum: MassComposition = obj_mc.add(obj_mc, name='new_name')

    # Confirm the Mass is twice the starting mass, but grades the same.
    df_expected: pd.DataFrame = pd.DataFrame.from_dict({'mass_wet': [200., 180., 220.],
                                                        'mass_dry': [180., 160., 180.],
                                                        'H2O': [10., 11.11111, 18.181818],
                                                        'Fe': [57., 59., 61.],
                                                        'SiO2': [5.2, 3.1, 2.2],
                                                        'Al2O3': [3.0, 1.7, 0.9],
                                                        'LOI': [5.0, 4.0, 3.0],
                                                        'group': ['grp_1', 'grp_1', 'grp_2']})
    df_expected.index.name = 'index'
    pd.testing.assert_frame_equal(df_expected, obj_mc_sum.data.to_dataframe())
