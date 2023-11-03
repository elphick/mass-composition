from typing import List

import pandas as pd
import numpy as np
from elphick.mass_composition import MassComposition
from elphick.mass_composition.utils.pd_utils import column_prefixes, column_prefix_counts, weight_average, \
    composition_to_mass, mass_to_composition, recovery

# noinspection PyUnresolvedReferences
from test.fixtures import demo_data


def test_col_prefixes():
    cols: List = ['Fe', 'One_DMT', 'One_Fe', 'Two_DMT', 'Two_Fe']
    res = column_prefixes(cols)
    expected = {'One': ['One_DMT', 'One_Fe'], 'Two': ['Two_DMT', 'Two_Fe']}
    assert res == expected


def test_col_counts():
    cols: List = ['Fe', 'One_DMT', 'One_Fe', 'Two_DMT', 'Two_Fe']
    res = column_prefix_counts(cols)
    expected = {'One': 2, 'Two': 2}
    assert res == expected


def test_mass_composition_round_trip(demo_data):
    df_in: pd.DataFrame = demo_data.drop(columns=['group'])
    df_mass: pd.DataFrame = df_in.pipe(composition_to_mass, mass_wet='wet_mass', mass_dry='mass_dry')
    df_composition: pd.DataFrame = df_mass.pipe(mass_to_composition, mass_wet='wet_mass', mass_dry='mass_dry')
    pd.testing.assert_frame_equal(df_in, df_composition.drop(columns=['H2O']))


def test_weight_avg(demo_data):
    df_wtd_avg: pd.DataFrame = demo_data.pipe(weight_average, mass_wet='wet_mass',
                                              mass_dry='mass_dry')
    obj_mc: MassComposition = MassComposition(demo_data, name='test_math')
    df_expected: pd.DataFrame = obj_mc.aggregate(as_dataframe=True).reset_index(drop=True)
    assert np.all(np.isclose(df_wtd_avg.values, df_expected.values))


def test_recovery(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4, name_1='Split-1', name_2='Split-2')

    df_recovery: pd.DataFrame = obj_mc_1.data.to_dataframe().pipe(recovery, df_ref=obj_mc.data.to_dataframe())
    df_expected: pd.DataFrame = obj_mc_1.compare(obj_mc, comparisons='recovery', as_dataframe=True,
                                                 explicit_names=False)
    pd.testing.assert_frame_equal(df_recovery, df_expected)
