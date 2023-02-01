import pandas as pd

# noinspection PyUnresolvedReferences
from fixtures import demo_data
from elphick.mc.mass_composition import MassComposition


def test_aggregation(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data)
    df_agg: pd.DataFrame = obj_mc.aggregate(as_dataframe=True)

    df_expected: pd.DataFrame = pd.DataFrame({'mass_wet': [300.0], 'mass_dry': [260.0],
                                              'H2O': [13.333333],
                                              'Fe': [59.0],
                                              'SiO2': [3.515385],
                                              'Al2O3': [1.873077],
                                              'LOI': [4.0]}, index=[0])
    df_expected.index.name = demo_data.index.name
    pd.testing.assert_frame_equal(df_expected, df_agg)
