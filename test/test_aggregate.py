import pandas as pd

# noinspection PyUnresolvedReferences
from test.data.fixtures import demo_data
from elphick.mass_composition import MassComposition


def test_aggregation(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data)
    df_agg: pd.DataFrame = obj_mc.aggregate(as_dataframe=True)

    df_expected: pd.DataFrame = pd.DataFrame({'mass_wet': [300.0], 'mass_dry': [260.0],
                                              'H2O': [13.333333],
                                              'Fe': [59.0],
                                              'SiO2': [3.515385],
                                              'Al2O3': [1.873077],
                                              'LOI': [4.0]}, index=pd.Index(['unnamed'], name='name'))
    pd.testing.assert_frame_equal(df_expected, df_agg)
