import pandas as pd

# noinspection PyUnresolvedReferences
from .fixtures import demo_data
from elphick.mass_composition import MassComposition
import xarray as xr


def test_names(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='name_test')
    xr_data: xr.Dataset = obj_mc._data
    df_export: pd.DataFrame = xr_data.mc.to_dataframe(original_column_names=True)
    for col in demo_data.columns:
        assert col in df_export.columns.to_list(), f'{col} is not in {demo_data.columns.to_list()}'
