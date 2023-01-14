"""
Iron Ore Assays
===============

Demonstrate with some real data
"""

# %%
from typing import List, Dict

import xarray as xr
import xarray.tests
import pandas as pd

from mass_composition.data.sample_data import sample_data
import mass_composition.mcxarray

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame
from mass_composition.utils.components import is_compositional

df_data: pd.DataFrame = pd.read_csv('Met.csv')
df_data.dropna(inplace=True)
print(df_data.shape)
print(df_data.head())

cols_components: List[str] = is_compositional(list(df_data.columns))

mass_dry = df_data['Dry Weight Lump (kg)'].apply(lambda x: x.replace('..', '.')).astype('float') + df_data[
    'Dry Weight Fines (kg)'].astype('float')
df_data['mass_dry'] = mass_dry
df_data.rename(
    columns={'Moisture (%)': 'H2O', 'Bulk_Hole_No': 'DHID', 'From (m)': 'interval_from', 'To (m)': 'interval_to'},
    inplace=True)

extra_cols: List[str] = ['DHID', 'interval_from', 'interval_to']
mass_comp_data: pd.DataFrame = df_data[['mass_dry', 'H2O'] + cols_components + extra_cols]
mass_comp_data.sort_values(['DHID', 'interval_from'], inplace=True)
mass_comp_data.index.name='index'

mass_comp_data.to_csv('iron_ore_sample_data.csv')
mass_comp_data.to_csv('../sample_data/iron_ore_sample_data.csv')

print('done')
