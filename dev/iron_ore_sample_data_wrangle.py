"""
Iron Ore Assays
===============

Demonstrate with some real data
"""

# %%
from typing import List

import pandas as pd

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame
from elphick.mass_composition import is_compositional

df_data: pd.DataFrame = pd.read_csv('../test/data/A072391_met.csv')
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
mass_comp_data: pd.DataFrame = df_data[['mass_dry', 'H2O'] + cols_components + extra_cols].copy()
mass_comp_data.sort_values(['DHID', 'interval_from'], inplace=True)
mass_comp_data.index.name = 'index'

mass_comp_data.to_csv('../test/data/iron_ore_sample_data_A072391.csv')

# %%
#
# Wrangle to create 3D data

df_collars: pd.DataFrame = pd.read_csv('../test/data/A072391_collars.csv')
df_collars.dropna(how='all', axis='index', inplace=True)
df_collars.drop(columns='Unnamed: 0', inplace=True)

# add the collar data to the mass_comp_data data
# The DHID does not match across the datasets - we'll make an assumption - it may be wrong!
df_collars['DHID'] = df_collars['HOLEID'].apply(lambda x: x[0:2] + 'S' + x[-2::])

df_res: pd.DataFrame = pd.merge(left=mass_comp_data, right=df_collars.iloc[0: 99][['DHID', 'EAST', 'NORTH', 'RL']],
                                left_on='DHID', right_on='DHID')
df_res.rename(columns={'EAST': 'x', 'NORTH': 'y'}, inplace=True)
df_res['z_lo'] = df_res['RL'] - df_res['interval_from']
df_res['z_hi'] = df_res['RL'] - df_res['interval_to']
df_res['z'] = df_res[['z_lo', 'z_hi']].mean(axis='columns')
df_res.drop(columns=['z_lo', 'z_hi', 'RL'], inplace=True)
df_res.index.name = 'index'

df_res.to_csv('../test/data/iron_ore_sample_data_xyz_A072391.csv')
