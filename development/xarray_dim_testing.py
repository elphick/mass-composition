"""
Testing interval coordinates
============================

We want to have a single dimension with a 2 coords, e.g. _from, _to

"""

# %%
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

# %%

# get some data

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
print(df_data.head())

# make it smaller for testing
df_data = df_data[['mass_dry', 'H2O', 'Fe', 'DHID', 'interval_from', 'interval_to']]
cols_pair: List = ['interval_from', 'interval_to']
df_data['interval'] = df_data[cols_pair].mean(axis='columns')
print(df_data.head)

# set the indexes to match the desired dimensions
dims: List = ['DHID', 'interval']
df_data = df_data.reset_index().set_index(dims)
print(df_data.head)

ds: xr.Dataset = df_data.to_xarray()
print(ds)

# set the coords
ds = ds.set_coords(cols_pair)
print(ds)

# so it seems that is all that it takes.
# it is not a problem that the interval end variables are also indexed by DHID, in fact it must be so.
# It is possible to have the same mean interval on different DHID's with different interval_from and interval_to
# In fact that is the case in the demo dataset.

# The nans visible in the interval_from coords should not concern us - not every hole has every interval

print(ds.sel(DHID='CBS02').to_dataframe())
print(ds.sel(DHID='CBS02').dropna('interval').to_dataframe())

print('done')
