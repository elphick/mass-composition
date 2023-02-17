"""
Weighted Statistics
===================

Demonstrate with some real data
"""

# %%
from pathlib import Path

import pandas as pd

from plotly.graph_objs import Figure
import xarray as xr

from elphick.mass_composition import MassComposition

# %%
#
# Create a MassComposition object
# -------------------------------
# We get some demo data in the form of a pandas DataFrame

filepath: Path = Path('../test/data/iron_ore_sample_data_A072391.csv')
name: str = filepath.stem.split('_')[-1]
df_data: pd.DataFrame = pd.read_csv(filepath, index_col='index')
print(df_data.shape)
print(df_data.head())

obj_mc: MassComposition = MassComposition(df_data, name=name)

fig: Figure = obj_mc.plot_parallel(color='Fe')
fig

# %%
#
# Calculate the weighted std dev across holes

df_mean_holes: pd.DataFrame = obj_mc.aggregate(group_var='DHID')

# TODO: weighted SD coming soon...

df_mean: pd.DataFrame = obj_mc.aggregate()

# weighted stdev

res: xr.Dataset = obj_mc.data.weighted(obj_mc.data['mass_dry']).std()

if len(res.dims) == 0:
    res = res.expand_dims('index')

df_wtd_std: pd.DataFrame = res.to_dataframe()

# weighted mean - compare with aggregate

res: xr.Dataset = obj_mc.data.weighted(obj_mc.data['mass_dry']).mean()

if len(res.dims) == 0:
    res = res.expand_dims('index')

df_wtd_mean: pd.DataFrame = res.to_dataframe()

# grades match - excellent.
