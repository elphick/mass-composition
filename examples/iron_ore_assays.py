"""
Iron Ore Assays
===============

Demonstrate with some real data
"""

# %%

import xarray as xr
import pandas as pd

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
print(df_data.shape)
print(df_data.head())

xr_ds: xr.Dataset = xr.Dataset(df_data)
xr_ds = xr_ds.mc.convert_chem_to_symbols()
print(xr_ds)

fig = xr_ds.mc.plot_parallel(color='Fe')
fig.show()

print('done')
