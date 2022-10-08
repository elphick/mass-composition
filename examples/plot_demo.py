"""
Plot Demo
=========

Demonstrating the mass-composition view (plot) methods.
"""

import xarray as xr
import xarray.tests
import pandas as pd

from mcxarray.data.sample_data import sample_data
import mcxarray.mcxarray

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Construct a Xarray Dataset and standardise the chemistry variables

xr_ds: xr.Dataset = xr.Dataset(df_data)
xr_ds = xr_ds.mc.convert_chem_to_symbols()
print(xr_ds)

# %%
#
# Create an interactive parallel plot

fig = xr_ds.mc.plot_parallel()
fig

# %%
#
# Create an interactive parallel plot with only the components

fig2 = xr_ds.mc.plot_parallel(composition_only=True)
fig2

# %%
#
# Create a parallel plot with color

fig3 = xr_ds.mc.plot_parallel(color='group')
fig3

# %%
#
# Create a ternary diagram for 3 composition variables

fig4 = xr_ds.mc.plot_ternary(variables=['Fe', 'SiO2', 'Al2O3'], color='group')
fig4