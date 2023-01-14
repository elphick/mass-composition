"""
Math Operations
===============

Demonstrate splitting and math operations that preserve the mass balance of components.
"""

# %%

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
# Split the original Dataset and return the complement of the split fraction.
# Splitting does not modify the absolute grade of the input.

xr_ds_split, xr_ds_comp = xr_ds.mc.split(fraction=0.1)
print(xr_ds_split)

# %%
print(xr_ds_comp)

# %%
#
# Add the split and complement parts using the mc.add method

xr_ds_sum: xr.Dataset = xr_ds_split.mc.add(xr_ds_comp)
print(xr_ds_sum)

# %%
#
# Confirm the sum of the splits is materially equivalent to the starting object.

xarray.tests.assert_allclose(xr_ds, xr_ds_sum)

# %%
#
# Add finally add and then subtract the split portion to the original object, and check the output.

xr_ds_sum: xr.Dataset = xr_ds.mc.add(xr_ds_split)
xr_ds_minus: xr.Dataset = xr_ds_sum.mc.minus(xr_ds_split)
xarray.tests.assert_allclose(xr_ds_minus, xr_ds)
print(xr_ds_minus)
