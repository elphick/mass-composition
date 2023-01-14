"""
Basic usage
===========

A simple example demonstrating how to use mass-composition.
"""

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
from mass_composition.mass_composition import MassComposition

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
# Validate the round trip by converting composition to mass and back to composition

xr_ds_mass = xr_ds.mc.composition_to_mass()
xr_ds_chem = xr_ds_mass.mc.mass_to_composition()

xarray.tests.assert_allclose(xr_ds, xr_ds_chem)

# %%
#
# Demonstrate the mc aggregate function
# -------------------------------------
#
# i.e. weight average of the dataset, a.k.a. head grade

print(xr_ds.mc.aggregate())

# %%
#
# Convert to a pandas DataFrame

print(xr_ds.mc.aggregate().to_dataframe())

# %%
#
# Aggregate by a group variable

print(xr_ds.mc.aggregate(group_var='group').to_dataframe())





