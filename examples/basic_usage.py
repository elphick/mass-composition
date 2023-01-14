"""
Basic usage
===========

A simple example demonstrating how to use mass-composition.
"""

import xarray as xr
import xarray.tests
import pandas as pd

from mass_composition.data.sample_data import sample_data
from mass_composition.mass_composition import MassComposition
import mass_composition.mcxarray

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Construct a MassComposition object and standardise the chemistry variables

obj_mc: MassComposition = MassComposition(df_data)
obj_mc.convert_chem_to_symbols()
print(obj_mc)

# %%
#
# Validate the round trip by converting composition to mass and back to composition

xr_ds_mass: xr.Dataset = obj_mc.data.mc.composition_to_mass()
xr_ds_chem: xr.Dataset = xr_ds_mass.mc.mass_to_composition()

xarray.tests.assert_allclose(obj_mc.data, xr_ds_chem)

# %%
#
# Demonstrate the aggregate function
# -----------------------------------
#
# i.e. weight average of the dataset, a.k.a. head grade

print(obj_mc.aggregate())

# %%
#
# Convert to a pandas DataFrame

print(obj_mc.aggregate().to_dataframe())

# %%
#
# Aggregate by a group variable

print(obj_mc.aggregate(group_var='group').to_dataframe())
