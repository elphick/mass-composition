"""
Basic usage XR
==============

A simple example demonstrating how to use mass-composition.
"""

import xarray as xr
import xarray.tests
import pandas as pd

from elphick.mass_composition import MassComposition
from test.data.sample_data import sample_data
# noinspection PyUnresolvedReferences
import elphick.mass_composition.mcxarray  # keep this "unused" import - it helps

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
# Construct a MassComposition object first to create a compliant xarray object with the concrete property

xr_ds: xr.Dataset = MassComposition(data=df_data, name='test').to_xarray()
print(xr_ds.mc.data())

# %%
#
# Validate the round trip by converting composition to mass and back to composition

xr_ds_mass = xr_ds.mc.composition_to_mass()
print(xr_ds_mass.mc.data())
xr_ds_chem = xr_ds_mass.mc.mass_to_composition()
print(xr_ds_chem.mc.data())

xarray.tests.assert_allclose(xr_ds, xr_ds_chem)

# %%
#
# Demonstrate splitting an object
xr_1, xr_2 = xr_ds.mc.split(fraction=0.25)
print(xr_1.mc.data())
print(xr_2.mc.data())

# %%
#
# Demonstrate the mc aggregate function
# -------------------------------------
#
# i.e. weight average of the dataset, a.k.a. head grade

xr_ds_wtd: xr.Dataset = xr_ds.mc.aggregate()
print(xr_ds_wtd.mc.data())

# xr_ds_wtd.mc.to_dataframe(original_column_names=True)

# %%
#
# Convert to a pandas DataFrame

print(xr_ds.mc.aggregate(as_dataframe=True, original_column_names=False))
print(xr_ds.mc.aggregate(as_dataframe=True, original_column_names=True))

# %%
#
# Aggregate by a group variable

print(xr_ds.mc.aggregate(group_var='group', as_dataframe=True))

# %%
#
# Math operations - we'll go full circle again, so we can check.

xr_ds_added: xr.Dataset = xr_1.mc.add(xr_2)
print(xr_ds_added.mc.data())

xarray.tests.assert_allclose(xr_ds, xr_ds_added)

xr_ds_add_sub: xr.Dataset = xr_ds.mc.add(xr_1).mc.sub(xr_1)
print(xr_ds_added.mc.data())

xarray.tests.assert_allclose(xr_ds, xr_ds_add_sub)


print('done')
