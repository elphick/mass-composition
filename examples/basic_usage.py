"""
Basic usage
===========

A simple example demonstrating how to use mass-composition.

Design notes:
Once data is loaded chemical analyte names and H2O will conform to the internal standard.

"""

import pandas as pd

from mass_composition.data.sample_data import sample_data
from mass_composition.mass_composition import MassComposition

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame
from mass_composition.utils.transform import composition_to_mass, mass_to_composition

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Construct a MassComposition object and standardise the chemistry variables

obj_mc: MassComposition = MassComposition(df_data)
# obj_mc.convert_chem_to_symbols()
print(obj_mc)

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
