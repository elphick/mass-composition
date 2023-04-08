"""
Basic usage
===========

A simple example demonstrating how to use mass-composition.

Design notes:
Once data is loaded chemical analyte names and H2O will conform to the internal standard.

"""

import pandas as pd

from elphick.mass_composition import MassComposition
from test.data import sample_data

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
df_data

# %%
#
# Construct a MassComposition object

obj_mc: MassComposition = MassComposition(df_data)
print(obj_mc)

# %%
#
# Demonstrate the aggregate function
# -----------------------------------
#
# i.e. weight average of the dataset, a.k.a. head grade

obj_mc.aggregate()

# %%
obj_mc.aggregate(as_dataframe=False)

# %%
#
# Aggregate by a group variable

obj_mc.aggregate(group_var='group')

# %%
#
# Filter the object
# -----------------
#
# Filter with a criteria, just like pandas

obj_mc.query(queries={'index': 'Fe>58'}).aggregate()

