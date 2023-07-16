"""
Filtering
=========

Filtering is often required to reduce the data of a MassComposition object to a specific subset of interest.

Both individual objects can be filtered, as can multiple objects contained within a MCNetwork object.

"""

import pandas as pd

from elphick.mass_composition import MassComposition
from elphick.mass_composition.mc_network import MCNetwork
from elphick.mass_composition.demo_data.sample_data import sample_data

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

obj_mc: MassComposition = MassComposition(df_data, name='demo')
obj_mc.data.to_dataframe()

# %%
#
# Filtering Single Objects
# ------------------------
#
# One of the most common subsets is one that contains records above a particular grade.
# The method used to filter is called query, for consistency with the xarray and pandas methods that execute the same.

obj_1: MassComposition = obj_mc.query({'index': 'Fe>58'})
obj_1.data.to_dataframe()

# %%
# Notice that the record with an Fe value below 58 has been removed.

# %%
#
# Filtering Multiple Objects
# --------------------------
#
# Multiple objects can be loaded into a MCNetwork.  We'll make a small network to demonstrate.

obj_one, obj_two = obj_mc.split(fraction=0.6, name_1='one', name_2='two')

mcn: MCNetwork = MCNetwork.from_streams([obj_mc, obj_one, obj_two], name='Network')

# %%
# The weighted mean mass-composition of each object/edge/stream in the network can be reported out with the
# report method.

mcn.report()

# %%
# Now we'll filter as we did before, though we must specify which object the query criteria is to be applied to.

mcn.query(mc_name='demo', queries={'index': 'Fe>58'}).report()
