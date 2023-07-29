"""
Math Operations
===============

Demonstrate splitting and math operations that preserve the mass balance of components.
"""

# %%

import xarray.tests
import pandas as pd

from elphick.mass_composition.datasets.sample_data import sample_data
from elphick.mass_composition import MassComposition

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%

# Construct a MassComposition object and standardise the chemistry variables

obj_mc: MassComposition = MassComposition(df_data)
print(obj_mc)

# %%
#
# Split the original Dataset and return the complement of the split fraction.
# Splitting does not modify the absolute grade of the input.

obj_mc_split, obj_mc_comp = obj_mc.split(fraction=0.1)
print(obj_mc_split)

# %%
print(obj_mc_comp)

# %%
#
# Add the split and complement parts using the mc.add method

obj_mc_sum: MassComposition = obj_mc_split + obj_mc_comp
print(obj_mc_sum)

# %%
#
# Confirm the sum of the splits is materially equivalent to the starting object.

xarray.tests.assert_allclose(obj_mc.data, obj_mc_sum.data)

# %%
#
# Add finally add and then subtract the split portion to the original object, and check the output.

obj_mc_sum: MassComposition = obj_mc + obj_mc_split
obj_mc_minus: MassComposition = obj_mc_sum - obj_mc_split
xarray.tests.assert_allclose(obj_mc_minus.data, obj_mc.data)
print(obj_mc_minus)


# %%
#
# Demonstrate division.

obj_mc_div: MassComposition = obj_mc_split / obj_mc
print(obj_mc_div)


# %%
#
# Math operations with rename
# The alternative syntax, methods rather than operands, allows renaming of the result object

obj_mc_sum_renamed: MassComposition = obj_mc.add(obj_mc_split, name='Summed object')
print(obj_mc_sum_renamed)

# %%
obj_mc_sub_renamed: MassComposition = obj_mc.sub(obj_mc_split, name='Subtracted object')
print(obj_mc_sum_renamed)

print('done')
