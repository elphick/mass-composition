"""
Splitting and Partitioning Objects
==================================

Demonstrate splitting by mass and partitioning along dimensions.
"""

import plotly
import xarray.tests
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial

from elphick.mass_composition.mc_network import MCNetwork
from elphick.mass_composition.utils.partition import perfect
from test.data.sample_data import sample_data, size_by_assay
from elphick.mass_composition import MassComposition

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
df_data.head()

# %%

# Construct a MassComposition object and standardise the chemistry variables

obj_mc: MassComposition = MassComposition(df_data, name='test data')
print(obj_mc)
obj_mc.aggregate()

# %%
#
# Split by Mass
# -------------
#
# Split the original Dataset by mass and return both the defined split and complement objects.
# Splitting with a float, splits mass but does not modify the absolute grade of the input.

obj_mc_split, obj_mc_comp = obj_mc.split(fraction=0.1)
print(obj_mc_split)
obj_mc_split.aggregate()

# %%
obj_mc_comp.aggregate()

# %%
#
# Add the split and complement parts using the mc.add method

obj_mc_sum: MassComposition = obj_mc_split + obj_mc_comp
obj_mc_sum.aggregate()

# %%
#
# Confirm the sum of the splits is materially equivalent to the starting object.

xarray.tests.assert_allclose(obj_mc.data, obj_mc_sum.data)

# %%
#
# Partition by Dimension
# ----------------------
#
# In mineral processing, unit operations separate particles based on one (or more) property.
# Consider a sizing screen, separating by size. The characteristics of such separations can be defined by a function.
# The function is often called a partition curve or tromp curve.
#
# First we load a size x assay dataset, with size fractions as the index.
# While the data is multidimensional (considering all assays), from the MassComposition object definition it is a 1D
# dataset (indexed by size).

df_data: pd.DataFrame = size_by_assay()
df_data

# %%
# Create the object

mc_size: MassComposition = MassComposition(df_data, name='size sample')
print(mc_size)
mc_size.aggregate()

# %%
# Visualise the data

fig = mc_size.plot_parallel(color='Fe')
fig

# %%
#
# We partially initialise the perfect partition function
# The dim argument is added to inform the split method which dimension to apply the function/split to

partition = partial(perfect, d50=0.150, dim='size')

# %%
#
# Separate the object using the defined partition

mc_coarse, mc_fine = mc_size.partition(definition=partition)
mc_coarse.name = 'coarse'
mc_fine.name = 'fine'

mcn: MCNetwork = MCNetwork().from_streams([mc_size, mc_coarse, mc_fine])

hf = mcn.plot()
plt.show()

# %%

fig = mcn.table_plot(table_pos='top',
                     sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery



