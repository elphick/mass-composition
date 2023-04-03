"""
Sankey Plots
============

Related MassComposition objects are managed as a network.

"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from elphick.mass_composition import MassComposition
from elphick.mass_composition.mc_network import MCNetwork
from test.data import sample_data

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create some MassComposition objects
# -----------------------------------
#
# Create an object, and split it to create two more objects.

df_data: pd.DataFrame = sample_data()
obj_mc: MassComposition = MassComposition(df_data, name='Feed')
obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

# %%
#
# Create a MCNetwork object
# -------------------------
#
# This requires passing an Iterable of MassComposition objects

mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])

hf = mcn.plot_network()
plt.show()

# %%
#
# Sankey Plot
# -----------
#
# A sankey represents the network with the width of each edge representing its mass.
# The optional color of the edge represents the value of the selected composition analyte.
# In this example, grades are the same, so will not add any value.

fig: Figure = mcn.plot_sankey()
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

