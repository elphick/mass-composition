"""
Network Layouts
===============

Related MassComposition objects can be represented in network using the MCNetwork object.

This example demonstrates layouts of such networks.

"""

import plotly
import xarray.tests
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial

from elphick.mass_composition.mc_network import MCNetwork
from elphick.mass_composition.utils.partition import perfect, napier_munn
from test.data.sample_data import sample_data, size_by_assay
from elphick.mass_composition import MassComposition

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create a mass-composition object
# --------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = size_by_assay()


# %%
# Create the object

mc_feed: MassComposition = MassComposition(df_data, name='feed')


# %%
# Split the Sample
# ----------------

mc_1, mc_2 = mc_feed.split(0.5, 'split 1', 'split 2')
mc_3, mc_4 = mc_1.split(0.5, 'split 3', 'split 4')
mc_5, mc_6 = mc_2.split(0.5, 'split 5', 'split 6')


# %%
#
# Create a network and plot with both orientations

mcn: MCNetwork = MCNetwork().from_streams([mc_feed,
                                           mc_1, mc_2,
                                           mc_3, mc_4, mc_5, mc_6])

fig = mcn.plot_network(orientation='horizontal')
fig

fig = mcn.plot_network(orientation='vertical')
fig

# %%
#
# The table plot also supports orientation with the network_orientation argument.

fig = mcn.table_plot(table_pos='left', plot_type='network', network_orientation='vertical')
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

