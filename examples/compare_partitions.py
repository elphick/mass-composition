"""
Compare Partition Separations
=============================

Demonstrate splitting a single sample and applying two different partition models along a dimension.

We will compare an ideal (perfect) partition with that with the partition model of Napier-Munn (1998)

This demonstrates why using only ore characterisation data to infer plant performance is dangerous.
The robust method uses ore characterisation and process characterisation (partition) data

The two cases tested are:

* IDEAL - Perfect partition, represents using ore characterisation data only
* SIM - Simulated reality with Napier-Munn partition, represents using ore characterisation and
  process characterisation data.

..  note::
    The Ep parameter injected to the Napier-Munn partition is speculative - for illustrative purposes only.

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
df_data.rename(columns={'mass_pct': 'mass_dry'}, inplace=True)
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
# Split the Sample
# ----------------

mc_ideal_feed, mc_sim_feed = mc_size.split(0.5, 'ideal feed', 'sim feed')

# %%
# Apply the Partitions
# --------------------
#
# We partially initialise the two partitions
# The dim argument is added to inform the split method which dimension to apply the function/split to

part_ideal = partial(perfect, d50=150, dim='size')
part_sim = partial(napier_munn, d50=150, ep=100, dim='size')

# %%
#
# Separate the object using the defined partitions

mc_ideal_coarse, mc_ideal_fine = mc_ideal_feed.partition(definition=part_ideal,
                                                         name_1='ideal_coarse', name_2='ideal_fine')
mc_sim_coarse, mc_sim_fine = mc_sim_feed.partition(definition=part_sim, name_1='sim_coarse', name_2='sim_fine')

# manually tweak the nodes (to overcome a known bug with node allocation on large networks)
mc_ideal_coarse._data.mc.nodes = [2, 7]
mc_ideal_fine._data.mc.nodes = [2, 8]

mcn: MCNetwork = MCNetwork().from_streams([mc_size, mc_ideal_feed, mc_sim_feed,
                                           mc_ideal_coarse, mc_ideal_fine,
                                           mc_sim_coarse, mc_sim_fine])

fig = mcn.table_plot(table_pos='left',
                     sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

# %%
# ..  note::
#     The mass split and grades are different as shown in the table above.
#     More work reviewing recovery of both cases would be prudent.
#     This is illustrative only but demonstrates why using ore characterisation and ignoring
#     process characterisation to capture the real world process inefficiencies is dangerous.
