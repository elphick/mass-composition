"""
Simplify a Network
==================

There are times when a simplified view of a complex network is required, to provide a summary of the overall
mass-balance. This can be achieved by collapsing the network into a single node, which represents the system
internals. This is useful for high-level reporting and visualisation.

In this example, we will demonstrate how to simplify a network using the `to_simple` method.

"""

import plotly
import pandas as pd
from functools import partial

from elphick.mass_composition.flowsheet import Flowsheet
from elphick.mass_composition.utils.partition import perfect, napier_munn
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition import MassComposition

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create a network
# ----------------
#
# We create the same network as in the [Compare Partition Separations](#compare-partition-separations) example.

mc_size: MassComposition = MassComposition(size_by_assay(), name='size sample')
mc_ideal_feed, mc_sim_feed = mc_size.split(0.5, 'ideal feed', 'sim feed')
part_ideal = partial(perfect, d50=0.150, dim='size')
part_sim = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
# Separate the object using the defined partitions
mc_ideal_coarse, mc_ideal_fine = mc_ideal_feed.split_by_partition(partition_definition=part_ideal,
                                                                  name_1='ideal_coarse', name_2='ideal_fine')
mc_sim_coarse, mc_sim_fine = mc_sim_feed.split_by_partition(partition_definition=part_sim, name_1='sim_coarse',
                                                            name_2='sim_fine')

fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_ideal_feed, mc_sim_feed,
                                          mc_ideal_coarse, mc_ideal_fine,
                                          mc_sim_coarse, mc_sim_fine])

fig = fs.table_plot(table_pos='left',
                    sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
fig

# %%
# Simplify the Network
# --------------------

fs_simple = fs.to_simple(node_name='system')

fig = fs_simple.table_plot(table_pos='left',
                           sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery
