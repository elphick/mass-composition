"""
Interval Data - Advanced
========================

Intervals are encountered in Metallurgy, aka fractions,
e.g. size fractions.  In that case the typical nomenclature is size_retained, size passing, since the data
originates from a sieve stack.

This example walks through unifying the intervals across samples to be represented in the same network.
Consider the case where the feed, oversize and undersize from a screening operation are sampled and sieved.
It is likely that the undersize size distribution will be generated with fewer sieves in the sieve stack.
If this data is not unified (made consistent) the network cannot be constructed - by unifying it we can
construct a network and check the magnitude of any imbalance.
"""

import logging
from functools import partial
from pathlib import Path

import pandas as pd
import plotly

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition.flowsheet import Flowsheet
from elphick.mass_composition.utils.partition import napier_munn

# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
#
# Create some data
# ----------------
#
# We get some demo sizing data, split it with a partition, and manually drop sieves for the undersize stream.

# We create this object as 1D based on the pandas index.

df_data: pd.DataFrame = size_by_assay()
mc_feed: MassComposition = MassComposition(df_data, name='FEED')
mc_feed.data.to_dataframe()

# %%
# We partially initialise a partition function, and split the feed stream accordingly.

partition = partial(napier_munn, d50=0.150, ep=0.05, dim='size')
mc_oversize, mc_undersize = mc_feed.apply_partition(definition=partition, name_1='OS', name_2='US')

# %%
# Drop the two size fractions from mc_fine that have near zero mass.
#
# This simulates a common situation where fines samples will likely have fewer fractions reported in the results.

df_fine: pd.DataFrame = mc_undersize.data.to_dataframe()
df_fine = df_fine.loc[df_fine.index.left < 0.5, :]

mc_undersize.set_data(df_fine)
mc_undersize.data.to_dataframe()

# %%
# Notice that the top two fractions are now missing.

# %%
#
# Unifying Indexes
# ----------------
#
# If the dataset contains a single IntervalIndex called 'size' missing coarse size fractions will be added
# automatically.
#
# That said, the remaining sizes must be consistent.  Alignment of sizes across streams/mc objects is coming soon.

fs: Flowsheet = Flowsheet().from_streams([mc_feed, mc_oversize, mc_undersize])
fig = fs.table_plot()
fig

# %%
#
# Troubleshooting the imbalance
# -----------------------------
#
# So we now have our network, but it does not balance.  Perhaps the fractions we removed to generate our test data
# contained enough mass to breach our balance threshold?  Let's dig deeper with our balance plot.

fig = fs.plot_balance(color='size')
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

# %%
# What is the balance threshold set at?

print('Node error tolerance:', fs.graph.nodes[1]['mc']._tolerance)

# %%
# That plot does not reveal the problem, so we'll resort to another report.

fs.graph.nodes[1]['mc']._balance_errors

# %%
# Let's change the node error tolerance.

fs.graph.nodes[1]['mc']._tolerance = 0.001
fig = fs.table_plot()
fig

# %%
# .. admonition:: TODO
#
#    Create a single imbalance report across the entire network.

