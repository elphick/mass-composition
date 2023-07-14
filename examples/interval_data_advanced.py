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

import pandas as pd
import plotly

from elphick.mass_composition import MassComposition
from elphick.mass_composition.demo_data.sample_data import size_by_assay
from elphick.mass_composition.mc_network import MCNetwork
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

# We create this object as 1D based on the pandas index

df_data: pd.DataFrame = size_by_assay()
mc_feed: MassComposition = MassComposition(df_data, name='FEED')

# We partially initialise a partition function
partition = partial(napier_munn, d50=0.150, ep=0.05, dim='size')

# Create a Network using the partition
mc_oversize, mc_undersize = mc_feed.partition(definition=partition, name_1='OS', name_2='US')
# drop the two size fractions from mc_fine that have near zero mass
df_fine: pd.DataFrame = mc_undersize.data.to_dataframe()
mc_undersize = MassComposition(df_fine.loc[df_fine.index.left < 0.5, :], name='US')

# %%
#
# Demonstrate the index problem
# -----------------------------

try:
    mcn: MCNetwork = MCNetwork().from_streams([mc_feed, mc_oversize, mc_undersize])
except KeyError as ex:
    print(ex)

# %%
# We get a key error indicating the problem.

# %%
# Workaround
# ----------
#
# So the task now is to add missing indexes with zero mass
# Seems the simplest way is via pandas merge and replace nans with zeros

df_streams: pd.DataFrame = pd.concat([s.data.to_dataframe().assign(stream=s.name) for s in [mc_feed, mc_oversize, mc_undersize]])
id_vars = df_streams.columns
df_streams_full = df_streams.pivot(columns=['stream'])

# %%
# .. admonition:: TODO
#
#    Require a check here to ensure we're only adding records outside the original range.
#    Alternatively, new records in between existing records require interpolation to preserve the mass balance.

df_streams_full = df_streams_full.fillna(0).stack(level=-1).reset_index('stream')
df_streams_full

# %%
# Try to recreate the network now that our indexes align
# We can do this directly from our tall dataframe

mcn: MCNetwork = MCNetwork().from_dataframe(df_streams_full, mc_name_col='stream')
fig = mcn.table_plot()
fig

# %%
# Oh, we need to now manually define the relationships between the streams
#
# This is too verbose - will look at allowing more succinct syntax.

mc_in: MassComposition = mcn.get_edge_by_name('FEED')
mc_os: MassComposition = mcn.get_edge_by_name('OS').set_parent(mcn.get_edge_by_name('FEED'))
mc_us: MassComposition = mcn.get_edge_by_name('US').set_parent(mcn.get_edge_by_name('FEED'))
mcn: MCNetwork = MCNetwork().from_streams([mc_in, mc_os, mc_us])

fig = mcn.table_plot()
fig

# %%
#
# Troubleshooting the im-balance
# ------------------------------
#
# So we now have our network, but it does not balance.  Perhaps the fractions we removed to generate our test data
# contained enough mass to breach our balance threshold?  Let's dig deeper with our balance plot.

fig = mcn.plot_balance(color='size')
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

# %%
# That plot does not reveal the problem, so we'll resort to another report.

mcn.imbalance_report(node=1)

# %%
# The imbalance is now identified in the report shown separately in the browser.

# %%
# .. admonition:: TODO
#
#    Create a single imbalance report across the entire network.
#    Improve the report format.
