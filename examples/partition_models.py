"""
Partition Models
================

Partition models, (a.k.a. partition curves) define the separation of a unit operation / process.

In the one dimensional case, the Partition Number (PN) is represents the probability that a particle will
report to the defined reference stream.

Consider a desliming cyclone that aims to separate a slurry at 150 micron.  The reference stream is defined as
the Underflow (UF), since that is the "stream of value" in our simple example.

..  Admonition:: TODO

    Add a reference to partition curves.

"""
import numpy as np
import plotly
import pandas as pd
from functools import partial
import plotly.express as px
import plotly.graph_objects as go

from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import perfect, napier_munn
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition import MassComposition
from elphick.mass_composition.utils.pd_utils import calculate_partition

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create a mass-composition object
# --------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = size_by_assay()
mc_feed: MassComposition = MassComposition(df_data, name='size sample')
print(mc_feed)

# %%
# Define and Apply the Partition
# ------------------------------
#
# We partially initialise the partition function
# The dim argument is added to inform the split method which dimension to apply the function/split to

part_cyclone = partial(napier_munn, d50=0.150, ep=0.1, dim='size')

# %%
# Separate the object using the defined partitions.  UF = Underflow, OF = Overflow

mc_uf, mc_of = mc_feed.apply_partition(definition=part_cyclone, name_1='underflow', name_2='overflow')
mcn: MCNetwork = MCNetwork().from_streams([mc_feed, mc_uf, mc_of])

fig = mcn.table_plot(table_pos='left',
                     sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
fig

# %%
# We'll now get the partition data from the objects

df_partition: pd.DataFrame = mc_feed.calculate_partition(ref=mc_uf)
df_partition

# %%
# Plot the extracted data on the input partition curve used to generate the output streams.

da = np.linspace(0.01, df_partition.index.right.max(), num=500)
pn = part_cyclone(da) / 100

fig = go.Figure(go.Scatter(x=da, y=pn, name='Input Partition'))
fig.add_trace(go.Scatter(x=df_partition['da'], y=df_partition['PN'], name='Calculated Partition', mode='markers'))
fig.update_xaxes(type="log")
fig.update_layout(title='Partition Round Trip Check', xaxis_title='da', yaxis_title='PN')

# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# Pandas Function
# ---------------
#
# The same functionality is available in pandas

df_partition_2: pd.DataFrame = mc_feed.data.to_dataframe().pipe(calculate_partition, df_ref=mc_uf.data.to_dataframe(),
                                                                col_mass_dry='mass_dry')
df_partition_2

# %%
pd.testing.assert_frame_equal(df_partition, df_partition_2)
