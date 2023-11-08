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
from functools import partial

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import napier_munn
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
# Create an interpolator from the data.  As a Callable, the spline can be used to split a MassComposition object.

da = np.linspace(0.01, df_partition.index.right.max(), num=500)
spline_partition = PchipInterpolator(x=df_partition.sort_index()['da'], y=df_partition.sort_index()['PN'])
pn_extracted = spline_partition(da)

# %%
# Plot the extracted data, and the spline on the input partition curve to visually validate.

pn_original = part_cyclone(da) / 100

fig = go.Figure(go.Scatter(x=da, y=pn_original, name='Input Partition', line=dict(width=5, color='DarkSlateGrey')))
fig.add_trace(go.Scatter(x=df_partition['da'], y=df_partition['PN'], name='Extracted Partition Data', mode='markers',
                         marker=dict(size=12, color='red', line=dict(width=2, color='DarkSlateGrey'))))
fig.add_trace(
    go.Scatter(x=da, y=pn_extracted, name='Extracted Partition Curve', line=dict(width=2, color='red', dash='dash')))

fig.update_xaxes(type="log")
fig.update_layout(title='Partition Round Trip Check', xaxis_title='da', yaxis_title='PN', yaxis_range=[0, 1.05])

# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# There are differences in the re-created partition at the coarser sizes.  It would be interesting to
# investigate if up-sampling is advance of partition generation would reduce this difference.  Alternatively,
# the parameteric partition function of the form defined by the `napier_munn` form could be fitted.

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
