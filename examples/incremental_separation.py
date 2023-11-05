"""
Incremental Separation
======================

This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
(of the remaining fractions) and recalculating the mass-composition and recovery of the portion remaining.
This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

The returned data can be used to assess the amenability of a fractionated sample (in the dimension of the sample).

This concept is only applicable in a single dimension where the mass-composition (sample) object is an interval index.

The example will use a dataset that represents a sample fractionated by size.

"""
import logging

import pandas as pd
import plotly
import xarray as xr

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import sample_data, size_by_assay

# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
# %%#
# Create the sample
# -----------------
#
# The sample is a MassComposition object

df_data: pd.DataFrame = size_by_assay()
df_data

# %%
# The size index is of the Interval type, maintaining the fractional information.

mc_size: MassComposition = MassComposition(df_data, name='Sample')
mc_size.data.to_dataframe

# %%
# Incrementally Separate
# ----------------------
#
# Leverage the method to return the incremental perfect separation in the size dimension.
# Here we will "de-slime" by discarding the smallest (lowest) sizes incrementally.

results: pd.DataFrame = mc_size.ideal_incremental_separation(discard_from="lowest")
results

# %%
# Repeat the process but by discarding the coarser sizes.

results_2: pd.DataFrame = mc_size.ideal_incremental_separation(discard_from="highest")
results_2

# %%
# Plot Grade-Recovery
# -------------------

fig = mc_size.plot_grade_recovery(target_analyte='Fe')
fig.update_layout(width=900)
# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# Plot Amenability
# ----------------

fig = mc_size.plot_amenability(target_analyte='Fe')
fig.update_layout(width=900)
fig
