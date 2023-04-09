"""
Resampling Interval Data
========================

Interval (or fractional) data is common in metallurgy and mineral processing.  Samples are sized using sieves
in a laboratory and each resultant fraction is often assayed to determine chemical composition.
The typical nomenclature is of the interval edges is size_retained, size passing - any particle within an interval
or fraction was retained by the lower sieve size, but passed the sieve size above it.

"""
import logging

import numpy as np
import pandas as pd
import plotly
import xarray as xr

from elphick.mass_composition import MassComposition
from elphick.mass_composition.utils.interp import interp_monotonic
from test.data.sample_data import size_by_assay

# %%
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    )

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame
# We create this object as 1D based on the pandas index

df_data: pd.DataFrame = size_by_assay()
df_data

# %%
#
# The size index is of the Interval type, maintaining the fractional information.

mc_size: MassComposition = MassComposition(df_data, name='Sample')
mc_size.data.to_dataframe()

# %%

mc_size.aggregate()

# %%
#
# We'll first take an overall look at the dataset using the parallel plot.

fig = mc_size.plot_parallel(color='Al2O3')
fig

# %%
#
# Size distributions are often plotted in the cumulative form, first we'll plot the intervals

fig = mc_size.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                             cumulative=False)
fig

# %%
#
# Cumulative passing

fig = mc_size.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                             cumulative=True, direction='ascending')
fig

# %%
#
# Cumulative retained

fig = mc_size.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                             cumulative=True, direction='descending')
fig

# %%
#
# Now we can resample

xr_ds: xr.Dataset = mc_size.data

# define the new coordinates
right_edges = pd.arrays.IntervalArray(xr_ds['size'].data).right
new_coords = np.round(np.geomspace(right_edges.min(), right_edges.max(), 50), 8)

xr_upsampled: xr.Dataset = interp_monotonic(xr_ds, coords={'size': new_coords}, include_original_coords=True)
mc_upsampled: MassComposition = MassComposition(xr_upsampled.to_dataframe(), name='Upsampled Sample')

# %%

fig = mc_upsampled.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'], cumulative=False)
fig

# %%

fig = mc_upsampled.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                                  cumulative=True, direction='ascending')
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

# %%

fig = mc_upsampled.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                                  cumulative=True, direction='descending')
fig

# %%
#
# Validate the head grade against the original sample

pd.testing.assert_frame_equal(mc_size.aggregate().reset_index(drop=True),
                              mc_upsampled.aggregate().reset_index(drop=True))

# TODO: Validate the grade of the up-sampled sample grouped by the original intervals.
#  This will validate that mass has been preserved within the original fractions.
