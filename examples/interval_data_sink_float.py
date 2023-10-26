"""
Interval Data - Sink Float
==========================

Intervals are encountered in Metallurgy, aka fractions,
e.g. size fractions.  In that case the typical nomenclature is size_retained, size passing, since the data
originates from a sieve stack.

The Sink Float metallurgical test splits/fractionates samples by density.  The density fraction is often conducted by
size fraction, resulting in 2D fractionation (interval) data.

"""

import logging
from functools import partial
from pathlib import Path
import plotly.express as px

import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
import plotly

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets import datasets
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import napier_munn

# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
#
# Load Data
# ---------
#
# We load some real data.

df_data: pd.DataFrame = datasets.load_nordic_iron_ore_sink_float()
df_data

# %%
# The dataset contains size x assay, plus size x density x assay data.  We'll create a dataframe and MassComposition
# object for each.

df_density: pd.DataFrame = df_data.dropna(subset=['density_lo', 'density_hi'], how='all')
# fill the nans with plausible values.
df_density['size_passing'].fillna(2.0, inplace=True)
df_density['density_lo'].fillna(1.0, inplace=True)
df_density['density_hi'].fillna(6.0, inplace=True)

df_merged = df_data.merge(df_density, how="left", left_index=True, right_index=True, indicator=True).query(
    "_merge == 'left_only'")
df_size: pd.DataFrame = df_data.loc[df_merged.index, :].drop(columns=['density_lo', 'density_hi'])
# fill the nans with plausible values.
df_size['size_passing'].fillna(2.0, inplace=True)
df_size

# %%
df_density.set_index(['size_retained', 'size_passing', 'density_lo', 'density_hi'], inplace=True)
mc_density: MassComposition = MassComposition(data=df_density, name='density')
mc_density.aggregate()

# %%
df_size.set_index(['size_retained', 'size_passing'], inplace=True)
mc_size: MassComposition = MassComposition(data=df_size, name='size')
mc_size.aggregate()

# %%
# The head grades of each sample are different - there are two reasons for this, the first being that the
# sub 0.04 mm material was not sink / float tested (a little impractical).  So how does the grade compare when
# that is accounted for by removing it from the size dataset?

MassComposition(data=df_size.query('size_retained > 0')).aggregate()

# %%
# The grades still do not match, but this is expected, since the tests were conducted on different subsamples.
# This is a common problem - which do we believe?  That is a question for another day.

# %%
#
# Visualise deportment
# --------------------
#
# First we view a 1D sample.

fig = mc_size.plot_deportment(variables=['Fe'])
fig