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

import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
import plotly

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets import datasets
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition.flowsheet import Flowsheet
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
# The dataset contains size x assay, plus size x density x assay data.  We'll drop the size x assay data to leave the
# sink / float data.

# df_sink_float: pd.DataFrame = df_data.query('density_lo != np.nan and density_hi != np.nan')
# df_sink_float

