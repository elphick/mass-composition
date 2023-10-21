"""
Datasets and Sample Data
========================

An example that demonstrates the two places to access data.

1. elphick.mass_composition.datasets.datasets where you will find methods like demo_data, and
2. elphick.mass_composition.datasets.sample_data where there are methods like sample_data.

Why?

Since we are after open, real data, that needs to be appropriately referenced - that is 1) (demo_data).  Some light
cleaning may have been applied, but essentially those datasets will be largely structured
as they are sourced.

The second location 2) (sample_data) is where the methods may load from 1) and apply transformations that prepare
data ready for injection into the package, thereby simplifying examples.

This approach retains the integrity of the original datasets, but creates sample_data that simplifies examples.

"""

import pandas as pd

from elphick.mass_composition.datasets import datasets
from elphick.mass_composition.datasets import sample_data

# %%
#
# Datasets
# --------
#
# We load some datasets

df_ds1: pd.DataFrame = datasets.load_size_by_assay()
df_ds1

# %%
#
# Sample Data
# -----------
#
# We load some sample data.

df_sd1: pd.DataFrame = sample_data.size_by_assay()
df_sd1
