"""
Datasets and Sample Data
========================

.. Admonition:: TLDR

   **Datasets** are sourced in the public domain, largely unaltered.

   **Sample Data** is for use in mass-composition examples, often sourced from a dataset with some transformation applied.

*Why two modules?*

We are after open, real data so our examples are realistic.  We are obliged to appropriately reference so the
original *dataset* is largely structured as they are sourced, potentially with some minor adjustments as noted.

The sample_data module contains methods that often load from the dataset module and apply transformations to prepare
data ready for injection into the package.  This keeps examples simple.

This approach retains the integrity of the original datasets, but creates sample_data that simplifies examples.

The Dataset Register can be found :ref:`here <Dataset Register>`.

"""

import pandas as pd

from elphick.mass_composition.datasets import datasets
from elphick.mass_composition.datasets import sample_data

# %%
#
# Datasets
# --------
#
# We load some datasets.  This will download the file after a hash check, thereby avoiding repeated downloads
# unless the source file has been updated.

df_ds1: pd.DataFrame = datasets.load_size_by_assay()
df_ds1

# %%
# When executing this method, you can view the
# '`profile report <https://elphick.github.io/mass-composition/_static/size_by_assay.html>`_'
# for the dataset, by setting the show_report argument to True.

df_ds1: pd.DataFrame = datasets.load_size_by_assay(show_report=True)

# %%
#
# Sample Data
# -----------
#
# We load some sample data.  The method called here utilises the file downloaded in the example above.
# Some minor changes have been made to the file to simplify instantiation of a MassComposition object.

df_sd1: pd.DataFrame = sample_data.size_by_assay()
df_sd1
