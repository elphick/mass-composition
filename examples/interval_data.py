"""
Interval Data
=============

This example adds a second dimension.  The second dimension is an interval, of the form interval_from, interval_to.
It is also known as binned data, where each 'bin' is bounded between and upper and lower limit.

An interval is relevant in geology, when analysing drill hole data.

Interval are also encountered in metallurgy, but in that discipline they are often called fractions,
e.g. size fractions.  In that case the typical nomenclature is size_retained, size passing, since the data
originates from a sieve stack.

"""
import logging
from pathlib import Path
from typing import List

import pandas as pd
from plotly.graph_objs import Figure

from elphick.mc.mass_composition.data.sample_data import sample_data
from elphick.mc.mass_composition import MassComposition

# %%
logging.basicConfig(level=logging.INFO,
                    filename=f'../logs/{Path(__file__).stem}.log',
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    )

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
df_data = df_data.reset_index().set_index(['index', 'DHID'])
print(df_data.head())

obj_mc: MassComposition = MassComposition(df_data,
                                          name='Drill program',
                                          mass_units='kg')
print(obj_mc)

# %%
#
# View the data

# fig: Figure = obj_mc.plot_parallel(color='Fe')
# fig.show()

# %%
#
# Our dataset is one dimension - the single dimension is 'DHID'.
# The dimensions (in xarray lingo) map to the indexes in a pandas DataFrame.  Since we indexed our DataFrame by
# the DHID variable, it has become a dimension in our xarray dataset.
#
# It follows then that in order to manage our intervals as a second dimension we'll need to add them to the index.

print(df_data.columns)

dim_cols: List[str] = ['index', 'DHID', 'interval_from', 'interval_to']
df_data = df_data.reset_index().set_index(dim_cols)

obj_mc: MassComposition = MassComposition(df_data,
                                          name='Drill program',
                                          mass_units='kg',
                                          dim_prefixes=['interval'])
print(obj_mc)

pass