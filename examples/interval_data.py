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
from matplotlib import pyplot as plt
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
# We create this object as 1D based on the pandas index

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data_A072391.csv', index_col='index')
print(df_data.head())

obj_mc: MassComposition = MassComposition(df_data,
                                          name='Drill program',
                                          mass_units='kg')
print(obj_mc)

print(obj_mc.aggregate())
print(obj_mc.aggregate('DHID'))

# %%
#
# View the data

# fig: Figure = obj_mc.plot_parallel(color='Fe')
# fig.show()


# %%
#
# We will now make a 2D dataset using DHID and the interval.
# We will first create a mean interval variable.  Then we will set the dataframe index to both variables before
# constructing the object.

print(df_data.columns)

df_data['DHID'] = df_data['DHID'].astype('category')
code, dh_id = pd.factorize(df_data['DHID'])
df_data['DH'] = code
df_data = df_data.reset_index().set_index(['DH', 'interval_from', 'interval_to'])

obj_mc_2d: MassComposition = MassComposition(df_data,
                                             name='Drill program',
                                             mass_units='kg')
# obj_mc_2d._data.assign(hole_id=dh_id)
print(obj_mc_2d)
print(obj_mc_2d.aggregate())
print(obj_mc_2d.aggregate('DHID'))

# %%

# create a pandas interval
df_data['interval'] = pd.IntervalIndex.from_arrays(left=df_data['interval_from'], right=df_data['interval_to'],
                                                   closed='left')

df_data = df_data.reset_index().set_index(['DH', 'interval'])

obj_mc_2d: MassComposition = MassComposition(df_data,
                                             name='Drill program',
                                             mass_units='kg')
# obj_mc_2d._data.assign(hole_id=dh_id)
print(obj_mc_2d)
print(obj_mc_2d.aggregate())
print(obj_mc_2d.aggregate('DHID'))

# %%
#
# View some plots
#
# First confirm the parallel plot still works
# TODO: work on the display order
# TODO - fails for DH (integer)
# fig: Figure = obj_mc_2d.plot_parallel(color='Fe')
# fig.show()

# now plot using the xarray data - take advantage of the multi-dim nature of the package

obj_mc_2d.data['Fe'].plot()
plt.show()

print('done')
