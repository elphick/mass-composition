"""
Grade Tonnage
=============

The "Grade Tonnage" curve is often used to characterise an entire deposit.
It is a cumulative view that presents the "mass (tonnes) and head grade" with increasing cut-off grade.

They are useful in comparing deposits.

"""
from pathlib import Path

import pandas as pd
import plotly

from elphick.mass_composition import MassComposition
import xarray as xr

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

filepath: Path = Path('../test/data/iron_ore_sample_data_A072391.csv')
name: str = filepath.stem.split('_')[-1]
df_data: pd.DataFrame = pd.read_csv(filepath, index_col='index')
df_data.drop(columns=['Na2O', 'CaO', 'MnO', 'TiO2', 'P', 'K2O', 'MgO'], inplace=True)

print(df_data.shape)
df_data.head()

# %%
obj_mc: MassComposition = MassComposition(df_data, name=name)

# %%
#
# Demonstrate the aggregate function
# -----------------------------------
#
# i.e. weight average of the dataset, a.k.a. head grade

print(obj_mc.aggregate())
print(obj_mc.aggregate(as_dataframe=False))

res: xr.Dataset = obj_mc.binned_mass_composition(cutoff_var='Fe', bin_width=1.0, cumulative=True, as_dataframe=False)
res

# %%
#
# Tabular Grade bin data
# ----------------------

res: pd.DataFrame = obj_mc.binned_mass_composition(cutoff_var='Fe', bin_width=1.0, cumulative=True,
                                                   direction='ascending', as_dataframe=True)
res

# %%
res: pd.DataFrame = obj_mc.binned_mass_composition(cutoff_var='Fe', bin_width=1.0, cumulative=True,
                                                   direction='descending', as_dataframe=True)
res

# %%
#
# Plot the Grade Bins
# -------------------

fig = obj_mc.plot_bins(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                       cutoff_var='Fe',
                       bin_width=1.0,
                       cumulative=True,
                       direction='descending')
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery
