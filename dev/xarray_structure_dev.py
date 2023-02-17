"""
Testing interval coordinates
============================

We want to have a single dimension with a 2 coords, e.g. _from, _to

"""

# %%
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

# %%

# get some data

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
print(df_data.head())

# make it smaller for testing
df_data = df_data[['mass_dry', 'H2O', 'Fe', 'DHID', 'interval_from', 'interval_to']]
df_data2 = df_data.copy()

# start with a single dim - the df index

cols_pair: List = ['interval_from', 'interval_to']
da: xr.DataArray = xr.DataArray(name='interval',
                                data=df_data[cols_pair].mean(axis='columns'),
                                dims=['index'],
                                coords={'index': df_data.index,
                                        'interval_from': (['index'], df_data['interval_from']),
                                        'interval_to': (['index'], df_data['interval_to'])})

print(da)

ds: xr.Dataset = df_data[[col for col in df_data.columns if col not in cols_pair]].to_xarray()
print(ds)

ds_merge: xr.Dataset = xr.merge([ds, da.to_dataset()])
print(ds_merge)

# that looks good - we have the index as a coord dimension, and the interval_from and interval_to coordinates are
# not dimensions

# move some vars to coords - useful for the extra/attr vars we have that are not mass or composition
ds_merge: xr.Dataset = ds_merge.set_coords(['DHID', 'interval'])
print(ds_merge)

# %% Try a 2D version

df2d: pd.DataFrame = df_data2
cols_pair: List = ['interval_from', 'interval_to']
df2d['interval'] = df2d[cols_pair].mean(axis='columns')
df2d = df2d.reset_index().set_index(['DHID', 'interval'])
print(df2d.head())

# so far so good
ds2d: xr.Dataset = df2d.to_xarray()
print(ds2d)

df_intervals: pd.DataFrame = df2d[['interval_from', 'interval_to']].reset_index('DHID', drop=True)
df_intervals.drop_duplicates(inplace=True)
da2: xr.DataArray = xr.DataArray(data=df_intervals.index,
                                 dims=['interval'],
                                 coords={
                                     # 'index': (['interval'], df_intervals['index']),
                                     'intervals': (['interval'], df_intervals.index),
                                     'interval_from': (['interval'], df_intervals['interval_from']),
                                     'interval_to': (['interval'], df_intervals['interval_to'])})

# The interval_from and to have the extra dimension - need to be a coord with only the interval dim
ds2d = xr.merge([ds2d.drop_vars(cols_pair), da2])  #, compat='override')
print(ds2d)

# swap to be indexed by interval
# ds_interval = ds_merge.swap_dims({'index': 'interval'})
# print(ds_interval)

# add dims from coords
# first we drop to avoid a name conflict
var_name: str = 'DHID'
var = ds_merge[var_name]
ds_merge = ds_merge.drop_vars(var_name).expand_dims({var_name: var.values})
print(ds_merge)

# not the right shape - maybe swap_dims?

# so now we'll NOT include the integer index in the pandas index and we'll try again

cols_pair: List = ['interval_from', 'interval_to']

df_data2['interval'] = df_data2[cols_pair].mean(axis='columns')
df_data2 = df_data2.reset_index().set_index(['DHID', 'interval'])

ds = df_data2.to_xarray()

# shape is right - move to some coords
ds: xr.Dataset = ds.set_coords(['index'] + cols_pair)
# drop the DHID index from the interval_from and to
# ds['interval_from'].drop_dims('DHID')
# Seems like making the da first (like at the top) and then concatenating may be the go.

da: xr.DataArray = xr.DataArray(name='interval',
                                data=df_data2[cols_pair].mean(axis='columns'),
                                dims=['index'],
                                coords={'index': df_data.index,
                                        'interval_from': (['index'], df_data['interval_from']),
                                        'interval_to': (['index'], df_data['interval_to'])})

print(da)

# -----------
data = xr.DataArray([1, 2, 3], dims='x', coords={'x': [10, 20, 30]})
data_newcoord = data.assign_coords(y='coord_value')
data_expanded = data_newcoord.expand_dims('y')
print(data_expanded)
# -------------------

# What about now setting the interval as a dimension - unsure if this is what we are after?
# we'll try again...

df_data['interval'] = df_data[cols_pair].mean(axis='columns')
df_data_2d = df_data.reset_index().set_index(['index', 'interval'])
print(df_data_2d.head())

da_2d: xr.DataArray = xr.DataArray(name='interval',
                                   data=df_data_2d[cols_pair],
                                   dims=['index', 'interval'],
                                   coords={'index': df_data_2d.index.get_level_values('index'),
                                           'interval': (['interval'], ['interval_from', 'interval_to']),
                                           'interval_from': (['index'], df_data['interval_from']),
                                           'interval_to': (['index'], df_data['interval_to'])})

print(da_2d)

# not sure if this is what we want or not???


# now to see what happens when we go to 2D with the same approach.
# we will add DHID

df_data_2d: pd.DataFrame = df_data.reset_index().set_index(['index', 'DHID'])
print(df_data_2d.head())

da_2d: xr.DataArray = xr.DataArray(name='interval',
                                   data=df_data_2d[cols_pair].mean(axis='columns'),
                                   dims=['index'],
                                   coords={'interval_from': (['index'], df_data_2d['interval_from']),
                                           'interval_to': (['index'], df_data_2d['interval_to'])})

print(da_2d)

ds_2d: xr.Dataset = df_data_2d[[col for col in df_data_2d.columns if col not in cols_pair]].to_xarray()
print(ds_2d)

ds_merge_2d: xr.Dataset = xr.merge([ds_2d, da_2d.to_dataset()])
print(ds_merge_2d)

# hmmm - seems a little inconsistent - expand dims perhaps?
# will try to move the DHID to a dim from the 1d dataset instead

ds_2d: xr.Dataset = ds.set_coords(['DHID'])
ds_2d.expand_dims(dim={'new': ds_2d['DHID']})
# da.expand_dims(dim={"y": np.arange(5)}, axis=0)
print(ds_2d)

# penny dropping moment???
# can we reduce complexity by taking the "extra" columns/vars that we currently manage as attr vars
# (creates confusion with xr.attrs), and move them into coords with the line above?
# it means we can no longer perform calculations on them, but that should be ok.
# Assuming we can still plot using them - TODO: check plot and data access functionality


print('done')

# -----------------------------------------------

# old code for reference
#
# # test a basic structure
# da_mass: xr.DataArray = xr.DataArray(data=df_data[[col for col in df_data if 'mass' in col]],
#                                      dims=['index', 'mass'])
#
# da_chem: xr.DataArray = xr.DataArray(data=df_data[['FE', 'al2o3']], dims=['index', 'analyte'])
# da_moisture: xr.DataArray = (da_mass.sel(mass='wet_mass') - da_mass.sel(mass='dry_mass')) / da_mass.sel(
#     mass='wet_mass') * 100
# da_moisture = da_moisture.rename({'mass': 'moisture'})
# da_moisture['moisture'] = 'H2O'
# # da_moisture.assign_coords(moisture='H20')
# da_moisture = da_moisture.expand_dims('moisture')
#
# ds: xr.Dataset = xr.Dataset(data_vars={'Mass': da_mass, 'Moisture': da_moisture, 'Chem': da_chem})
#
# # transform composition to mass
# dsm: xr.Dataset = ds.copy()
# dsm['Moisture'] = ds['Mass'].sel(mass='wet_mass') - ds['Mass'].sel(mass='dry_mass')
# dsm['Chem'] = ds['Chem'] * ds['Mass'].sel(mass='dry_mass') / 100
#
# # transform mass to composition
# dsc: xr.Dataset = dsm.copy()
# dsc['Moisture'] = dsm['Moisture'] / dsm['Mass'].sel(mass='wet_mass') * 100
# dsc['Chem'] = dsm['Chem'] / ds['Mass'].sel(mass='dry_mass') * 100
#
# print(dsm)
#
# print(ds - dsc)
#
# print(np.isclose(ds['Chem'], dsc['Chem']))
#
# # now use transformers
#
# dsm2 = composition_to_mass(ds)
# dsc2 = mass_to_composition(dsm2)
#
# print(dsm2)
#
# print(ds - dsc2)
#
# print(np.isclose(ds['Chem'], dsc2['Chem']))
#
# print('done')
