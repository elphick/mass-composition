"""
Cumulative Mass Transform
=========================

In or der resample while maintaining mass and component mass across the original fractions, we transform to
cumulative mass.  This allows interpolation in the cumulative space with methods that preserve monotonicity, hence
mass cannot be lost and will match the original intervals.

This script does not interpolate, but demonstrates the round trip from mass to mass_cum and back to mass.

When an array M x N is transformed to cumulative space it remains M x N.  This can be performed in one step
across all dimensions by ds.cumsum(dim='...')

It is more complicate to transform a mass_cum array back to mass using since diff.
When a cumulative array M x N is transformed back to mass space it becomes M-1 x N-1.
The missing record is equal to the first record in the dataset.  This is trivial for 1D, but more complicated for N-D,
since the first record in a given dimension needs to be diff'd before appending.
Finally, the corner record where M=0 & N=0 must also be appended.


"""

# %%

from typing import List, Dict

import numpy as np
import pandas as pd
import xarray.tests
import xarray as xr
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
import pvxarray
import pyvista as pv

from elphick.mc.mass_composition import MassComposition, MassCompositionAccessor
from elphick.mc.mass_composition.data.sample_data import dh_intervals

plt.switch_backend('TkAgg')

# TODO: resolve the bug when passing a single analyte only
# %%

# get some data

df_data: pd.DataFrame = dh_intervals(n=5, n_dh=2)
df_data = df_data.reset_index().set_index(['DHID', 'interval_to'])

obj_mc: MassComposition = MassComposition(df_data,
                                          name='Drill program',
                                          mass_units='kg')
print(obj_mc)

print(obj_mc.aggregate())
print(obj_mc.aggregate('DHID'))

# fig = obj_mc.plot_parallel(color='Fe')
# fig.show()

# %%
#
# Step 1 - Transform to mass

ds: xr.Dataset = obj_mc.data

ds_mass: xr.Dataset = ds.mc.composition_to_mass()
print(ds_mass)

df_mass = ds_mass.to_dataframe()

# %%
#
# Step 2 - Cumulative sum across both dims

ds_mass_cum: xr.Dataset = ds_mass.cumsum(keep_attrs=False)
ds_mass_cum = ds_mass_cum.assign_coords(**ds_mass.coords)
ds_mass_cum['DHID'] = ds_mass_cum['DHID'].to_pandas().astype('category').cat.codes
print(ds_mass_cum)
df_mass_cum = ds_mass_cum.to_dataframe()

# %%
#
# Step 3 - Difference (invert the cumulation)

# Attempt 1 - Fails
# The OOTB solution does not work - output is identical to input

# ds_mass_diff: xr.Dataset = ds_mass_cum.diff(dim='...')
# print(ds_mass_diff)
# df_mass_diff = ds_mass_diff.to_dataframe()
#
# xarray.tests.assert_allclose(ds_mass_cum, ds_mass_diff)

# Attempt 2 - Iterate through the dims

# try again using chaining - one dim at a time
ds_mass_diff: xr.Dataset = ds_mass_cum.copy()
d_parts: Dict = {}
for d in ds_mass_cum.dims:
    d_parts[d] = {d: ds_mass_diff.diff(dim=d)}
    print('dim', d, d_parts[d][d].to_dataframe(), '\n')
    for _d in ds_mass_cum.dims:
        if _d != d:
            d_parts[d][_d]: xr.Dataset = ds_mass_cum.isel({d: 0}).diff(dim=_d)
            print('_dim', _d, d_parts[d][_d].to_dataframe(), '\n')

    # ds_mass_diff: xr.Dataset = ds_mass_diff.diff(dim=d)

df_mass_diff = ds_mass_diff.to_dataframe()

d_first_elements: Dict = {}
for d in ds_mass_cum.dims:
    # create the first elements for each dim
    tmp_mass_cum_rev_first: xr.Dataset = ds_mass_cum.isel({d: 0})
    for _d in ds_mass_cum.dims:
        if _d != d:
            tmp_mass_cum_rev_first: xr.Dataset = tmp_mass_cum_rev_first.diff(dim=_d)
    d_first_elements[d] = tmp_mass_cum_rev_first

ds_mass_cum_rev_merge: xr.Dataset = ds_mass_cum_rev.copy()
for d, v in d_first_elements.items():
    # merge into the results
    v_newcoord = v.assign_coords({d: ds_mass_cum[d][0]})
    v_expanded = v_newcoord.expand_dims(d)
    ds_mass_cum_rev_merge = xr.merge([v_expanded, ds_mass_cum_rev_merge])

print(ds_mass_cum_rev_merge)

df_mass_cum_rev = ds_mass_cum_rev.to_dataframe()

df_mass_cum_rev_merge = ds_mass_cum_rev_merge.to_dataframe()

# try and use shift
ds_mass_cum_shift: xr.Dataset = ds_mass_cum.shift({'DHID': 1, 'interval_to': 1})
print(ds_mass_cum_shift)

ds_mass_cum_rev = ds_mass_cum - ds_mass_cum_shift
print(ds_mass_cum_rev)
df_mass_cum_rev: pd.DataFrame = ds_mass_cum_rev.to_dataframe()

xarray.tests.assert_allclose(ds_mass.to_array(), ds_mass_cum_rev.to_array())

print('debug')
#
# plt.figure()
# ds_mass_cum.plot.scatter(x='interval_to', y='Fe', hue='DHID')
# plt.show(block=False)
#
# plt.figure()
# ds_mass_cum.plot.scatter(x='interval_to', y='mass_dry', hue='DHID')
# plt.show(block=False)
#
# # a diversion - calculate cumulative grade
# ds_mass_cum_grade: xr.Dataset = ds_mass_cum.mc.mass_to_composition()
#
# plt.figure()
# ds_mass_cum_grade.sel(DHID=0).plot.scatter(x='interval_to', y='Fe', hue='DHID')
# plt.show(block=False)
#
# # absolute grades by fraction
# plt.figure()
# ds.sel(DHID='CBS02').plot.scatter(x='interval_to', y='Fe', hue='DHID')
# plt.show(block=False)

# ds_mass_cum['Fe'].plot.surface()
# plt.show()

ds_mass_cum_rev: xr.Dataset = ds_mass_cum.diff(dim='...')

tst: xr.DataArray = ds_mass_cum['Fe'].diff('...')
df_tst = tst.to_dataframe()

df_mass = ds_mass.to_dataframe()
df_mass_cum = ds_mass_cum.to_dataframe()
df_mass_cum_rev = ds_mass_cum_rev.to_dataframe()

xarray.tests.assert_allclose(ds_mass, ds_mass_cum_rev)

# %%
#
# Step 3 - Create a new regular grid

grid_spacing: float = float(ds_mass_cum['interval_to'].to_dataframe().diff().min().round(2))

new_intervals = np.arange(ds_mass_cum['interval_to'].min(), ds_mass_cum['interval_to'].max() + grid_spacing,
                          grid_spacing)

# %%
#
# Step 4 - Interpolate at the new grid coordinates

ds_mass_cum_upsampled: xr.Dataset = ds_mass_cum.interp(interval_to=new_intervals, method='linear')  # ,
# kwargs=dict(order='pchip'))


# ds_mass_cum_upsampled.set_coords('Fe')['Fe'].pyvista.plot(x='interval_to', y='DHID')
# # mesh = ds_mass_cum_upsampled.expand_dims('Fe')['Fe'].pyvista.mesh(x='interval_to', y='index', z='Fe')
#
# # Plot in 3D
# p = pv.Plotter()
# p.add_mesh(mesh, lighting=False, cmap='plasma', clim=[0, 35])
# p.view_vector([1, -1, 1])
# p.set_scale(zscale=0.001)
# p.show()

# %%
#
# Step 5 - Undo the cumsum to get back to fractional masses

ds_mass_upsampled: xr.Dataset = ds_mass_cum_upsampled.diff(dim='...')

# %%
#
# Step 6 Difference across both dimensions to recreate the fractional data

ds_comp_upsampled: xr.Dataset = ds_mass_upsampled.mc.mass_to_composition()

# %%
#
# Perform a rough check using the aggregate.

original: xr.Dataset = obj_mc.aggregate('DHID')['Fe']
upsampled: xr.Dataset = ds_comp_upsampled.mc.aggregate('DHID', as_dataframe=True)['Fe']

# TODO: check the mass balance across the original fractions

print('done')
