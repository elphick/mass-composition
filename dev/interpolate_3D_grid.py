"""
Interpolate 3D Grid
===================

The obvious 3 dimensional dataset is one in 3D world space - cartesian coordinates: x, y, z

We will take data on an irregular 3D grid and interpolate to a regular 3D grid.

"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
from IPython.core.display_functions import display
from numpy import meshgrid

from plotly.graph_objs import Figure
import xarray as xr
import pyvista as pv
import pvxarray

from elphick.mass_composition import MassComposition, MassCompositionAccessor

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # !IMPORTANT
# %%
#
# Load some real data and wrangle
# -------------------------------
# We get some demo data in the form of a pandas DataFrame - it has been pre-wrangled to include x, y, z.
#
# x == Easting, y = Northing, z = RL

filepath: Path = Path('../test/data/iron_ore_sample_data_xyz_A072391.csv')
name: str = filepath.stem.split('_')[-1]
df_data: pd.DataFrame = pd.read_csv(filepath, index_col='index')
print(df_data.shape)
print(df_data.head())

obj_mc: MassComposition = MassComposition(df_data, name=name)
display(obj_mc.aggregate(group_var='DHID'))

# %%
#
# Regrid
# ------

# Index by x, y, z - TODO: how to reindex to x,y,z from xarray?
df_xyz: pd.DataFrame = obj_mc.data.mc.to_dataframe().set_index(['x', 'y', 'z'])
df_xyz['DH'] = df_xyz['DHID'].apply(lambda x: int(x[-2:]))
obj_xyz: MassComposition = MassComposition(df_xyz, name='xyz')
ds_xyz: xr.Dataset = obj_xyz.data

ds_xyz.plot.scatter(x='x', y='y')
plt.show()

ds_xyz['DH'].mean('z').plot()  # .scatter(x='x', y='y')
plt.show()

ds_xyz['Fe'].pyvista.plot(x='x', y='y', z='z', show_edges=True)

x_range = ds_xyz['x'].max() - ds_xyz['x'].min()
y_range = ds_xyz['y'].max() - ds_xyz['y'].min()
z_range = ds_xyz['z'].max() - ds_xyz['z'].min()

z_res = ds_xyz['z'].diff('z').min()

# go with 50 x 50 x 0.1
dx = 50
dy = 50
dz = 0.1
x_min: float = float(np.floor(ds_xyz['x'].min() / dx) * dx)
x_max: float = float(np.ceil(ds_xyz['x'].max() / dx) * dx)
y_min: float = float(np.floor(ds_xyz['y'].min() / dy) * dy)
y_max: float = float(np.ceil(ds_xyz['y'].max() / dy) * dy)
z_min: float = float(np.floor(ds_xyz['z'].min()))
z_max: float = float(np.ceil(ds_xyz['z'].max()))

dims = (int((x_max - x_min) / dx), int((y_max - y_min) / dy), int((z_max - z_min) / dz))
# values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
# values.shape

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(dims) + 1

# Edit the spatial reference
grid.origin = (x_min, y_min, z_min)  # The bottom left corner of the data set
grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis

# interpolate onto that grid
ds_xyz_interp = ds_xyz.interp(x=grid.points[:, 0], y=grid.points[:, 1], z=grid.points[:, 2])

# Add the data values to the cell data
# grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!

# Now plot the grid!
grid.plot(show_edges=True)

# xr_irreg = xr_irreg.swap_dims({'index': 'x', 'index': 'y', 'index': 'z'})

# # interpolate spatially.  OK so it is not exactly krigging but hey...
# # xr_ds_interp: xr.Dataset = xr_ds.interp(coords={'x': xr_ds.x, 'y': xr_ds.y, 'z': xr_ds.z})
#
# # this plot is not overly useful - nans obscuring...
# # xr_ds_interp['Fe'].pyvista.plot(x='x', y='y', z='z')
#
# will plot just the non-nan values
mesh: pv.RectilinearGrid = xr_ds['Fe'].pyvista.mesh(x='x', y='y', z='z')
mesh: pv.UnstructuredGrid = mesh.cast_to_unstructured_grid()

threshed = mesh.threshold()
threshed.plot(show_edges=True)

mesh = mesh.point_data_to_cell_data(pass_point_data=True)

mesh.threshold(mesh.get_data_range(), all_scalars=True).plot()

poly: pv.PolyData = pv.PolyData(df.index.to_frame().to_numpy())
poly.cell_data['Fe'] = df['Fe']
poly = poly.threshold(poly.get_data_range(), all_scalars=True)

# ghosts = np.argwhere(mesh["Fe"] < 0.0)
# # This will act on the mesh inplace to mark those cell indices as ghosts
# mesh.remove_cells(ghosts)

# pts: pv.PointSet = mesh.cast_to_pointset()
mesh2: pv.RectilinearGrid = mesh.point_data_to_cell_data(pass_point_data=True)
mesh3: pv.RectilinearGrid = mesh.cell_data_to_point_data()

# Plot in 3D
p = pv.Plotter()
p.add_mesh_threshold(poly)  # , clim=[0, 35])

# p.add_mesh_threshold(mesh2)  # , clim=[0, 35])
# p.add_mesh_threshold(mesh)  # , lighting=False, cmap='plasma')  # , clim=[0, 35])
# p.view_vector([1, -1, 1])
# p.set_scale(zscale=0.001)
p.show()

# # TODO: resolve key error, nan
# fig: Figure = obj_mc.plot_parallel(color='Fe')
# fig.show()
#
# # %%
#
# fig: Figure = obj_mc.plot_parallel(color='Fe', plot_interval_edges=True)
# fig
#
# # %%
#
# # with selected variables
# fig: Figure = obj_mc.plot_parallel(color='Fe', var_subset=['mass_wet', 'H2O', 'Fe', 'SiO2'])
# # noinspection PyTypeChecker
# plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

print('done')
