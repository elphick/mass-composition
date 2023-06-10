"""
3D Data
=======

The obvious 3dimensional dataset is one in 3D world space - cartesian coordinates: x, y, z
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly

from plotly.graph_objs import Figure
import xarray as xr
import pyvista as pv
import pvxarray

from elphick.mass_composition import MassComposition

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
df_data.head()

# %%

obj_mc: MassComposition = MassComposition(df_data, name=name)
obj_mc.aggregate(group_var='DHID')

# %%
#
# Plotting using PyVista
# ----------------------

df: pd.DataFrame = df_data.reset_index().set_index(['x', 'y', 'z'])

obj_mc: MassComposition = MassComposition(df, name=name, mass_units='kg')

# get the underlying xarray dataset
xr_ds: xr.Dataset = obj_mc.data


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
poly.threshold(poly.get_data_range(), all_scalars=True).plot()


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
