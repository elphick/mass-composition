# xr-mass-composition

Manage mass-composition math operations and visualisation

**Why Xarray? Why not Pandas?**

Pandas is great for tabular data with a single index/dimension.  While it can handle multi-indexes, it is not 
fundamentally multi-dimensional friendly.

Xarray is designed for multi-dimensional data, and it is typical for mass-composition in the geo-sciences
to be multi-dimensional.  Consider a 3D block model, where rock is modelled in the ground in the x, y, z
dimensions.  When structured as an Xarray dataset, the model can have many variables (each being a xarray.DataArray) 
describing a particular property in that 3D space.