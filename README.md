# mass-composition-xarray

Manage mass-composition math operations and visualisation in xarray

**Why Xarray? Why not Pandas?**

Pandas is great for tabular data with a single index/dimension.  While it can handle multi-indexes, it is not 
fundamentally multi-dimensional friendly.

Xarray is designed for multi-dimensional data, and it is typical for mass-composition in the geo-sciences
to be multi-dimensional.  Consider a 3D block model, where rock is modelled in the ground in the x, y, z
dimensions.  When structured as an Xarray dataset, the model can have many variables (each being a xarray.DataArray) 
describing a particular property in that 3D space.

If your data is 1D, with sequential or timestamp indexes, don't stress, mass-composition-xarray will still work for you.

If you haven't used xarray before, you should check it out.  Once you get the hang of it you will find it sweet for
regular 3D block models, particularly when leveraging the pyvista-xarray extension.