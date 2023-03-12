Quick Start Guide
=================

Once you have xarray and mass-composition installed in your environment, you will typically need the following imports.

..  code-block:: python

    import xarray as xr
    from elphick.mc.mass_composition import MassComposition

It is possible that you already have your mass-composition data in a pandas DataFrame.

If this is the case, provided some pre-requisites are met, we can create an xarray mass-composition Dataset
from your pandas DataFrame.

DataFrame requirements:

- mass_dry column must exist
- mass_wet column is optional
- chemical elements/components/oxides will be automatically detected.

If the DataFrame meets the above requirements, a MassComposition object can be created by:

..  code-block:: python

    obj_mc: MassComposition = MassComposition(df_data)

It is then trivial to calculate the weight average aggregate of the dataset.

..  code-block:: python

    obj_mc.aggregate()

If you want to or need to go "under the hood" you can access the underlying xarray dataset.


..  code-block:: python

    xr_ds: xr.Dataset = obj_mc.data

The mc xarray accessor provides access to mass-composition properties and methods while working with the xarray dataset.

..  code-block:: python

    xr_ds_wtd: xr.Dataset = xr_ds.mc.aggregate()

For examples that illustrate math operations and visualisation, see the :doc:`/auto_examples/index`.
