Quick Start Guide
=================

Once you have xarray and mass-composition xarray installed in your environment, you will need the following imports.

..  code-block:: python

    import xarray as xr
    import mcxarray.mcxarray


It is possible that you already have your mass-composition data in a pandas DataFrame.

If this is the case, provided some pre-requisites are met, we can create an xarray mass-composition Dataset
from your pandas DataFrame.

DataFrame requirements:

- mass_dry column must exist
- mass_wet column is optional
- chemical elements/components/oxides will be automatically detected.

If the DataFrame meets the above requirements, an xarray mass-composition Dataset can be created by:

..  code-block:: python

    xr_ds: xr.Dataset = xr.Dataset(df_data)

It is then a simple case of standardising the composition symbols if desired.

..  code-block:: python

    xr_ds = xr_ds.mc.convert_chem_to_symbols()

Note the use of the mc accessor to access the mass-composition method called convert_chem_to_symbols.

We can calculate the weight average of the mass-composition by executing the following.

..  code-block:: python

    xr_ds.mc.aggregate()

For examples that illustrate math operations and visualisation, see the :doc:`/auto_examples/index`.
