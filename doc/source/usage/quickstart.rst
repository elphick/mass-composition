Quick Start Guide
=================

It is possible that you already have your mass-composition data in a pandas DataFrame.

If this is the case, provided some pre-requisites are met, we can create an xarray mass-composition Dataset.

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


Note the use of the mc accessor to access the mass-composition method.

For more examples, see the :doc:`/auto_examples/index`.
