# mass-composition

Manage mass-composition math operations and visualisation.
Working in Metallurgy or Geoscience and tired of writing the same weight-averaging code in separate projects?
Mass-Composition is for you - Not only does it do the heavy lifting (alright, not-so-heavy lifting), but also offers 
automatic analyte detection and various interactive visualisations out of the box.

**Why Xarray? Why not Pandas?**

Mass-Composition is backed by Xarray.

Pandas is great for tabular data with a single index/dimension. While it can handle multi-indexes, it is not
fundamentally multi-dimensional friendly.

Xarray is designed for labelled multi-dimensional data, and it is typical for mass-composition in the geo-sciences
to be multi-dimensional. Consider a 3D block model, where rock is modelled in the ground in the x, y, z
dimensions. When structured as an Xarray dataset, the model can have many variables (each being a xarray.DataArray)
describing a particular property in that 3D space.

If your data is 1D, with sequential or timestamp indexes, don't stress, mass-composition-xarray will still work for you.

If you haven't used xarray before, you should check it out. Once you get the hang of it you will find it sweet for
regular 3D block models, particularly when leveraging the pyvista-xarray extension.

## Design Notes

1) The data provided must be a pd.DataFrame.
2) The concrete (underlying) data is xarray.
3) The dataframe index/es become xarray dimensions. Dataframe columns become xarray variables.
4) Moisture is not a concrete property - it is calculated when needed from mass_wet and mass_dry.
5) When instantiated, two of the 3 variables, mass_wet, mass_dry, moisture are used to resolve the third.
6) If all 3 mass related variables are provided the mass balance will be checked.
7) Any variable not in the mass vars, moisture var or chem vars is an extra / attribute var.
8) The underlying xarray shall be capable of standalone math operations. Accessible via a xarray accessor called 'mc'.
9) To deliver standalone math operations of the xarray, variables names are standardised: mass_wet, mass_dry, H2O,
   chemical analytes are standardised.
10) xarray attributes are used to specify column type (chemistry/composition versus extra/attribute columns).
11) xarray attributes are used to store the original column names, allowing recreation of a dataframe consistent with
    the input.

## Roadmap

- 0.1.0 - Preliminary development
- 0.2.0 - Math operations, aggregation, visualisation of a single mass_composition object
- 0.3.0 - Resampling along dimensions - e.g. depth down hole, size fractions, etc
- 0.X.0 - Feel free to submit a PR with your suggestions here...
- 1.0.0 - First stable release