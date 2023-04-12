from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

import xarray as xr


def interp_monotonic(ds: xr.Dataset, coords: Dict, include_original_coords: bool = True) -> xr.Dataset:
    """Interpolate with zero mass loss using pchip

    The pchip interpolation cannot be used via the xr.Dataset.interp method directly due to an error.
    This interpolates data_vars independently for a single dimension (coord) at a time.

    The function will:
    - convert from relative composition (%) to absolute (mass)
    - convert the index from interval to a float representing the right edge of the interval
    - cumsum to provide monotonic increasing data
    - interpolate with a pchip spline to preserve mass
    - diff to recover the original fractional data
    - reconstruct the interval index from the right edges
    - convert from absolute to relative composition

    Args:
        ds: The xarray Dataset with relative composition context
        include_original_coords: If True include the original coordinates in the result
        coords: A dictionary of coordinates mapped to the interpolated values.

    Returns:

    """

    if len(coords) > 1:
        raise NotImplementedError("Not yet tested for more than one dimension")

    ds_res: xr.Dataset = ds
    for coord, x in coords.items():

        ds_mass: xr.Dataset = ds.mc.composition_to_mass().sortby(variables=coord, ascending=True)
        # preserve the minimum interval index for later
        original_index = pd.arrays.IntervalArray(ds_mass[coord].data)
        mass: xr.Dataset = ds_mass.cumsum(keep_attrs=True)

        # put the coords back
        mass = mass.assign_coords(**ds_mass.coords)

        # # we'll work in cumulative mass space, using the right edge of the fraction (passing in the size context)
        mass['size'] = pd.arrays.IntervalArray(mass['size'].data).right

        # check the input is monotonic
        mass_check: pd.Series = mass.to_dataframe().apply(lambda col: col.is_monotonic_increasing, axis='index')
        if not np.all(mass_check):
            raise ValueError("The input data is not monotonic - have you not passed a cumulative mass dataset?")

        chunks: List[np.ndarray] = []
        for v in list(mass.data_vars):
            chunks.append(pchip_interpolate(mass[coord], mass[v], x))

        df = pd.DataFrame(data=chunks, index=list(mass.data_vars), columns=x).T
        df.index.name = coord
        mass_check: pd.Series = df.apply(lambda col: col.is_monotonic_increasing, axis='index')
        if not np.all(mass_check):
            raise ValueError("The interpolation is not monotonic - mass has not been preserved.")

        if include_original_coords:
            ds_res: xr.Dataset = xr.concat([mass, xr.Dataset.from_dataframe(df)], dim=coord, combine_attrs='override')
            ds_res = ds_res.drop_duplicates(dim=coord).sortby(variables=coord, ascending=True)
        else:
            ds_res: xr.Dataset = xr.Dataset.from_dataframe(df)
            ds_res.attrs.update(ds_res.attrs)
            da: xr.DataArray
            for new_da, da in zip(ds_res.values(), ds_res.values()):
                new_da.attrs.update(da.attrs)

        # back to fractions using diff, concat to inject in the correct first record
        ds_res = xr.concat([mass.isel({coord: 0}).expand_dims(coord), ds_res.diff(dim=coord)], dim=coord)

        # create a new interval index
        interval_index: pd.Series = pd.Series(pd.IntervalIndex.from_arrays(
            left=ds_res[coord].shift({coord: 1}).fillna(original_index.min().left).values, right=ds_res[coord].values,
            closed='left'), name=coord)

        ds_res[coord] = interval_index.values

        ds_res = ds_res.sortby(variables=coord, ascending=False)
        ds_res = ds_res.mc.mass_to_composition()

    return ds_res
