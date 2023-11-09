from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

import xarray as xr

from elphick.mass_composition.utils.pd_utils import composition_to_mass, mass_to_composition


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


def mass_preserving_interp(df_intervals: pd.DataFrame, grid_vals,
                           include_original_edges: bool = True, precision: Optional[int] = None,
                           mass_wet: str = 'mass_wet', mass_dry: str = 'mass_dry') -> pd.DataFrame:
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
        df_intervals: A pd.DataFrame with a single interval index, with mass, composition context.
        grid_vals: The values of the new grid (interval edges).
        include_original_edges: If True include the original index edges in the result
        precision: Number of decimal places to round the index (edge) values.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:

    """

    if not isinstance(df_intervals.index, pd.IntervalIndex):
        raise NotImplementedError(f"The index `{df_intervals.index}` of the dataframe is not a pd.Interval. "
                                  f" Only 1D interval indexes are valid")

    composition_in: pd.DataFrame = df_intervals.copy()
    if precision is not None:
        composition_in.index = pd.IntervalIndex.from_arrays(np.round(df_intervals.index.left, precision),
                                                            np.round(df_intervals.index.right, precision),
                                                            closed=df_intervals.index.closed,
                                                            name=df_intervals.index.name)
        grid_vals = np.round(grid_vals, precision)

    if include_original_edges:
        original_edges = np.hstack([df_intervals.index.left, df_intervals.index.right])
        grid_vals = np.sort(np.unique(np.hstack([grid_vals, original_edges])))
    # convert from relative composition (%) to absolute (mass)
    mass_in: pd.DataFrame = composition_to_mass(composition_in, mass_wet=mass_wet, mass_dry=mass_dry)
    # convert the index from interval to a float representing the right edge of the interval
    mass_in.index = mass_in.index.right
    # add a row of zeros
    mass_in = pd.concat([mass_in, pd.Series(0, index=mass_in.columns).to_frame().T], axis=0).sort_index(ascending=True)
    # cumsum to provide monotonic increasing data
    mass_cum: pd.DataFrame = mass_in.cumsum()
    #  interpolate with a pchip spline to preserve mass
    chunks = []
    for col in mass_cum:
        new_vals = pchip_interpolate(mass_cum.index.values, mass_cum[col].values, grid_vals)
        chunks.append(new_vals)
    mass_cum_upsampled: pd.DataFrame = pd.DataFrame(chunks, index=mass_in.columns, columns=grid_vals).T
    # diff to recover the original fractional data
    mass_fractions_upsampled: pd.DataFrame = mass_cum_upsampled.diff().dropna(axis=0)
    # reconstruct the interval index from the right edges
    mass_fractions_upsampled.index = pd.IntervalIndex.from_arrays(left=[0] + list(mass_fractions_upsampled.index)[:-1],
                                                                  right=mass_fractions_upsampled.index,
                                                                  closed=df_intervals.index.closed,
                                                                  name=df_intervals.index.name)
    # convert from absolute to relative composition
    res = mass_to_composition(mass_fractions_upsampled, mass_wet=mass_wet, mass_dry=mass_dry).sort_index(
        ascending=False)
    return res
