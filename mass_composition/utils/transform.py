"""
Transformer functions to-from mass<->composition (metal/component mass)

This is to simplify aggregation operations.
"""
from typing import Optional, List

import xarray as xr


def get_chem_dataset(mc_ds: xr.Dataset) -> xr.Dataset:
    chem_vars: List[str] = [str(n) for n, da in mc_ds.items() if da.attrs['mc_type'] == 'chemistry']
    return mc_ds[chem_vars]


def get_mass_dataset(mc_ds: xr.Dataset) -> xr.Dataset:
    mass_vars: List[str] = [str(n) for n, da in mc_ds.items() if da.attrs['mc_type'] == 'mass']
    return mc_ds[mass_vars]


def get_attrs_dataset(mc_ds: xr.Dataset) -> xr.Dataset:
    attr_vars: List[str] = [str(n) for n, da in mc_ds.items() if da.attrs['mc_type'] == 'attribute']
    return mc_ds[attr_vars]


def composition_to_mass(mc_ds: xr.Dataset) -> xr.Dataset:
    """Transform a mc xr Dataset from composition (wt%) to mass

    Args:
        mc_ds: a mc compliant xr.Dataset

    Returns:
        xr.Dataset of component mass
    """

    xr.set_options(keep_attrs=True)

    dsm: xr.Dataset = xr.merge([get_mass_dataset(mc_ds), get_chem_dataset(mc_ds) * mc_ds['mass_dry'] / 100,
                                get_attrs_dataset(mc_ds)])
    for name, da in dsm.items():
        if da.attrs['mc_type'] == 'chemistry':
            da.attrs['units'] = dsm['mass_wet'].attrs['units']

    xr.set_options(keep_attrs='default')

    return dsm


def mass_to_composition(mc_ds: xr.Dataset) -> xr.Dataset:
    """Transform a xr Dataset from mass to composition (wt%)

    Args:
        mc_ds: a mc compliant xr.Dataset

    Returns:
        xr.Dataset of composition (wt%)
    """

    xr.set_options(keep_attrs=True)

    dsc: xr.Dataset = xr.merge([get_mass_dataset(mc_ds), get_chem_dataset(mc_ds) / mc_ds['mass_dry'] * 100])
    for name, da in dsc.items():
        if da.attrs['mc_type'] == 'chemistry':
            da.attrs['units'] = '%'

    xr.set_options(keep_attrs='default')

    return dsc
