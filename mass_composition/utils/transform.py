"""
Transformer functions to-from mass<->composition (metal/component mass)

This is to simplify aggregation operations.
"""

import xarray as xr


def composition_to_mass(ds_composition: xr.Dataset) -> xr.Dataset:
    """Transform a composition xr Dataset from composition (wt%) to mass

    Args:
        ds_composition: Composition xr.Dataset

    Returns:
        xr.Dataset of component mass
    """

    dsm: xr.Dataset = ds_composition.copy()
    dsm['Moisture'] = ds_composition['Mass'].sel(mass='wet_mass') - ds_composition['Mass'].sel(mass='dry_mass')
    dsm['Chem'] = ds_composition['Chem'] * ds_composition['Mass'].sel(mass='dry_mass') / 100
    return dsm


def mass_to_composition(ds_mass: xr.Dataset) -> xr.Dataset:
    """Transform a xr Dataset from mass to composition (wt%)

    Args:
        ds_mass: Mass xr.Dataset

    Returns:
        xr.Dataset of composition (wt%)
    """
    dsc: xr.Dataset = ds_mass.copy()
    dsc['Moisture'] = ds_mass['Moisture'] / ds_mass['Mass'].sel(mass='wet_mass') * 100
    dsc['Chem'] = ds_mass['Chem'] / ds_mass['Mass'].sel(mass='dry_mass') * 100
    return dsc
