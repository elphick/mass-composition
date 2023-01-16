"""
Transformer functions to-from mass<->composition (metal/component mass)

This is to simplify aggregation operations.
"""
from typing import Optional

import xarray as xr


def composition_to_mass(mass_dry: xr.DataArray,
                        composition: xr.Dataset,
                        mass_wet: Optional[xr.DataArray] = None,
                        attributes: Optional[xr.Dataset] = None) -> xr.Dataset:
    """Transform a xr Dataset from composition (wt%) to mass

    Args:
        mass_dry: Dry Mass must be supplied
        composition: Composition xr.Dataset
        mass_wet: If wet mass is provided, H2O is returned
        attributes: Optional dataset of attribute (extra) variables.  Will not be converted, but simply appended.

    Returns:
        xr.Dataset of component mass
    """

    if mass_wet is not None:
        dsm: xr.Dataset = xr.merge([mass_wet, mass_dry])
        dsm['H2O'] = mass_wet - mass_dry
    else:
        dsm: xr.DataArray = mass_dry

    dsm = xr.merge([dsm, composition.copy() * mass_dry / 100])
    if attributes is not None:
        dsm = xr.merge([dsm, attributes])

    return dsm


def mass_to_composition(mass_dry: xr.DataArray,
                        component_mass: xr.Dataset,
                        mass_wet: Optional[xr.DataArray] = None,
                        attributes: Optional[xr.Dataset] = None) -> xr.Dataset:
    """Transform a xr Dataset from mass to composition (wt%)

    Args:
        mass_dry: Dry Mass must be supplied
        component_mass: Mass of components xr.Dataset
        mass_wet: If wet mass is provided, H2O is returned
        attributes: Optional dataset of attribute (extra) variables.  Will not be converted, but simply appended.

    Returns:
        xr.Dataset of composition (wt%)
    """

    if mass_wet is not None:
        dsc: xr.Dataset = xr.merge([mass_wet, mass_dry])
        dsc['H2O'] = (mass_wet - mass_dry) / mass_wet * 100
    else:
        dsc: xr.DataArray = mass_dry

    dsc = xr.merge([dsc, component_mass.copy() / mass_dry * 100])
    if attributes is not None:
        dsc = xr.merge([dsc, attributes])

    return dsc
