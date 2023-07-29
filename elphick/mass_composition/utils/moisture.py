from copy import deepcopy
from typing import Optional, Dict, List, Union

import pandas as pd
import xarray as xr


def solve_mass_moisture(mass_wet: Optional[Union[pd.Series, xr.DataArray]] = None,
                        mass_dry: Optional[Union[pd.Series, xr.DataArray]] = None,
                        moisture: Optional[Union[pd.Series, xr.DataArray]] = None) -> Union[pd.Series, xr.DataArray]:
    """Solves the missing component of the mass-moisture trifecta

    Args:
        mass_wet: The wet mass - optional if the other two arguments are supplied
        mass_dry: The dry mass - optional if the other two arguments are supplied
        moisture: The moisture [0-100] - optional if the other two arguments are supplied

    Returns:
        A series for the argument that was not supplied
    """

    _vars: Dict = deepcopy(locals())
    vars_supplied: List[str] = [k for k, v in _vars.items() if v is not None]

    if len(vars_supplied) == 3:
        pass
        # raise NotImplementedError('Over-specified - validation code to check the balance is coming soon...')
    elif len(vars_supplied) == 1:
        raise ValueError('Insufficient arguments supplied - at least 2 required.')

    var_to_solve: List[str] = [k for k, v in _vars.items() if v is None]

    res: Optional[pd.Series] = None
    if var_to_solve:
        var_to_solve: str = var_to_solve[0]

        if var_to_solve == 'mass_wet':
            res: pd.Series = mass_dry / (1 - moisture / 100)
            res.name = var_to_solve
        elif var_to_solve == 'mass_dry':
            res: pd.Series = mass_wet - (mass_wet * moisture / 100)
            res.name = var_to_solve
        elif var_to_solve == 'moisture':
            res = (mass_wet - mass_dry) / mass_wet * 100
            res.name = 'H2O'

    return res


if __name__ == '__main__':
    from elphick.mass_composition.datasets.sample_data import sample_data
    import numpy as np

    data = sample_data()
    wet: pd.Series = data['wet_mass']
    dry: pd.Series = data['dry_mass']

    res_1: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=dry, moisture=None)

    h20: pd.Series = res_1.copy()

    dry_calc: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=None, moisture=h20)
    wet_calc: pd.Series = solve_mass_moisture(mass_wet=None, mass_dry=dry, moisture=h20)

    assert all(np.isclose(wet, wet_calc))
    assert all(np.isclose(dry, dry_calc))

    # These should fail
    res_4: pd.Series = solve_mass_moisture(mass_wet=None, mass_dry=None, moisture=h20)
    res_5: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=dry, moisture=h20)
