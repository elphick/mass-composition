from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from elphick.mass_composition.config import read_yaml
from elphick.mass_composition.variables import Variables
# noinspection PyUnresolvedReferences
from test.data import demo_data
from elphick.mass_composition import MassComposition
import xarray as xr


@pytest.fixture(scope="module")
def script_loc(request):
    """Return the directory of the currently running test script"""

    # uses .join instead of .dirname, so we get a LocalPath object instead of
    # a string. LocalPath.join calls normpath for us when joining the path
    return Path(request.fspath.join('..'))


def test_config(script_loc):
    read_yaml(script_loc / 'config/test_mc_config.yml')


def test_variables(demo_data):
    config_file = Path(__file__).parent / './config/test_mc_config.yml'
    _config = read_yaml(config_file)

    obj: Variables = Variables(supplied=list(demo_data.columns), config=_config['vars'],
                               specified_map={'mass_wet': 'wet_mass'})
    assert obj.mass.get_var_names() == ['mass_wet', 'mass_dry']
    assert obj.mass.get_col_names() == ['wet_mass', 'mass_dry']
    assert obj.mass.col_to_var() == {'wet_mass': 'mass_wet', 'mass_dry': 'mass_dry'}
    assert obj.mass.var_to_col() == {'mass_wet': 'wet_mass', 'mass_dry': 'mass_dry'}
    assert obj.moisture.var_to_col() == {'H2O': None}

    assert obj.chemistry.var_to_col() == {'Fe': 'FE', 'SiO2': 'SIO2', 'Al2O3': 'al2o3', 'LOI': 'LOI'}
