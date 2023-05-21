from typing import Dict

import numpy as np
import pandas as pd
from numpy import inf

# noinspection PyUnresolvedReferences
from test.data import demo_data
from elphick.mass_composition import MassComposition
import xarray as xr


def test_component_constraint_dict(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='test_constraint')

    expected_1: Dict = {'mass_wet': [0.0, inf],
                        'mass_dry': [0.0, inf],
                        'H2O': [0.0, 100.0],
                        'Fe': [0.0, 100.0],
                        'SiO2': [0.0, 100.0],
                        'Al2O3': [0.0, 100.0],
                        'LOI': [0.0, 100.0]}

    assert obj_mc.constraints == expected_1

    obj_mc_2: MassComposition = MassComposition(demo_data, name='test_constraint_Fe', constraints={'Fe': [0.0, 69.97]})

    expected_2: Dict = {'mass_wet': [0.0, inf],
                        'mass_dry': [0.0, inf],
                        'H2O': [0.0, 100.0],
                        'Fe': [0.0, 69.97],
                        'SiO2': [0.0, 100.0],
                        'Al2O3': [0.0, 100.0],
                        'LOI': [0.0, 100.0]}

    assert obj_mc_2.constraints == expected_2


def test_component_constraints(demo_data):
    # in range
    obj_mc: MassComposition = MassComposition(demo_data, name='test_constraint')
    assert obj_mc.status.ok is True

    # out of range
    obj_mc_2: MassComposition = MassComposition(demo_data, name='test_constraint_oor', constraints={'Fe': [0.0, 60.0]})
    res2 = obj_mc_2.status.oor
    assert obj_mc_2.status.ok is False
    assert len(res2) == 1
    assert list(res2.dropna(axis=1).columns) == ['Fe']
    assert res2.dropna(axis=1).values.ravel() == np.array([61.])
