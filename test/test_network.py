import pandas as pd

# noinspection PyUnresolvedReferences
from test.data import demo_data
from elphick.mass_composition import MassComposition


def test_node_initialise(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data)
    assert 'nodes' in obj_mc.data.attrs.keys()
    assert obj_mc.data.attrs['nodes'] == [0, 1]


def test_node_split(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data)
    assert obj_mc.data.attrs['nodes'] == [0, 1]

    obj_mc_1, obj_mc_2 = obj_mc.split(0.5)

    assert obj_mc_1.data.attrs['nodes'] == [1, 2]
    assert obj_mc_2.data.attrs['nodes'] == [1, 3]
