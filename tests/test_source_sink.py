import pandas as pd

# noinspection PyUnresolvedReferences
from .fixtures import demo_data
from elphick.mass_composition import MassComposition, Flowsheet


def test_in_edges(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    fs: Flowsheet = Flowsheet().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    assert len(fs.get_input_streams()) == 1
    assert fs.get_input_streams()[0].name == 'Feed'


def test_out_edges(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    fs: Flowsheet = Flowsheet().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    assert len(fs.get_output_streams()) == 2
    assert [mc.name for mc in fs.get_output_streams()] == ['(0.4 * Feed)', '(0.6 * Feed)']
