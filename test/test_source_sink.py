import pandas as pd

from elphick.mass_composition.network import MCNetwork
# noinspection PyUnresolvedReferences
from test.fixtures import demo_data
from elphick.mass_composition import MassComposition


def test_in_edges(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    assert len(mcn.get_input_edges()) == 1
    assert mcn.get_input_edges()[0].name == 'Feed'


def test_out_edges(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='Feed')
    obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

    mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])
    assert len(mcn.get_output_edges()) == 2
    assert [mc.name for mc in mcn.get_output_edges()] == ['(0.4 * Feed)', '(0.6 * Feed)']
