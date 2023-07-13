import pytest

from elphick.mass_composition.mc_network import MCNetwork
# noinspection PyUnresolvedReferences
from test.fixtures import demo_data
from elphick.mass_composition import MassComposition


def test_indexes(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='one')
    obj_mc_2: MassComposition = MassComposition(demo_data.drop(index=[0]), name='two').set_parent(obj_mc)

    with pytest.raises(KeyError):
        mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_2])

