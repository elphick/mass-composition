from elphick.mass_composition import MassComposition
from .fixtures import demo_data


def test_init(demo_data):
    # test that the MassComposition object can be initialized with the expected pd.DataFrame being passed.
    obj_mc: MassComposition = MassComposition(demo_data, name='test')

    # Assert that the name attribute is correctly set
    assert obj_mc.name == 'test', "The name attribute was not correctly set during initialization"
