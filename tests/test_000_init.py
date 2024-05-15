import logging
import pickle

import pandas as pd

from elphick.mass_composition import MassComposition
from .fixtures import demo_data


def test_init(demo_data):
    # test that the MassComposition object can be initialized with the expected pd.DataFrame being passed.
    obj_mc: MassComposition = MassComposition(demo_data, name='test')

    # Assert that the name attribute is correctly set
    assert obj_mc.name == 'test', "The name attribute was not correctly set during initialization"


def test_multiindex_warning(demo_data, caplog):
    # set a 3 level index on our demo dataframe
    index = pd.MultiIndex.from_tuples([(1, 2, 3), (4, 5, 6), (7, 8, 9)], names=['x', 'y', 'z'])
    demo_data.set_index(index, inplace=True)

    # Create a MassComposition object
    mc = MassComposition()

    with caplog.at_level(logging.WARNING):
        # Pass the DataFrame to the set_data method
        mc.set_data(demo_data)

    # Check if a warning was logged
    assert "The data has more than 2 levels in the index" in caplog.text


def test_masscomposition_pickleability(demo_data):
    # Create a MassComposition object
    obj_mc: MassComposition = MassComposition(demo_data, name='test')

    # Pickle the object
    pickled_mass_comp = pickle.dumps(obj_mc)

    # Unpickle the object
    unpickled_mass_comp = pickle.loads(pickled_mass_comp)

    # Check if the original and unpickled objects are the same
    assert obj_mc == unpickled_mass_comp, "The original and unpickled MassComposition objects are not the same."
