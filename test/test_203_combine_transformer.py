import pandas as pd
import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.transformers import Combine
from .fixtures import demo_data


def test_combine_fit_with_invalid_input():
    feed = Combine()
    invalid_input = ["not", "a", "mass", "composition"]  # replace with your invalid input

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        feed.fit(invalid_input)


def test_combine_initial_state():
    obj = Combine()
    assert obj.is_fitted is False
    assert obj.is_transformed is False

    obj = Combine(name='new_name')
    assert obj.is_fitted is False
    assert obj.is_transformed is False
    assert obj.name == 'new_name'


def test_combine_fit_two(demo_data):
    obj = Combine()
    mc1: MassComposition = MassComposition(demo_data, name='stream_1')
    mc2: MassComposition = MassComposition(demo_data, name='stream_2')

    obj.fit([mc1, mc2])
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_combine_fit_three(demo_data):
    obj = Combine()
    mc1: MassComposition = MassComposition(demo_data, name='stream_1')
    mc2: MassComposition = MassComposition(demo_data, name='stream_2')
    mc3: MassComposition = MassComposition(demo_data, name='stream_3')
    obj.fit([mc1, mc2, mc3])
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_combine_transform(demo_data):
    obj = Combine()
    mc1: MassComposition = MassComposition(demo_data, name='stream_1')
    mc2: MassComposition = MassComposition(demo_data, name='stream_2')
    mc3: MassComposition = MassComposition(demo_data, name='stream_3')
    obj.fit([mc1, mc2, mc3])
    obj.transform()
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.inputs[0].add(obj.inputs[1]).add(obj.inputs[2]).data.to_dataframe(),
                                  obj.outputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)


def test_combine_fit_transform(demo_data):
    obj = Combine()
    mc1: MassComposition = MassComposition(demo_data, name='stream_1')
    mc2: MassComposition = MassComposition(demo_data, name='stream_2')
    mc3: MassComposition = MassComposition(demo_data, name='stream_3')
    obj.fit_transform([mc1, mc2, mc3])
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.inputs[0].add(obj.inputs[1]).add(obj.inputs[2]).data.to_dataframe(),
                                  obj.outputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)
