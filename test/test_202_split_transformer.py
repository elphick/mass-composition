import pandas as pd
import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.transformers import Split
from .fixtures import demo_data


def test_split_fit_with_invalid_input():
    feed = Split()
    invalid_input = ["not", "a", "mass", "composition"]  # replace with your invalid input

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        feed.fit(invalid_input)


def test_split_initial_state():
    obj = Split()
    assert obj.is_fitted is False
    assert obj.is_transformed is False

    obj = Split(fraction=0.3, name_1='new_name_1', name_2='new_name_2')
    assert obj.is_fitted is False
    assert obj.is_transformed is False
    assert obj.fraction == 0.3
    assert obj.name_1 == 'new_name_1'
    assert obj.name_2 == 'new_name_2'


def test_split_fit_single(demo_data):
    obj = Split()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit(mc)
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_split_fit_iterable(demo_data):
    obj = Split()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit([mc])
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_split_transform(demo_data):
    obj = Split()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit([mc])
    obj.transform()
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.outputs[0].add(obj.outputs[1]).data.to_dataframe(),
                                  obj.inputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)


def test_split_fit_transform(demo_data):
    obj = Split()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit_transform(mc)
    assert obj.is_fitted is True
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.outputs[0].add(obj.outputs[1]).data.to_dataframe(),
                                  obj.inputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)
