import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.transformers import Feed
from .fixtures import demo_data


def test_feed_fit_with_invalid_input():
    feed = Feed()
    invalid_input = ["not", "a", "mass", "composition"]  # replace with your invalid input

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        feed.fit(invalid_input)


def test_feed_initial_state():
    obj = Feed()
    assert obj.is_fitted is False
    assert obj.is_transformed is False


def test_feed_fit_single(demo_data):
    obj = Feed()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit(mc)
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_feed_fit_iterable(demo_data):
    obj = Feed()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit([mc])
    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_feed_transform(demo_data):
    obj = Feed()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit([mc])
    obj.transform()
    assert obj.is_transformed is True
    assert obj.inputs == obj.outputs


def test_feed_fit_transform(demo_data):
    obj = Feed()
    mc: MassComposition = MassComposition(demo_data, name='feed')
    obj.fit_transform(mc)
    assert obj.is_fitted is True
    assert obj.is_transformed is True
    assert obj.inputs == obj.outputs
