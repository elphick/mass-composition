from functools import partial

import pandas as pd
import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.transformers import Partition
from elphick.mass_composition.utils.partition import napier_munn
from elphick.mass_composition.datasets.sample_data import size_by_assay

def test_partition_fit_with_invalid_input():
    part_func = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    obj = Partition(partition=part_func)
    invalid_input = ["not", "a", "mass", "composition"]  # replace with your invalid input

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        obj.fit(invalid_input)


def test_partition_initial_state():
    part_func = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    obj = Partition(partition=part_func)
    assert obj.is_fitted is False
    assert obj.is_transformed is False

    obj = Partition(partition=part_func, name_1='new_name_1', name_2='new_name_2')
    assert obj.is_fitted is False
    assert obj.is_transformed is False
    assert obj.name_1 == 'new_name_1'
    assert obj.name_2 == 'new_name_2'


def test_partition_fit():
    part_func = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    obj = Partition(partition=part_func)
    mc: MassComposition = MassComposition(size_by_assay(), name='feed')
    obj.fit(mc)

    assert obj.is_fitted is True
    assert obj.is_transformed is False


def test_partition_transform():
    part_func = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    obj = Partition(partition=part_func)
    mc: MassComposition = MassComposition(size_by_assay(), name='feed')
    obj.fit(mc)
    obj.transform()
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.outputs[0].add(obj.outputs[1]).data.to_dataframe(),
                                  obj.inputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)


def test_partition_fit_transform():
    part_func = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    obj = Partition(partition=part_func)
    mc: MassComposition = MassComposition(size_by_assay(), name='feed')
    obj.fit_transform([mc])
    assert obj.is_transformed is True
    pd.testing.assert_frame_equal(obj.outputs[0].add(obj.outputs[1]).data.to_dataframe(),
                                  obj.inputs[0].data.to_dataframe(), check_column_type=False, check_index_type=False,
                                  check_like=True)
