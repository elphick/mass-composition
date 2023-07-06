from typing import List

from elphick.mass_composition.utils.pd_utils import column_prefixes, column_prefix_counts


def test_col_prefixes():
    cols: List = ['Fe', 'One_DMT', 'One_Fe', 'Two_DMT', 'Two_Fe']
    res = column_prefixes(cols)
    expected = {'One': ['One_DMT', 'One_Fe'], 'Two': ['Two_DMT', 'Two_Fe']}
    assert res == expected


def test_col_counts():
    cols: List = ['Fe', 'One_DMT', 'One_Fe', 'Two_DMT', 'Two_Fe']
    res = column_prefix_counts(cols)
    expected = {'One': 2, 'Two': 2}
    assert res == expected
