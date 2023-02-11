"""
This code is to test intervals

Intervals are used for metallurgical fractional data.  i.e. sieve fractions
We need:
1) to preserve custom suffixes for the left and right edges (for better context)
2) to provide a gmean mid function for the 'size' intervals (a special case)
3) manage the zero left interval for size intervals (which needs a replacement to allow gmean calc)
4) tranform in and out of xarray from pandas

"""
from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Hashable, Union, Tuple, Type

import numpy as np
import pandas as pd
from pandas import IntervalIndex, Index, Interval
from pandas._typing import IntervalClosedType, Dtype
from pandas.core.arrays import IntervalArray
from pandas.core.indexes.extension import ExtensionIndex
from scipy.stats import gmean

from elphick.mc.mass_composition import sample_data, MassComposition


class NamedInterval(Interval):

    def __init__(self, left, right, closed, name: str = 'unnamed', left_name: str = 'left', right_name: str = 'right'):
        super().__init__(left, right, closed)

        self.name = name
        self.left_name = left_name
        self.right_name = right_name

    @property
    def mid(self):
        res = super().mid
        if self.name.lower() == 'size':
            res = gmean([self.left, self.right])
        return res


I = NamedInterval(3, 5, 'left', name='???')
print(I, I.mid)

I = NamedInterval(3, 5, 'left', name='size')
print(I, I.mid)

print(isinstance(I, Interval), isinstance(I, NamedInterval))


class NamedIntervalArray(IntervalArray):

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.
        """
        # try:
        #     return 0.5 * (self.left + self.right)
        # except TypeError:
        #     # datetime safe version
        #     return self.left + 0.5 * self.length
        res = super().mid
        if str(self.name).lower() == 'size':
            res = gmean([self.left, self.right])
        return res


class NamedIntervalIndex(IntervalIndex):
    _typ = "namedintervalindex"

    # annotate properties pinned via inherit_names
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: NamedIntervalArray
    _values: NamedIntervalArray
    _can_hold_strings = False
    _data_cls = NamedIntervalArray

    @classmethod
    def from_series(
            cls,
            left: pd.Series,
            right: pd.Series,
            closed: IntervalClosedType = "left",
            name: Hashable = None,
            copy: bool = False,
            dtype: Dtype | None = None) -> Type[NamedIntervalIndex]:

        intervals: List = []
        for i, row in pd.concat([left, right], axis='columns').iterrows():
            intervals.append(NamedInterval(row[left.name], row[right.name], closed=closed,
                                           left_name=str(left.name), right_name=str(right.name)))

        if name is None:
            name = intervals[0].name

        obj = cls.__new__(cls=cls, data=intervals, name=name, closed=intervals[0].closed)
        obj = obj.view(cls)
        # obj.__array_finalize__()

        return obj

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.
        """
        # try:
        #     return 0.5 * (self.left + self.right)
        # except TypeError:
        #     # datetime safe version
        #     return self.left + 0.5 * self.length
        res = super().mid
        if str(self.name).lower() == 'size':
            res = gmean([self.left, self.right])
        return res


#
# class NamedIntervalIndex(IntervalIndex, ExtensionIndex):
#     left_name: str
#     right_name: str
#
#     @classmethod
#     def from_series(
#             cls,
#             left: pd.Series,
#             right: pd.Series,
#             closed: IntervalClosedType = "left",
#             name: Hashable = None,
#             copy: bool = False,
#             dtype: Dtype | None = None) -> IntervalIndex:
#         cls = super().from_arrays(left=left,
#                                   right=right,
#                                   closed=closed,
#                                   name=name,
#                                   copy=copy,
#                                   dtype=dtype)
#         cls.left_name = left.name
#         cls.right_name = right.name
#
#         cls = cls.view(cls)
#
#         return cls
#
#     def mid(self) -> Index:
#         if str(self.name).lower() == 'size':
#             # TODO: manage the zero case
#             res = Index(gmean(self.left, self.right), copy=False)
#         else:
#             res = Index(self._data.mid, copy=False)
#         return res


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S%z',
                        )

    _tic = time.time()
    logging.info(f'{Path(__file__).stem} commenced')

    # %%

    # We get some demo data in the form of a pandas DataFrame

    df_data: pd.DataFrame = sample_data()
    df_data['size_retained'] = [0, 3, 6]
    df_data['size_passing'] = [3, 6, 10]
    print(df_data.head())

    # df_data['size'] = IntervalIndex.from_arrays(left=df_data['size_retained'],
    #                                             right=df_data['size_passing'])

    df_data['size'] = NamedIntervalIndex.from_series(left=df_data['size_retained'],
                                                     right=df_data['size_passing'])

    obj: NamedIntervalIndex = df_data['size']
    print(obj.array.mid)

    print(obj.name, obj.array.name, obj.array.left_name, obj.array.mid)

    # demonstrate detecting that column
    interval_cols: List[str] = [col for col in df_data.columns if df_data[col].dtype == 'interval']

    # Idea 1 - preserve the suffixes in the series attrs
    suffixes = [n.split('_')[-1] for n in ['size_retained', 'size_passing']]

    df_data['size'].attrs = {'left_name': 'size_retained', 'right_name': 'size_passing'}

    # Construct a MassComposition object

    obj_mc: MassComposition = MassComposition(df_data)
    print(obj_mc)

    # %%

    logging.info(f'{Path(__file__).name} complete in {timedelta(seconds=time.time() - _tic)}')
