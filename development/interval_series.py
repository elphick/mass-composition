"""
This code is to test intervals

Initially we subclassed the Interval to add left_name, right_name and a custom mid method for the 
case where name == size.

This does not help us once an interval array is stored in a pandas series, since the pandas.series.array 
has it's own mid function.

So now we will try subclassing the series...


"""
import logging
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pandas import IntervalIndex, Index
from pandas.core.arrays import IntervalArray, ExtensionArray

from elphick.mc.mass_composition import sample_data


class IntervalSeries(pd.Series):
    @property
    def _constructor(self):
        return IntervalSeries

    @property
    def _constructor_expanddim(self):
        return SubclassedDataFrame

    @property
    def array(self) -> ExtensionArray:
        return NamedIntervalArray(self._mgr.array_values())


class SubclassedDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return SubclassedDataFrame

    @property
    def _constructor_sliced(self):
        return IntervalSeries


class NamedIntervalArray(IntervalArray):

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.
        """
        res = super().mid
        if str(self.name).lower() == 'size':
            res = (self.left * self.right) ** 0.5
        return res


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

    df_data['size'] = IntervalIndex.from_arrays(left=df_data['size_retained'],
                                                right=df_data['size_passing'])

    obj: pd.Series = IntervalSeries(df_data['size'])
    print(type(obj))
    print(obj.array.mid)

    # %%

    logging.info(f'{Path(__file__).name} complete in {timedelta(seconds=time.time() - _tic)}')
