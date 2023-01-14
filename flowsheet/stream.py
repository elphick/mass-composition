from typing import List

import pandas as pd
import xarray as xr

from mass_composition.mass_composition import MassComposition


class Stream(MassComposition):
    """
    A Stream is a subclass of MassComposition.  It has a name and supports Flowsheet operations.
    """
    # _operators: List[str] = ['+', '-', '*', '/']

    def __init__(self, xarray_obj: xr.Dataset, name: str = 'unnamed'):
        super().__init__(xarray_obj)
        self.name: str = name

    def __str__(self) -> str:
        res = f'\n{self.name}\n'
        res += f"{len(self.name) * '_'}"
        res += f'\n{super(Stream, self).__str__()}'
        return res

    def split(self, fraction: float) -> tuple['Stream', 'Stream']:
        obj_mc_1, obj_mc_2 = super().split(fraction)
        stream_1: Stream = Stream(xarray_obj=obj_mc_1.data, name=f'({fraction} * {self.name})')
        stream_2: Stream = Stream(xarray_obj=obj_mc_2.data, name=f'({1 - fraction} * {self.name})')
        return stream_1, stream_2

    def __add__(self, other) -> 'Stream':
        name = f'{self.name} + {other.name}'
        res: Stream = Stream(super().__add__(other).data, name=name)
        return res

    def __sub__(self, other) -> 'Stream':
        name = f'{self.name} - {other.name}'
        res: Stream = Stream(super().__sub__(other).data, name=name)
        return res

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, name: str = 'unnamed') -> 'Stream':
        return cls(xr.Dataset(data), name=name)

    def to_dataframe(self) -> pd.DataFrame:
        return self._obj.to_dataframe()
