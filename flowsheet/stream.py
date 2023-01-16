from copy import deepcopy, copy
from typing import List, Union, Optional, Dict

import pandas as pd
import xarray as xr

from mass_composition.mass_composition import MassComposition


class Stream(MassComposition):
    """
    A Stream is a subclass of MassComposition.  It has a name and supports Flowsheet operations.
    """

    # _operators: List[str] = ['+', '-', '*', '/']

    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None,
                 in_node: Optional[Union[int, str]] = None,
                 out_node: Optional[Union[int, str]] = None):
        d_attr: Dict = deepcopy(locals())
        kwargs: Dict = {k: v for k, v in d_attr.items() if
                        (k not in ['self', 'name', 'in_node', 'out_node']) and ('__' not in k)}
        super().__init__(**kwargs)

        self.name: str = name
        self.in_node: Optional[Union[int, str]] = in_node
        self.out_node: Optional[Union[int, str]] = out_node

    def __str__(self) -> str:
        res = f'\n{self.name}\n'
        res += f"{len(self.name) * '_'}"
        res += f'\n{super(Stream, self).__str__()}'
        return res

    def split(self, fraction: float) -> tuple['Stream', 'Stream']:
        obj_mc_1, obj_mc_2 = super().split(fraction)
        stream_1: Stream = Stream(data=obj_mc_1.data.to_dataframe(), name=f'({fraction} * {self.name})')
        stream_2: Stream = Stream(data=obj_mc_2.data.to_dataframe(), name=f'({1 - fraction} * {self.name})')
        return stream_1, stream_2

    def __add__(self, other) -> 'Stream':
        name = f'{self.name} + {other.name}'
        res: Stream = Stream(data=super().__add__(other).data.to_dataframe(), name=name)
        return res

    def __sub__(self, other) -> 'Stream':
        name = f'{self.name} - {other.name}'
        res: Stream = Stream(data=super().__sub__(other).data.to_dataframe(), name=name)
        return res

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, name: str) -> 'Stream':
        return cls(data=data, name=name)

    @classmethod
    def from_mass_composition(cls, obj: MassComposition, name: str) -> 'Stream':
        # d_obj: Dict = copy(obj.__dict__)
        # d_obj = {k: v for k, v in d_obj.items() if '_' not in k}
        # non_init_vars: List[str] = ['attr_vars', 'chemistry_var_map', 'input_columns']
        # d_init: Dict = {k: v for k, v in d_obj.items() if k not in non_init_vars}
        # d_non_init: Dict = {k: v for k, v in d_obj.items() if k in non_init_vars}
        # new_obj: cls = cls(name=name,
        #                    **d_init)
        # for k, v in d_non_init.items():
        #     new_obj.__dict__[k] = v

        new_obj: cls = cls(name=name,
                           data=obj.data.to_dataframe(),
                           mass_wet_var=obj.mass_wet_var,
                           mass_dry_var=obj.mass_dry_var,
                           chem_vars=obj.chem_vars)

        return new_obj

    def to_dataframe(self) -> pd.DataFrame:
        return self.data.to_dataframe()
