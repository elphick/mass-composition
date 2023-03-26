import json
from enum import Enum
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from elphick.mass_composition import MassComposition
# noinspection PyUnresolvedReferences
import elphick.mass_composition.mcxarray  # keep this "unused" import - it helps


class NodeType(Enum):
    SOURCE = 'input'
    SINK = 'output'
    BALANCE = 'degree 2+'


class MCNode:
    def __init__(self,
                 node_id: int = 0,
                 node_name: str = 'Node',
                 ):

        self.node_id: int = node_id
        self.node_name: str = node_name
        self._tolerance: float = np.finfo('float32').eps

        self._inputs: Optional[List[MassComposition]] = None
        self._outputs: Optional[List[MassComposition]] = None
        self._balanced: Optional[bool] = None
        self._balance_errors: Optional[Tuple] = None

    def __str__(self) -> str:
        props: Dict = {k: str(v) for k, v in self.__dict__.items() if k[0] != '_'}
        props['node_type'] = self.node_type.name
        if self.balanced is not None:
            props['balanced'] = str(self.balanced)

        res: str = json.dumps(props)
        return res

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value: List[MassComposition]):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value: List[MassComposition]):
        self._outputs = value

    @property
    def balanced(self):
        balance_elements = self.node_balance()
        if self.node_type == NodeType.BALANCE:
            balance_mask = np.abs(balance_elements) < self._tolerance
            self._balance_errors = balance_elements.loc[~balance_mask.all(axis='columns'), ~balance_mask.all(axis='index')]
            self._balanced = self._balance_errors.empty
        return self._balanced

    @balanced.setter
    def balanced(self, value):
        self._balanced = value

    @property
    def node_type(self) -> Optional[NodeType]:
        if self.inputs and not self.outputs:
            res = NodeType.SINK
        elif self.outputs and not self.inputs:
            res = NodeType.SOURCE
        elif self.inputs and self.outputs:
            res = NodeType.BALANCE
        else:
            res = None
        return res

    @property
    def output_node(self) -> bool:
        res: bool = False
        if self.outputs:
            res = True
        return res

    def mass_sum(self, direction: str) -> Optional[pd.DataFrame]:
        res: Optional[pd.DataFrame] = None
        obj_list = []
        if direction == 'in':
            obj_list = self.inputs
        elif direction == 'out':
            obj_list = self.outputs

        if obj_list:
            obj_sum: MassComposition = obj_list[0]
            for obj_mc in obj_list[1:]:
                obj_sum = obj_sum + obj_mc
            cols = obj_sum.data.mc.mc_vars_mass + ['H2O'] + obj_sum.data.mc.mc_vars_chem
            res = obj_sum.data.mc.composition_to_mass().to_dataframe()[cols]
        return res

    def node_balance(self) -> Optional[pd.DataFrame]:
        if self.node_type == NodeType.BALANCE:
            res = self.mass_sum('in') - self.mass_sum('out')
        else:
            res = None
        return res
