import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from elphick.mass_composition import MassComposition



class NodeType(Enum):
    SOURCE = 'input'
    SINK = 'output'
    BALANCE = 'degree 2+'


class MCNode:
    def __init__(self,
                 node_id: int = 0,
                 node_name: str = 'Node',
                 node_subset: int = 0,
                 ):

        self.node_id: int = node_id
        self.node_name: str = node_name
        self.node_subset: int = node_subset
        self._tolerance: float = np.finfo('float32').eps

        self._inputs: Optional[List[MassComposition]] = None
        self._outputs: Optional[List[MassComposition]] = None
        self._balanced: Optional[bool] = None
        self._balance_errors: Optional[pd.DataFrame] = None

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
            self._balance_errors = balance_elements.loc[
                ~balance_mask.all(axis='columns'), ~balance_mask.all(axis='index')]
            self._balanced = self._balance_errors.empty
        return self._balanced

    @balanced.setter
    def balanced(self, value):
        self._balanced = value

    def imbalance_report(self) -> Path:
        """A html report of records that do not balance around the node

        Returns:

        """
        rpt_path: Path = Path(f'balance_errors_node_{self.node_id}.html')
        self._balance_errors.to_html(open(rpt_path, 'w'))
        return rpt_path

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
            cols = obj_list[0].data.mc.mc_vars_mass + ['H2O'] + obj_list[0].data.mc.mc_vars_chem
            res: pd.DataFrame = obj_list[0].data.mc.composition_to_mass().to_dataframe()[cols]
            for obj_mc in obj_list[1:]:
                res = res + obj_mc.data.mc.composition_to_mass().to_dataframe()[cols]
        return res

    def add(self, direction: str) -> Optional[pd.DataFrame]:
        """The weighted addition of either node inputs or outputs

        Args:
            direction: 'in' | 'out'

        Returns:

        """
        res: Optional[pd.DataFrame] = None
        obj_list = []
        if direction == 'in':
            obj_list = self.inputs
        elif direction == 'out':
            obj_list = self.outputs

        if obj_list:
            obj_agg: MassComposition = obj_list[0]
            for o in obj_list[1:]:
                obj_agg = obj_agg.add(o)
            obj_agg.name = f"{self.node_name}_{direction}"
            res = obj_agg.data.to_dataframe()
        return res

    def node_balance(self) -> Optional[pd.DataFrame]:
        if self.node_type == NodeType.BALANCE:
            res = self.mass_sum('in') - self.mass_sum('out')
        else:
            res = None
        return res
