import logging
from typing import List

import pandas as pd


class Status:
    def __init__(self, df_oor: pd.DataFrame):
        self.oor: pd.DataFrame = df_oor
        self.num_oor: int = len(df_oor)
        self.failing_components: List[str] = list(df_oor.dropna(axis=1).columns) if self.num_oor > 0 else None

    @property
    def ok(self) -> bool:
        if self.num_oor > 0:
            logging.warning(f'{self.num_oor} OOR records exist.')
        return True if self.num_oor == 0 else False

    def __str__(self) -> str:
        res: str = f'status.ok: {self.ok}\n'
        res += f'num_oor: {self.num_oor}'
        return res
