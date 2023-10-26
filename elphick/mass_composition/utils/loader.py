import logging
from typing import Dict, Optional, List

import pandas as pd

from elphick.mass_composition import MassComposition
from elphick.mass_composition.utils.pd_utils import column_prefix_counts, column_prefixes


def streams_from_dataframe(df: pd.DataFrame,
                           mc_name_col: Optional[str] = None) -> Dict[str, MassComposition]:
    """Objects from a DataFrame

    Args:
        df: The DataFrame
        mc_name_col: The column specified contains the names of objects to create.
          If None the DataFrame is assumed to be wide and the mc objects will be extracted from column prefixes.

    Returns:

    """
    logger: logging.Logger = logging.getLogger(__name__)

    res: List = []
    index_names: List = []
    if mc_name_col:
        if mc_name_col in df.index.names:
            index_names = df.index.names
            df.reset_index(mc_name_col, inplace=True)
        if mc_name_col not in df.columns:
            raise KeyError(f'{mc_name_col} is not in the columns or indexes.')
        names = df[mc_name_col].unique()
        for obj_name in names:
            res.append(MassComposition(
                data=df.query(f'{mc_name_col} == @obj_name')[[col for col in df.columns if col != mc_name_col]],
                name=obj_name))
        if index_names:  # reinstate the index on the original dataframe
            df.reset_index(inplace=True)
            df.set_index(index_names, inplace=True)
    else:
        # wide case - find prefixes where there are at least 3 columns
        prefix_counts = column_prefix_counts(df.columns)
        prefix_cols = column_prefixes(df.columns)
        for prefix, n in prefix_counts.items():
            if n >= 3:
                logger.info(f"Creating object for {prefix}")
                cols = prefix_cols[prefix]
                res.append(MassComposition(
                    data=df[[col for col in df.columns if col in cols]].rename(
                        columns={col: col.replace(f'{prefix}_', '') for col in df.columns}),
                    name=prefix))
    return {mc.name: mc for mc in res}
