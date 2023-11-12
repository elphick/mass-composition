import logging
from typing import Dict, Optional, List, Union, Iterable

import numpy as np
import pandas as pd

from elphick.mass_composition import MassComposition
from elphick.mass_composition.utils.interp import _upsample_grid_by_factor
from elphick.mass_composition.utils.pd_utils import column_prefix_counts, column_prefixes


def streams_from_dataframe(df: pd.DataFrame,
                           mc_name_col: Optional[str] = None,
                           interval_edges: Optional[Union[Iterable, int]] = None) -> Dict[str, MassComposition]:
    """Objects from a DataFrame

    Args:
        df: The DataFrame
        mc_name_col: The column specified contains the names of objects to create.
          If None the DataFrame is assumed to be wide and the mc objects will be extracted from column prefixes.
        interval_edges: The values of the new grid (interval edges).  If an int, will up-sample by that factor, for
         example the value of 10 will automatically define edges that create 10 x the resolution (up-sampled).
         Applicable only to 1d interval indexes.

    Returns:

    """
    logger: logging.Logger = logging.getLogger(__name__)

    stream_data: Dict = {}
    index_names: List = []
    if mc_name_col:
        if mc_name_col in df.index.names:
            index_names = df.index.names
            df.reset_index(mc_name_col, inplace=True)
        if mc_name_col not in df.columns:
            raise KeyError(f'{mc_name_col} is not in the columns or indexes.')
        names = df[mc_name_col].unique()
        for obj_name in names:
            stream_data[obj_name] = df.query(f'{mc_name_col} == @obj_name')[
                [col for col in df.columns if col != mc_name_col]]
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
                stream_data[prefix] = df[[col for col in df.columns if col in cols]].rename(
                    columns={col: col.replace(f'{prefix}_', '') for col in df.columns})

    if interval_edges:
        # unify the edges - this will also interp missing grades
        if not isinstance(df.index, pd.IntervalIndex):
            raise NotImplementedError(f"The index `{df.index}` of the dataframe is not a pd.Interval. "
                                      f" Only 1D interval indexes are valid")
        if isinstance(interval_edges, int):
            all_edges = []
            for strm_data in stream_data.values():
                all_edges.extend(list(np.sort(np.unique(list(strm_data.index.left) + list(strm_data.index.right)))))
            all_edges = list(set(all_edges)).sort()
            indx = pd.IntervalIndex.from_arrays(left=all_edges[0:-1], right=all_edges[1:])
            interval_edges = _upsample_grid_by_factor(indx=indx, factor=interval_edges)

        res = {stream: MassComposition(data=data, name=stream).resample_1d(interval_edges=interval_edges) for
               stream, data in stream_data.items()}
    else:
        res = {stream: MassComposition(data=data, name=stream) for stream, data in stream_data.items()}

    return res
