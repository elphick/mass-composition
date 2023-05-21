from typing import Optional, List, Union, Dict

import pandas as pd
import plotly.graph_objects as go

from elphick.mass_composition.utils.size import mean_size
from elphick.mass_composition.utils.viz import plot_parallel


def parallel_plot(data: pd.DataFrame,
                  color: Optional[str] = None,
                  vars_include: Optional[List[str]] = None,
                  vars_exclude: Optional[List[str]] = None,
                  title: Optional[str] = None,
                  include_dims: Optional[Union[bool, List[str]]] = True,
                  plot_interval_edges: bool = False) -> go.Figure:
    """Create an interactive parallel plot

    Useful to explore multidimensional data like mass-composition data

    Args:
        data: The DataFrame to plot
        color: Optional color variable
        vars_include: Optional List of variables to include in the plot
        vars_exclude: Optional List of variables to exclude in the plot
        title: Optional plot title
        include_dims: Optional boolean or list of dimension to include in the plot.  True will show all dims.
        plot_interval_edges: If True, interval edges will be plotted instead of interval mid

    Returns:

    """
    df: pd.DataFrame = data.copy()
    if vars_include is not None:
        missing_vars = set(vars_include).difference(set(df.columns))
        if len(missing_vars) > 0:
            raise KeyError(f'var_subset provided contains variable not found in the data: {missing_vars}')
        df = df[vars_include]
    if vars_exclude:
        df = df[[col for col in df.columns if col not in vars_exclude]]

    if include_dims is True:
        df.reset_index(inplace=True)
    elif isinstance(include_dims, List):
        for d in include_dims:
            df.reset_index(d, inplace=True)

    interval_cols: Dict[str, int] = {col: i for i, col in enumerate(df.columns) if df[col].dtype == 'interval'}

    for col, pos in interval_cols.items():
        if plot_interval_edges:
            df.insert(loc=pos + 1, column=f'{col}_left', value=df[col].array.left)
            df.insert(loc=pos + 2, column=f'{col}_right', value=df[col].array.right)
            df.drop(columns=col, inplace=True)
        else:
            # workaround for https://github.com/Elphick/mass-composition/issues/1
            if col == 'size':
                df[col] = mean_size(pd.arrays.IntervalArray(df[col]))
            else:
                df[col] = df[col].array.mid

    fig = plot_parallel(data=df, color=color, title=title)
    return fig
