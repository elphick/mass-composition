"""
Pandas utils
"""
import inspect
import logging
from typing import List, Dict, Optional

import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_float_dtype

from elphick.mass_composition.utils import solve_mass_moisture
from elphick.mass_composition.utils.size import mean_size


def column_prefixes(columns: List[str]) -> Dict[str, List[str]]:
    return {prefix: [col for col in columns if prefix == col.split('_')[0]] for prefix in
            list(dict.fromkeys([col.split('_')[0] for col in columns if len(col.split('_')) > 1]))}


def column_prefix_counts(columns: List[str]) -> Dict[str, int]:
    return {k: len(v) for k, v in column_prefixes(columns).items()}


def mass_to_composition(df: pd.DataFrame,
                        mass_wet: str = 'mass_wet',
                        mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Convert a mass DataFrame to composition

    Args:
        df: The pd.DataFrame containing mass.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Dataframe containing mass (wet and dry mass) and composition
    """
    non_float_cols = _detect_non_float_columns(df)
    non_component_cols: List[str] = [mass_wet.lower(), mass_dry.lower(), 'h2o', 'moisture'] + [col.lower() for col in
                                                                                               non_float_cols]

    mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    component_cols = [col for col in df.columns if col.lower() not in non_component_cols]
    component_mass: pd.DataFrame = df[component_cols]
    composition: pd.DataFrame = component_mass.div(mass[mass_dry], axis=0) * 100.0
    moisture: pd.Series = solve_mass_moisture(mass_wet=mass[mass_wet], mass_dry=mass[mass_dry])

    return pd.concat([mass, moisture, composition], axis='columns')


def composition_to_mass(df: pd.DataFrame,
                        mass_wet: str = 'mass_wet',
                        mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Convert a composition Dataframe to mass

    Args:
        df: The pd.DataFrame containing mass_wet, mass+_dry and composition columns.  H2O if provided will be dropped.
          All columns other than the mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting
          is valid.  Assumes composition is in %w/w units.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Dataframe containing mass for all components
    """
    non_float_cols = _detect_non_float_columns(df)
    non_component_cols: List[str] = [mass_wet.lower(), mass_dry.lower(), 'h2o', 'moisture'] + [col.lower() for col in
                                                                                               non_float_cols]

    mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    component_cols = [col for col in df.columns if col.lower() not in non_component_cols]
    composition: pd.DataFrame = df[component_cols]
    component_mass: pd.DataFrame = composition.mul(mass[mass_dry], axis=0) / 100.0
    moisture_mass: pd.Series = pd.Series(mass[mass_wet] - mass[mass_dry], name='H2O', index=mass.index)
    return pd.concat([mass, moisture_mass, component_mass], axis='columns')


def weight_average(df: pd.DataFrame,
                   mass_wet: str = 'mass_wet',
                   mass_dry: str = 'mass_dry') -> DataFrame:
    """Weight Average a DataFrame containing mass-composition

    Args:
        df: The pd.DataFrame containing mass-composition.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Series containing the total mass and weight averaged composition.
    """
    non_float_cols = _detect_non_float_columns(df)

    mass_sum: pd.DataFrame = df.pipe(composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry).sum(
        axis="index").to_frame().T
    moisture: pd.Series = solve_mass_moisture(mass_wet=mass_sum[mass_wet],
                                              mass_dry=mass_sum[mass_dry])
    component_cols = [col for col in df.columns if
                      col.lower() not in [mass_wet, mass_dry, 'h2o', 'moisture'] + non_float_cols]
    weighted_composition: pd.Series = mass_sum[component_cols].div(mass_sum[mass_dry], axis=0) * 100

    return pd.concat([mass_sum[[mass_wet, mass_dry]], moisture, weighted_composition], axis=1)


def calculate_recovery(df: pd.DataFrame,
                       df_ref: pd.DataFrame,
                       mass_wet: str = 'mass_wet',
                       mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Calculate recovery of mass-composition for two DataFrames

    Args:
        df: The pd.DataFrame containing mass-composition.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Series containing the total mass and weight averaged composition.
    """

    res: pd.DataFrame = df.pipe(composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry) / df_ref.pipe(
        composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry)
    return res


def calculate_partition(df_feed: pd.DataFrame,
                        df_ref: pd.DataFrame,
                        col_mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Calculate the partition curve from two streams

    Applicable to the one dimensional case only.  The PN is bounded [0, 1].
    The interval mean for size is the geometric mean, otherwise the arithmetic mean.
    The interval mean is named `da`, which can be interpreted as `diameter-average` or `density-average`.
    TODO: consider a generalised name, fraction-average -> fa?

    Args:
        df_feed: The pd.DataFrame containing mass-composition representing the fractionated feed.
        df_ref: The pd.DataFrame containing mass-composition representing the fractionated reference stream.
        col_mass_dry: The dry mass column, not optional.

    Returns:
        A pd.DataFrame containing the partition data.
    """

    res: pd.DataFrame = df_ref[[col_mass_dry]].div(df_feed[[col_mass_dry]]).rename(columns={col_mass_dry: 'PN'})
    if df_ref.index.name.lower() == 'size':
        res.insert(loc=0, column='da', value=mean_size(res.index))
    else:
        res.insert(loc=0, column='da', value=res.index.mid)
    return res


def _detect_non_float_columns(df):
    _logger: logging.Logger = logging.getLogger(inspect.stack()[1].function)
    non_float_cols: List = [col for col in df.columns if col not in df.select_dtypes(include=[float]).columns]
    if len(non_float_cols) > 0:
        _logger.info(f"The following columns are not float columns and will be ignored: {non_float_cols}")
    return non_float_cols
