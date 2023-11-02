"""
Pandas utils
"""
import logging
from typing import List, Dict, Optional

import pandas as pd

from elphick.mass_composition.utils import solve_mass_moisture


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
    mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    component_cols = [col for col in df.columns if col.lower() not in [mass_wet, mass_dry, 'h2o', 'moisture']]
    component_mass: pd.DataFrame = df[component_cols]
    composition: pd.DataFrame = component_mass / mass[mass_dry] * 100.0
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

    mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    component_cols = [col for col in df.columns if col.lower() not in [mass_wet, mass_dry, 'h2o', 'moisture']]
    composition: pd.DataFrame = df[component_cols]
    component_mass: pd.DataFrame = composition * mass[mass_dry] / 100.0
    moisture_mass: pd.Series = mass_wet - mass_dry
    return pd.concat([mass, moisture_mass, component_mass], axis='columns')


def weight_average(df: pd.DataFrame,
                   mass_wet: str = 'mass_wet',
                   mass_dry: str = 'mass_dry') -> pd.Series:
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
    mass_sum: pd.DataFrame = df.pipe(composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry).sum(axis="index")
    moisture: pd.Series = solve_mass_moisture(mass_wet=mass_sum[mass_wet], mass_dry=mass_sum[mass_dry])
    component_cols = [col for col in df.columns if col.lower() not in [mass_wet, mass_dry, 'h2o', 'moisture']]
    weighted_composition: pd.Series = mass_sum[component_cols] / mass_sum['mass_dry'] * 100

    return pd.concat([mass_sum[mass_wet], mass_sum[mass_dry], moisture, weighted_composition])


def recovery(df: pd.DataFrame,
             df_ref: pd.DataFrame,
             mass_wet: str = 'mass_wet',
             mass_dry: str = 'mass_dry') -> pd.Series:
    """Calculate recovery of mass-composition for two DataFrames

    Args:
        df: The pd.DataFrame containing mass-composition.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        df_ref: Of the form consistent with df.  This variable represents the denominator in the recovery calculation.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Series containing the total mass and weight averaged composition.
    """
    pass
