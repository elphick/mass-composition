"""
To provide sample data
"""
import os
import random
from pathlib import Path
from typing import Optional, Iterable, List

import pandas as pd
import pytest

from elphick.mass_composition.utils.components import is_compositional

#
# def script_loc(request):
#     '''Return the directory of the currently running test script'''
#     res: Path = Path(__file__):
#     if str(res) ==  '.':
#
#     # a string. LocalPath.join calls normpath for us when joining the path
#     return Path(request.fspath.join('..'))


@pytest.fixture
def demo_data():
    data: pd.DataFrame = sample_data()
    return data


def sample_data(include_wet_mass: bool = True, include_dry_mass: bool = True,
                include_moisture: bool = False) -> pd.DataFrame:
    """Creates synthetic data for testing

    Args:
        include_wet_mass: If True, wet mass is included.
        include_dry_mass: If True, dry mass is included.
        include_moisture: If True, moisture (H2O) is included.

    Returns:

    """

    # mass_wet: pd.Series = pd.Series([100, 90, 110], name='wet_mass')
    # mass_dry: pd.Series = pd.Series([90, 80, 100], name='dry_mass')
    mass_wet: pd.Series = pd.Series([100., 90., 110.], name='wet_mass')
    mass_dry: pd.Series = pd.Series([90., 80., 90.], name='mass_dry')
    chem: pd.DataFrame = pd.DataFrame.from_dict({'FE': [57., 59., 61.],
                                                 'SIO2': [5.2, 3.1, 2.2],
                                                 'al2o3': [3.0, 1.7, 0.9],
                                                 'LOI': [5.0, 4.0, 3.0]})
    attrs: pd.Series = pd.Series(['grp_1', 'grp_1', 'grp_2'], name='group')

    mass: pd.DataFrame = pd.concat([mass_wet, mass_dry], axis='columns')
    if include_wet_mass is True and mass_dry is False:
        mass = mass_wet
    elif include_dry_mass is False and mass_dry is True:
        mass = mass_dry
    elif include_dry_mass is False and mass_dry is False:
        raise AssertionError('Arguments provided result in no mass column')

    if include_moisture is True:
        moisture: pd.DataFrame = (mass_wet - mass_dry) / mass_wet * 100
        moisture.name = 'H2O'
        res: pd.DataFrame = pd.concat([mass, moisture, chem, attrs], axis='columns')
    else:
        res: pd.DataFrame = pd.concat([mass, chem, attrs], axis='columns')

    res.index.name = 'index'

    return res


def dh_intervals(n: int = 5,
                 n_dh: int = 2,
                 analytes: Optional[Iterable[str]] = ('Fe', 'Al2O3')) -> pd.DataFrame:
    """Down-samples The drillhole data for testing

    Args:
        n: Number of samples
        n_dh: The number of drill-holes included
        analytes: the analytes to include
    Returns:

    """

    df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')

    drillholes: List[str] = []
    for i in range(0, n_dh):
        drillholes.append(random.choice(list(df_data['DHID'].unique())))

    df_data = df_data.query('DHID in @drillholes').groupby('DHID').sample(5)

    cols_to_drop = [col for col in is_compositional(df_data.columns) if (col not in analytes) and (col != 'H2O')]
    df_data.drop(columns=cols_to_drop, inplace=True)

    df_data.index.name = 'index'

    return df_data


def size_distribution() -> pd.DataFrame:
    d: Path = Path(__file__).parent
    print(d)
    df_psd: pd.DataFrame = pd.read_csv(d / 'size_distribution_ore_1.csv', index_col=0)
    return df_psd


def iron_ore_sample_data() -> pd.DataFrame:
    d: Path = Path(__file__).parent
    print(d)
    print('cwd files')
    print(os.listdir())
    print('cwd files')
    print(os.listdir(d))
    df_psd: pd.DataFrame = pd.read_csv(d / 'iron_ore_sample_data_A072391.csv', index_col=0)
    return df_psd
