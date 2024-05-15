from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from pandas import Interval

from elphick.mass_composition.utils.partition import napier_munn
# noinspection PyUnresolvedReferences
from tests.fixtures import demo_data, size_assay_data
from elphick.mass_composition import MassComposition, Flowsheet


def test_indexes(demo_data):
    obj_mc: MassComposition = MassComposition(demo_data, name='one')
    obj_mc_2: MassComposition = MassComposition(demo_data.drop(index=[0]), name='two').set_parent_node(obj_mc)

    with pytest.raises(KeyError):
        fs: Flowsheet = Flowsheet().from_streams([obj_mc, obj_mc_2])


def test_missing_sizes(size_assay_data):
    # We get some demo sizing data, split it with a partition, and manually drop sieves for the undersize stream.
    mc_feed: MassComposition = MassComposition(size_assay_data, name='FEED')
    # We partially initialise a partition function
    partition = partial(napier_munn, d50=0.150, ep=0.05, dim='size')
    # Create a Network using the partition
    mc_oversize, mc_undersize = mc_feed.apply_partition(definition=partition, name_1='OS', name_2='US')
    # drop the two size fractions from mc_fine that have near zero mass
    df_fine: pd.DataFrame = mc_undersize.data.to_dataframe()
    df_fine = df_fine.loc[df_fine.index.left < 0.5, :]
    mc_undersize.set_data(df_fine)

    fs: Flowsheet = Flowsheet().from_streams([mc_feed, mc_oversize, mc_undersize])
    df_test: pd.DataFrame = fs.get_edge_by_name('US').data.to_dataframe()
    df_test = df_test.loc[df_test.index.left >= 0.5, :]
    d_expected: Dict = {'mass_wet': {Interval(0.85, 2.0, closed='left'): 0.0,
                                     Interval(0.5, 0.85, closed='left'): 0.0},
                        'mass_dry': {Interval(0.85, 2.0, closed='left'): 0.0,
                                     Interval(0.5, 0.85, closed='left'): 0.0},
                        'H2O': {Interval(0.85, 2.0, closed='left'): np.nan,
                                Interval(0.5, 0.85, closed='left'): np.nan},
                        'Fe': {Interval(0.85, 2.0, closed='left'): 0.0,
                               Interval(0.5, 0.85, closed='left'): 0.0},
                        'SiO2': {Interval(0.85, 2.0, closed='left'): 0.0,
                                 Interval(0.5, 0.85, closed='left'): 0.0},
                        'Al2O3': {Interval(0.85, 2.0, closed='left'): 0.0,
                                  Interval(0.5, 0.85, closed='left'): 0.0}}
    df_expected: pd.DataFrame = pd.DataFrame.from_dict(d_expected)
    df_expected.index.names = ['size']
    pd.testing.assert_frame_equal(df_expected, df_test)
