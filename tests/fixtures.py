from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from elphick.mass_composition import MassComposition, Flowsheet
from elphick.mass_composition.datasets.sample_data import sample_data, size_by_assay
from elphick.mass_composition.utils.partition import perfect, napier_munn


@pytest.fixture
def demo_data():
    data: pd.DataFrame = sample_data()
    return data


@pytest.fixture
def demo_data_2():
    data: pd.DataFrame = sample_data(include_wet_mass=True,
                                     include_dry_mass=False,
                                     include_moisture=True)
    return data


@pytest.fixture
def size_assay_data():
    data: pd.DataFrame = size_by_assay()
    return data


@pytest.fixture
def demo_size_network(size_assay_data) -> Flowsheet:
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')
    partition = partial(perfect, d50=0.150, dim='size')
    mc_coarse, mc_fine = mc_size.split_by_partition(partition_definition=partition)
    mc_coarse.name = 'coarse'
    mc_fine.name = 'fine'
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_coarse, mc_fine])
    return fs


@pytest.fixture
def demo_size_network_complex(size_assay_data) -> Flowsheet:
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')
    mc_ideal_feed, mc_sim_feed = mc_size.split(0.5, 'ideal feed', 'sim feed')
    part_ideal = partial(perfect, d50=0.150, dim='size')
    part_sim = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    mc_ideal_coarse, mc_ideal_fine = mc_ideal_feed.split_by_partition(partition_definition=part_ideal,
                                                                      name_1='ideal_coarse', name_2='ideal_fine')
    mc_sim_coarse, mc_sim_fine = mc_sim_feed.split_by_partition(partition_definition=part_sim, name_1='sim_coarse',
                                                                name_2='sim_fine')

    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_ideal_feed, mc_sim_feed,
                                              mc_ideal_coarse, mc_ideal_fine,
                                              mc_sim_coarse, mc_sim_fine])
    return fs


@pytest.fixture()
def script_loc(request):
    """Return the directory of the currently running test script"""

    # uses .join instead of .dirname, so we get a LocalPath object instead of
    # a string. LocalPath.join calls normpath for us when joining the path
    return Path(request.fspath.join('..'))
