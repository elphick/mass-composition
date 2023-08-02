from functools import partial

import pandas as pd
import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import sample_data, size_by_assay
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import perfect


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
def demo_size_network(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')
    partition = partial(perfect, d50=0.150, dim='size')
    mc_coarse, mc_fine = mc_size.partition(definition=partition)
    mc_coarse.name = 'coarse'
    mc_fine.name = 'fine'
    mcn: MCNetwork = MCNetwork().from_streams([mc_size, mc_coarse, mc_fine])
    return mcn
