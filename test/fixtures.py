import pandas as pd
import pytest

from elphick.mass_composition.datasets.sample_data import sample_data, size_by_assay

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
