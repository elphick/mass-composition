import pandas as pd
import pytest

from elphick.mc.mass_composition.data.sample_data import sample_data


@pytest.fixture
def demo_data():
    data: pd.DataFrame = sample_data()
    return data
