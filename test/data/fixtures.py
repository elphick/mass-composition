import pandas as pd
import pytest

from test.data import sample_data
from test.data.sample_data import size_by_assay


@pytest.fixture
def demo_data():
    data: pd.DataFrame = sample_data()
    return data


@pytest.fixture
def size_assay_data():
    data: pd.DataFrame = size_by_assay()
    return data