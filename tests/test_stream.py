import pandas as pd
import pytest

from flowsheet import Stream
# from fixtures import demo_data
from elphick.mc.mass_composition.data.sample_data import sample_data


@pytest.fixture()
def demo_data():
    data: pd.DataFrame = sample_data()
    return data


def test_stream_init(demo_data):
    obj_strm: Stream = Stream.from_dataframe(data=demo_data, name='my_name')
    assert True
