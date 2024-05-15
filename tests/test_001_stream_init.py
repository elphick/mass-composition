import pandas as pd

from elphick.mass_composition import MassComposition, Stream
from .fixtures import demo_data


def test_stream_init(demo_data):
    # Initialize a MassComposition object
    obj_mc: MassComposition = MassComposition(demo_data, name='test')

    # Initialize a Stream object from the MassComposition object
    obj_stream: Stream = Stream.from_mass_composition(obj_mc)

    assert obj_stream.name == obj_mc.name, "The name attribute was not correctly set during initialization"

    # Assert that the source_node and destination_node attributes are correctly set
    assert obj_stream.source_node == obj_mc._nodes[0], ("The source_node attribute was not correctly set during"
                                                        " initialization")
    assert obj_stream.destination_node == obj_mc._nodes[1], ("The destination_node attribute was not correctly set"
                                                             " during initialization")

    pd.testing.assert_frame_equal(obj_stream.data.to_dataframe(), obj_mc.data.to_dataframe(), check_dtype=True)

    # Assert that the constraints attribute is correctly set
    assert obj_stream.constraints == obj_mc.constraints, ("The constraints attribute was not correctly set during "
                                                          "initialization")

    # Assert that the status attribute is correctly set
    assert obj_stream.status == obj_mc.status, "The status attribute was not correctly set during initialization"

    # Assert that the variables attribute is correctly set
    assert obj_stream.variables == obj_mc.variables, ("The variables attribute was not correctly set during "
                                                      "initialization")