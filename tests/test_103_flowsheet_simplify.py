from elphick.mass_composition import Flowsheet
# noinspection PyUnresolvedReferences
from .fixtures import demo_size_network_complex, size_assay_data


def test_simplify_flowsheet(demo_size_network_complex):
    fs: Flowsheet = demo_size_network_complex
    fs_overall: Flowsheet = fs.to_simple()

    # Check that the overall flowsheet is balanced
    assert fs_overall.balanced is True, "Mass is not preserved after the simplification"

    # Check that there is only one node of order > 1
    assert len([n for n, d in fs_overall.graph.degree() if d > 1]) == 1, \
        "There is more than one node of degree > 1"


