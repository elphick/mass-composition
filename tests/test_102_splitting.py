from functools import partial

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline

from elphick.mass_composition import MassComposition, Stream
from elphick.mass_composition.mc_node import MCNode
from elphick.mass_composition.utils.partition import perfect
from elphick.mass_composition.utils.sklearn import PandasPipeline
from .fixtures import size_assay_data

from elphick.mass_composition import Flowsheet


def test_split_by_partition(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')
    partition = partial(perfect, d50=0.150, dim='size')
    mc_coarse, mc_fine = mc_size.split_by_partition(partition_definition=partition, name_1='lump', name_2='fines')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_coarse, mc_fine])

    # Get the MCNode object from the graph
    mc_split_node: MCNode = fs.graph.nodes[1]['mc']

    # Assertion 1: The mass of mc_size is equal to the sum of the masses of mc_coarse and mc_fine
    assert mc_split_node.balanced is True, "Mass is not preserved after the split"

    # Assertion 2: The names of mc_coarse and mc_fine are 'lump' and 'fines', respectively
    assert mc_coarse.name == 'lump', "mc_coarse name is not 'lump'"
    assert mc_fine.name == 'fines', "mc_fine name is not 'fines'"


def test_split_by_partition_stream(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')
    strm_size: Stream = Stream.from_mass_composition(mc_size)

    partition = partial(perfect, d50=0.150, dim='size')
    strm_coarse, strm_fine = strm_size.split_by_partition(partition_definition=partition, name_1='lump', name_2='fines')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([mc_size, strm_coarse, strm_fine])

    # Get the MCNode object from the graph
    mc_split_node: MCNode = fs.graph.nodes[1]['mc']

    # Assertion 1: The mass is preserved after the split
    assert mc_split_node.balanced is True, "Mass is not preserved after the split"

    # Assertion 2: The names of mc_coarse and mc_fine are 'lump' and 'fines', respectively
    assert strm_coarse.name == 'lump', "mc_coarse name is not 'lump'"
    assert strm_fine.name == 'fines', "mc_fine name is not 'fines'"


def test_split_by_function(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')

    def splitting_function(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy().div(2)
        return res

    partial(perfect, d50=0.150, dim='size')
    mc_1, mc_2 = mc_size.split_by_function(split_function=splitting_function, name_1='one', name_2='two')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_1, mc_2])

    # Assertion 1: The mass is preserved after the split
    assert fs.graph.nodes[1]['mc'].balanced is True, "Mass is not preserved after the split"


def test_split_by_function_stream(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')

    def splitting_function(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy().div(2)
        return res

    partial(perfect, d50=0.150, dim='size')

    strm_size: Stream = Stream.from_mass_composition(mc_size)
    strm_1, strm_2 = strm_size.split_by_function(split_function=splitting_function, name_1='one', name_2='two')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([strm_size, strm_1, strm_2])

    # Assertion 1: The mass is preserved after the split
    assert fs.graph.nodes[1]['mc'].balanced is True, "Mass is not preserved after the split"


def test_split_by_estimator(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')

    x = mc_size.data.to_dataframe()
    y = x.copy().div(2)
    dummy_regressor = PandasPipeline.from_pipeline(make_pipeline(DummyRegressor(strategy='mean'))).fit(X=x, y=y)

    mc_1, mc_2 = mc_size.split_by_estimator(estimator=dummy_regressor, name_1='one', name_2='two')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_1, mc_2])

    # Assertion 1: The mass is preserved after the split
    assert fs.graph.nodes[1]['mc'].balanced is True, "Mass is not preserved after the split"


def test_split_by_estimator_stream(size_assay_data):
    mc_size: MassComposition = MassComposition(size_assay_data, name='size sample')

    x = mc_size.data.to_dataframe()
    y = x.copy().div(2)
    dummy_regressor = PandasPipeline.from_pipeline(make_pipeline(DummyRegressor(strategy='mean'))).fit(X=x, y=y)

    strm_size: Stream = Stream.from_mass_composition(mc_size)
    strm_1, strm_2 = strm_size.split_by_estimator(estimator=dummy_regressor, name_1='one', name_2='two')

    # Create a Flowsheet from the MassComposition objects
    fs: Flowsheet = Flowsheet().from_streams([mc_size, strm_1, strm_2])

    # Assertion 1: The mass is preserved after the split
    assert fs.graph.nodes[1]['mc'].balanced is True, "Mass is not preserved after the split"
