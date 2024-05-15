import pytest

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag import DAG
from .fixtures import demo_data


def test_dag_instance(demo_data):
    # Build a simple DAG of one node
    dag = DAG(n_jobs=1)


def test_dag_with_node(demo_data):
    # Build a simple DAG of one node
    mc_sample: MassComposition = MassComposition(demo_data, name='sample')
    dag = DAG(n_jobs=1).add_input(name='feed')


def test_dag_fit(demo_data):
    # Build a simple DAG of one node and run
    mc_sample: MassComposition = MassComposition(demo_data, name='sample')
    dag = DAG(n_jobs=1).add_input(name='feed')
    with pytest.raises(ValueError, match="Orphan nodes: \['feed'\] exist.  Please check your configuration."):
        dag.run({'feed': mc_sample})
