from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag import DAG
from tests.fixtures import demo_data


def test_dag_instance(demo_data):
    # Build a simple DAG of one node
    dag = DAG(n_jobs=1)


def test_dag_with_node(demo_data):
    # Build a simple DAG of one node
    mc_sample: MassComposition = MassComposition(demo_data, name='sample')
    dag = DAG(n_jobs=1).add_node('feed', DAG.input, [])


def test_dag_fit(demo_data):
    # Build a simple DAG of one node and run
    mc_sample: MassComposition = MassComposition(demo_data, name='sample')
    dag = DAG(n_jobs=1).add_node('feed', DAG.input, [])
    dag.run({'feed': mc_sample})
