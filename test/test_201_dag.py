from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.builder import DAGBuilder
from elphick.mass_composition.dag.transformers import Feed
from .fixtures import demo_data


def test_dag_single(demo_data):
    # Build a simple DAG of one node
    dag = DAGBuilder().add_step("feed", Feed()).make_dag()


def test_dag_single_fit(demo_data):
    # Build a simple DAG of one node
    obj_mc: MassComposition = MassComposition(demo_data, name='test_dag')

    dag = DAGBuilder().add_step("feed", Feed()).make_dag()
    dag.fit(obj_mc)
    dag


def test_dag_single_run(demo_data):
    # Build a simple DAG of one node
    obj_mc: MassComposition = MassComposition(demo_data, name='test_dag')

    dag = DAGBuilder().add_step("feed", Feed()).make_dag()
    dag.fit(obj_mc)
