"""
DAG to Define and Simulate
==========================

Splitting and partitioning will preserve the relationships between streams enabling network creation for simple cases.
In more complex cases the DAG (Directed Acyclic Graph) construct can be useful to define the relationships between
streams resulting from transformations (operations) on the streams.

The DAG can be used to define the network and, with the run method, simulate the network to produce the final results.

"""
import logging
from copy import deepcopy

import plotly

from elphick.mass_composition import MassComposition, Stream
from elphick.mass_composition.dag import DAG
from elphick.mass_composition.datasets.sample_data import sample_data
from elphick.mass_composition.flowsheet import Flowsheet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Define the DAG
# --------------
#
# The DAG is defined by adding nodes to the graph.  Each node is an input, output or Stream operation
# (e.g. add, split, etc.).  The nodes are connected by the streams they operate on.

mc_sample: MassComposition = MassComposition(sample_data(), name='sample')

dag = DAG(n_jobs=1)
dag.add_input(name='feed_1')
dag.add_input(name='feed_2')
dag.add_step(name='joiner', operation=Stream.add, streams=['feed_1', 'feed_2'], kwargs={'name': 'feed'})
dag.add_step(name='split', operation=Stream.split, streams=['feed'],
             kwargs={'fraction': 0.3, 'name_1': 'lump', 'name_2': 'fines'})
dag.add_step(name='split_2', operation=Stream.split, streams=['lump'],
             kwargs={'fraction': 0.3, 'name_1': 'lumpier', 'name_2': 'less_lumpy'})
dag.add_step(name='split_3', operation=Stream.split, streams=['fines'],
             kwargs={'fraction': 0.3, 'name_1': 'finer', 'name_2': 'less_fine'})
dag.add_step(name='joiner_1', operation=Stream.add, streams=['less_lumpy', 'less_fine'],
             kwargs={'name': 'mix_1'})
dag.add_step(name='joiner_2', operation=Stream.add, streams=['lumpier', 'finer'],
             kwargs={'name': 'mix_2'})
dag.add_output(name='product_1', stream='mix_1')
dag.add_output(name='product_2', stream='mix_2')


# %%
# Run the DAG
# -----------
#
# The dag is run by providing MassComposition (or Stream) objects for all inputs.  They must be compatible i.e. have the
# same indexes.

dag.run({'feed_1': mc_sample,
         'feed_2': deepcopy(mc_sample).rename('sample_2')  # names must be unique
         }, progress_bar=False)

# %%
# Create a Flowsheet object from the dag, enabling all the usual network plotting and analysis methods.

fs: Flowsheet = Flowsheet.from_dag(dag)

fig = fs.plot_network()
plotly.io.show(fig)

# %%

fig = fs.table_plot(plot_type='sankey')
fig
