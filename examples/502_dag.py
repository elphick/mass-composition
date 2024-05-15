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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Define the DAG
# --------------
#
# The DAG is defined by adding nodes to the graph.  Each node is a MassComposition operation
# (or a DAG.input or DAG_output).

mc_sample: MassComposition = MassComposition(sample_data(), name='sample')

dag = DAG(n_jobs=1)
dag.add_input(name='feed_1')
dag.add_input(name='feed_2')
dag.add_step(name='joiner', operation=Stream.add, streams=['feed_1', 'feed_2'], kwargs={'name': 'feed'})
dag.add_step(name='split', operation=Stream.split, streams=['feed'],
             kwargs={'fraction': 0.3, 'name_1': 'lump', 'name_2': 'fines'})
dag.add_step(name='split_2', operation=Stream.split, streams=['lump'],
             kwargs={'fraction': 0.3, 'name_1': 'lumpier', 'name_2': 'less_lumpy'})
dag.add_output(name='lumpier', stream='lumpier')
dag.add_output(name='mid', stream='less_lumpy')
dag.add_output(name='fines', stream='fines')


# %%
# Run the DAG
# -----------
#
# The dag is run by providing MassComposition objects for all inputs.  They must be compatible i.e. have the
# same indexes.

dag.run({'feed_1': mc_sample,
         'feed_2': deepcopy(mc_sample).rename('sample_2')  # names must be unique
         })

# %%
# Create a Flowsheet object from the dag, enabling all the usual network plotting and analysis methods.

fs: Flowsheet = Flowsheet.from_dag(dag)

fig = fs.plot_network()
plotly.io.show(fig)

# %%

fig = fs.table_plot(plot_type='sankey')
fig
