"""
DAG with Partitions
===================

In the context of this script, partitions are used to divide the data into different segments based on certain
criteria. These partitions are defined using the napier_munn function, which is partially applied to set the d50
and ep parameters. The dim argument is used to select the dimension to partition on.

The partitions are then used in the Directed Acyclic Graph (DAG) to define the relationships between streams
resulting from transformations (operations) on the streams.

Each node in the DAG represents an operation on one or more streams, and the edges represent the flow of data
from one operation to the next.  The DAG, along with the defined partitions, can be used to simulate
the network of operations and produce the final results. This approach allows for the management of
complex relationships between streams in stream operations.
"""
import logging
from copy import deepcopy
from functools import partial

import plotly

from elphick.mass_composition import MassComposition, Stream
from elphick.mass_composition.dag import DAG
from elphick.mass_composition.datasets.sample_data import size_by_assay
from elphick.mass_composition.flowsheet import Flowsheet
from elphick.mass_composition.utils.partition import napier_munn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Define the Partitions
# ---------------------
#
# These partitions are defined in the `napier_munn` function.  The function is partially applied to set the d50 and ep.
# The `dim` argument is used to select the dimension to partition on.
# These have no basis in reality and are for illustrative purposes only.

part_screen = partial(napier_munn, d50=0.5, ep=0.2, dim='size')
part_rgr_cyclone = partial(napier_munn, d50=0.045, ep=0.1, dim='size')
part_clr_cyclone = partial(napier_munn, d50=0.038, ep=0.1, dim='size')
part_scav_cyclone = partial(napier_munn, d50=0.045, ep=0.1, dim='size')

# %%
# Define the DAG
# --------------
#
# The DAG is defined by adding nodes to the graph.  Each node is an input, output or Stream operation
# (e.g. add, split, etc.).  The nodes are connected by the streams they operate on.

mc_sample: MassComposition = MassComposition(size_by_assay(), name='sample')

dag = DAG(n_jobs=1)
dag.add_input(name='feed')
dag.add_step(name='screen', operation=Stream.split_by_partition, streams=['feed'],
             kwargs={'partition_definition': part_screen, 'name_1': 'oversize', 'name_2': 'undersize'})
dag.add_step(name='rougher', operation=Stream.split_by_partition, streams=['undersize'],
             kwargs={'partition_definition': part_rgr_cyclone, 'name_1': 'rgr_uf', 'name_2': 'rgr_of'})
dag.add_step(name='cleaner', operation=Stream.split_by_partition, streams=['rgr_uf'],
             kwargs={'partition_definition': part_clr_cyclone, 'name_1': 'clr_uf', 'name_2': 'clr_of'})
dag.add_step(name='scavenger', operation=Stream.split_by_partition, streams=['rgr_of'],
             kwargs={'partition_definition': part_scav_cyclone, 'name_1': 'scav_uf', 'name_2': 'scav_of'})
dag.add_step(name='overflow', operation=Stream.add, streams=['scav_of', 'clr_of'],
             kwargs={'name': 'tailings'})
dag.add_step(name='joiner', operation=Stream.add, streams=['oversize', 'clr_uf', 'scav_uf'],
             kwargs={'name': 'product'})
dag.add_output(name='reject', stream='tailings')
dag.add_output(name='product', stream='product')

# %%
# Run the DAG
# -----------
#
# The dag is run by providing MassComposition (or Stream) objects for all inputs.  They must be compatible i.e. have the
# same indexes.

dag.run({'feed': mc_sample}, progress_bar=True)

# %%
# Create a Flowsheet object from the dag, enabling all the usual network plotting and analysis methods.

fs: Flowsheet = Flowsheet.from_dag(dag)

fig = fs.plot_network()
fig

# %%

fig = fs.table_plot(plot_type='sankey', sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=52,
                    sankey_vmax=70)
plotly.io.show(fig)
