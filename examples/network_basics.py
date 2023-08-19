"""
Network Basics
==============

Related MassComposition objects can be managed as a network.  In the Process Engineering/Metallurgy
disciplines the network will often be called a _flowsheet_.

"""
from copy import deepcopy
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from elphick.mass_composition import MassComposition
from elphick.mass_composition.mc_node import MCNode
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.datasets.sample_data import sample_data

# %%
#
# Create some MassComposition objects
# -----------------------------------
#
# Create an object, and split it to create two more objects.

df_data: pd.DataFrame = sample_data()
obj_mc: MassComposition = MassComposition(df_data, name='Feed')
obj_mc_1, obj_mc_2 = obj_mc.split(0.4, name_1='stream 1', name_2='stream 2')

# %%
# Placeholder random nodes are created for each MassComposition object.
# This is done to capture the relationships implicitly defined by any math operations performed on the objects.

for obj in [obj_mc, obj_mc_1, obj_mc_2]:
    print(obj.name, obj.nodes)

# %%
#
# Create a MCNetwork object
# -------------------------
#
# This requires passing an Iterable of MassComposition objects

mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])

# %%
# Print the node object detail

for node in mcn.graph.nodes:
    print(mcn.graph.nodes[node]['mc'])

# %%
# Note that the random node placeholder integers have been renumbered for readability.

for obj in [obj_mc, obj_mc_1, obj_mc_2]:
    print(obj.name, obj.nodes)

# %%
# Print the overall network balanced status
#
# NOTE: presently this only includes node balance status
# edge balance status will assure the mass-moisture balance is satisfied

print(mcn.balanced)

# %%
# Plot the network.
# Imbalanced Nodes will appear red.  Later, Imbalanced Edges will also appear red.

mcn.plot()
plt

# %%
# Display the weight averages for all edges (streams) in the network (flowsheet)

df_report: pd.DataFrame = mcn.report()
df_report

# %%

df_report: pd.DataFrame = mcn.report(apply_formats=True)
df_report

# %%
# Plot the interactive network using plotly

fig = mcn.plot_network()
fig

# %%
# Plot the Sankey

fig = mcn.plot_sankey()
fig

# %%
# Demonstrate the table-plot

fig = mcn.table_plot(plot_type='sankey', table_pos='top', table_area=0.3)
fig

# %%

fig = mcn.table_plot(plot_type='network', table_pos='bottom', table_area=0.3)
fig

# %%
#
# Expand the Network with Math Operators
# --------------------------------------
#

obj_mc_3, obj_mc_4 = obj_mc_2.split(0.8, name_1='stream 3', name_2='stream 4')
obj_mc_5 = obj_mc_1.add(obj_mc_3, name='stream 5')

mcn2: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2, obj_mc_3, obj_mc_4, obj_mc_5])

fig = mcn2.table_plot(plot_type='sankey', table_pos='left')
fig

# %%
#
# Setting Node names
# ------------------

nodes_before: Dict[int, MCNode] = mcn.nodes_to_dict()
print({n: o.node_name for n, o in nodes_before.items()})

# %%
mcn.set_node_names(node_names={0: 'node_0', 1: 'node_1', 2: 'node_2', 3: 'node_3'})
nodes_after: Dict[int, MCNode] = mcn.nodes_to_dict()
print({n: o.node_name for n, o in nodes_after.items()})

# %%
#
# Setting Stream data
# -------------------
#
# First we show how to easily access the stream data as a dictionary

stream_data: Dict[str, MassComposition] = mcn.streams_to_dict()
print(stream_data.keys())

# %%
# We will replace stream 2 with the same data as stream 1.

new_stream: MassComposition = deepcopy(mcn.get_edge_by_name('stream 1'))
# we need to rename to avoid a creating a duplicate stream name
new_stream.name = 'stream 1 copy'
mcn.set_stream_data({'stream 2': new_stream})
print(mcn.streams_to_dict().keys())

# %%
# Of course the network is now unbalanced as highlighted in the Sankey

fig = mcn.table_plot()
fig

# %%
#
# Methods to modify relationships
# -------------------------------
#
# Sometimes the network that is automatically created may not be what you are after - for example flow may be in
# the wrong direction.  We'll learn how to modify an existing network, by picking up the network above.
#
# Let's break the links for the _stream 1_.

mcn.reset_stream_nodes(stream="stream 1")
fig = mcn.table_plot()
fig

# %%
# We'll now break all remaining connections (we could have done this from the start).

mcn.reset_stream_nodes()
fig = mcn.table_plot()
fig

# %%
# Now we'll create some linkages - of course they will be completely rubbish and not balance.

mcn.set_stream_parent(stream="stream 1", parent="Feed")
mcn.set_stream_child(stream="stream 1", child="stream 1 copy")
fig = mcn.table_plot()
fig

# %%
# Perhaps less useful, but possible, we can build relationships by setting nodes directly.

mcn.reset_stream_nodes()
mcn.set_stream_nodes(stream="stream 1", nodes=(1, 2))
mcn.set_stream_nodes(stream="stream 1 copy", nodes=(2, 3))
fig = mcn.table_plot()
fig
