"""
Network Basics
==============

Related MassComposition objects are managed as a network.

"""

import pandas as pd
from matplotlib import pyplot as plt

from elphick.mass_composition import MassComposition
from elphick.mass_composition.mc_network import MCNetwork
from test.data import sample_data


# %%
#
# Create some MassComposition objects
# -----------------------------------
#
# Create an object, and split it to create two more objects.

df_data: pd.DataFrame = sample_data()
obj_mc: MassComposition = MassComposition(df_data, name='Feed')
obj_mc_1, obj_mc_2 = obj_mc.split(0.4, name_1='stream 1', name_2='stream 2')

for obj in [obj_mc, obj_mc_1, obj_mc_2]:
    print(obj.name, obj.data.nodes)

# %%
#
# Create a MCNetwork object
# -------------------------
#
# This requires passing an Iterable of MassComposition objects

mcn: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2])

# %%
# Print the node object detail

for node in mcn.nodes:
    print(mcn.nodes[node]['mc'])

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
plt.show()

# %%
# Display the weight averages for all edges (streams) in the network (flowsheet)

df_report: pd.DataFrame = mcn.report()
df_report

# %%
# Plot the interactive network using plotly

fig = mcn.plot_network()
fig

# %%
# ..  note::
#     This plot requires development to include labels and colors per the matplotlib version.

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
# ..  note::
#     This highlights that the "auto node assignment" logic fails.  Need to develop a new approach.
#     https://github.com/Elphick/mass-composition/issues/8#issuecomment-1492893274

obj_mc_3, obj_mc_4 = obj_mc_2.split(0.8, name_1='stream 3', name_2='stream 4')
obj_mc_5 = obj_mc_1.add(obj_mc_3, name='stream 5')

for obj in [obj_mc, obj_mc_1, obj_mc_2, obj_mc_3, obj_mc_4]:
    print(obj.name, obj.data.nodes)


mcn2: MCNetwork = MCNetwork().from_streams([obj_mc, obj_mc_1, obj_mc_2, obj_mc_3, obj_mc_4, obj_mc_5])

fig = mcn2.table_plot(plot_type='sankey', table_pos='top', table_area=0.3)
fig

# %%
#  .. admonition:: Info
#
#     While the plot above shows a problem to be resolved, it also demonstrates the feature where
#     unbalanced nodes are highlighted red..
