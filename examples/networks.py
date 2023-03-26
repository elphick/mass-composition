"""
Networks
========

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
obj_mc_1, obj_mc_2 = obj_mc.split(0.4)

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

mcn.plot_network()
plt.show()

# %%
# Display the weight averages for all edges (streams) in the network (flowsheet)

df_report: pd.DataFrame = mcn.report()
df_report

