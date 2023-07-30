"""
Mass Balancing
==============

In the examples so far, MCNetwork objects are created by math operations, so they inherently balance.
A common problem in Mineral Processing (Metallurgy/Chemical/Process Engineering) is mass (or metallurgical) balancing.
When auditing a processing plant the data is collected by measurement and sampling/assaying. This data will never
balance of course due to sampling and measurement errors.

There exists a fundamental optimisation process that can balance the overall mass and components across a
system (network/flowsheet).  This example demonstrates that functionality.

"""
import logging
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import plotly

from elphick.mass_composition import MassComposition
from elphick.mass_composition.balance import MCBalance
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.partition import napier_munn
from elphick.mass_composition.datasets.sample_data import size_by_assay

# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = size_by_assay()
df_data

# %%
# Create the object

mc_size: MassComposition = MassComposition(df_data, name='size sample')
print(mc_size)
mc_size.aggregate()

# %%
#
# We partially initialise a partition function
# The dim argument is added to inform the split method which dimension to apply the function/split to

partition = partial(napier_munn, d50=0.150, ep=0.1, dim='size')

# %%
# Create a Network that balances
# ------------------------------
#
# Separate the object using the defined partition

mc_coarse, mc_fine = mc_size.partition(definition=partition, name_1='coarse', name_2='fine')

mcn: MCNetwork = MCNetwork().from_streams([mc_size, mc_coarse, mc_fine])
print(mcn.balanced)

fig = mcn.table_plot(plot_type='network', table_pos='left', table_area=0.3)
fig

# %%
#
# Demonstrate that the data balances with the balance plot

fig = mcn.plot_balance()
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

# %%
#
# The balance plot can be colored by a specified column or index/dimension.

fig = mcn.plot_balance(color='size')
fig

# %%
# Create an imbalanced network
# ----------------------------
#
# Modify one stream to corrupt the balance

df_coarse_2 = mc_coarse.data.to_dataframe().apply(lambda x: np.random.normal(loc=x, scale=np.std(x)))
mc_coarse_2: MassComposition = MassComposition(data=df_coarse_2, name='coarse')
mc_coarse_2 = mc_coarse_2.set_parent(mc_size)

# create a new network - which does not balance
mcn_ub: MCNetwork = MCNetwork().from_streams([mc_size, mc_coarse_2, mc_fine])
print(mcn_ub.balanced)

fig = mcn_ub.table_plot(plot_type='network', table_pos='left', table_area=0.3)
fig

# %%
fig = mcn_ub.plot_balance()
fig

# %%
# Balance the Flowsheet
# ---------------------

# %%
# ..  note::
#
#     This example has not yet been completed...

mcb: MCBalance = MCBalance(mcn=mcn_ub)

# SD configuration
# df_sds: pd.DataFrame = mcb.create_balance_config(best_measurements='input')

# cost functions
cfs: Dict = mcb._create_cost_functions()
# check for a zero cost when passing the measured values
for k, v in cfs.items():
    x = mcb.mcn.to_dataframe().loc[k, :].drop(columns=['mass_wet']).values.ravel()
    y = v(x=x)
    print(k, y)

df_bal: pd.DataFrame = mcb.optimise()

# create a network using the balanced data
mcn_bal: MCNetwork = MCNetwork.from_dataframe(df=df_bal, name='balanced', mc_name_col='name')
fig = mcn_bal.plot_parallel(color='name')
fig.show()
