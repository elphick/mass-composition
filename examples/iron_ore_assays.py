"""
Iron Ore Assays
===============

Demonstrate with some real data
"""

# %%

import pandas as pd
import plotly
from elphick.mass_composition.demo_data.sample_data import iron_ore_sample_data
from plotly.graph_objs import Figure

from elphick.mass_composition import MassComposition

# %%
#
# Create a MassComposition object
# -------------------------------
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = iron_ore_sample_data()
name = 'A072391'

print(df_data.shape)
df_data.head()

# %%
# ...and create a MassComposition from the DataFrame.

obj_mc: MassComposition = MassComposition(df_data, name=name)
obj_mc.aggregate(group_var='DHID')

# %%
#
# Parallel plots
# --------------

obj_mc: MassComposition = MassComposition(df_data.reset_index().set_index(['DHID', 'interval_from', 'interval_to']),
                                          name=name)

fig: Figure = obj_mc.plot_parallel(color='Fe')
fig

# %%

fig: Figure = obj_mc.plot_parallel(color='Fe', plot_interval_edges=True)
fig

# %%

# with selected variables
fig: Figure = obj_mc.plot_parallel(color='Fe', vars_include=['mass_wet', 'H2O', 'Fe', 'SiO2'])
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery
