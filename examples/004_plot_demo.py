"""
Plot Demo
=========

Demonstrating the mass-composition plot methods.
"""

import pandas as pd
import plotly
from plotly.graph_objs import Figure
from elphick.mass_composition.datasets.sample_data import sample_data
from elphick.mass_composition import MassComposition

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Construct a MassComposition object and standardise the chemistry variables

obj_mc: MassComposition = MassComposition(df_data)
print(obj_mc)

# %%
#
# Create an interactive parallel plot

fig: Figure = obj_mc.plot_parallel()
fig

# %%
#
# Create an interactive parallel plot with only the components

fig2 = obj_mc.plot_parallel(vars_include=['mass_wet', 'H2O', 'Fe'])
fig2

# %%
#
# Create a parallel plot with color

fig3 = obj_mc.plot_parallel(color='group')
fig3

# %%
#
# Create a ternary diagram for 3 composition variables

fig4 = obj_mc.plot_ternary(variables=['SiO2', 'Al2O3', 'LOI'], color='group')
# noinspection PyTypeChecker
plotly.io.show(fig4)  # this call to show will set the thumbnail for use in the gallery

