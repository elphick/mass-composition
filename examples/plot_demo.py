"""
Plot Demo
=========

Demonstrating the mass-composition plot methods.
"""
from pathlib import Path

import pandas as pd
from plotly.graph_objs import Figure

from elphick.mc.mass_composition.data.sample_data import sample_data
from elphick.mc.mass_composition import MassComposition

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
fig.show()

# %%
#
# Create an interactive parallel plot with only the components

fig2 = obj_mc.plot_parallel(composition_only=True)
fig2.show()

# %%
#
# Create a parallel plot with color

fig3 = obj_mc.plot_parallel(color='group')
fig3.show()

# %%
#
# Create a ternary diagram for 3 composition variables

fig4 = obj_mc.plot_ternary(variables=['SiO2', 'Al2O3', 'LOI'], color='group')
# save the figure for use as the sphinx-gallery thumbnail
fig4.write_image('../doc/source/_static/ternary.png')
# sphinx_gallery_thumbnail_path = '_static/ternary.png'
fig4.show()

print(Path(__file__).name, 'done')
