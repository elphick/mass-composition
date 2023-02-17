"""
Iron Ore Assays
===============

Demonstrate with some real data
"""

# %%
from pathlib import Path

import pandas as pd
import plotly
from IPython.core.display_functions import display

from plotly.graph_objs import Figure

from elphick.mass_composition import MassComposition


# %%
#
# Create a MassComposition object
# -------------------------------
# We get some demo data in the form of a pandas DataFrame

filepath: Path = Path('../test/data/iron_ore_sample_data_A072391.csv')
name: str = filepath.stem.split('_')[-1]
df_data: pd.DataFrame = pd.read_csv(filepath, index_col='index')
print(df_data.shape)
print(df_data.head())

obj_mc: MassComposition = MassComposition(df_data, name=name)
display(obj_mc.aggregate(group_var='DHID'))




# %%
#
# Parallel plots
# --------------

obj_mc: MassComposition = MassComposition(df_data.reset_index().set_index(['DHID', 'interval_from', 'interval_to']), name=name)

fig: Figure = obj_mc.plot_parallel(color='Fe')
fig

# %%

fig: Figure = obj_mc.plot_parallel(color='Fe', plot_interval_edges=True)
fig

# %%

# with selected variables
fig: Figure = obj_mc.plot_parallel(color='Fe', var_subset=['mass_wet', 'H2O', 'Fe', 'SiO2'])
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for the gallery

