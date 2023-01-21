"""
Stream Example
==============

A Stream object is a superclass of a MassComposition object.
It has additional properties - a name for example.

This is useful when multiple objects are managed within a system (or flowsheet)
"""

# %%

import pandas as pd
from plotly.graph_objs import Figure

from flowsheet.stream import Stream
from elphick.mc.mass_composition import MassComposition

# %%
#
# Create a Stream object
# ----------------------
# #
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')
print(df_data.shape)
print(df_data.head())

obj_stream: Stream = Stream.from_dataframe(df_data, name='stream 1')
print(obj_stream)

obj_mc: MassComposition = MassComposition(df_data)
obj_stream_mc: Stream = Stream.from_mass_composition(obj_mc, name='stream from mc')
print(obj_stream_mc)

stream_1, stream_2 = obj_stream.split(fraction=0.6)
print(stream_1)

stream_3 = stream_1 + stream_2
print(stream_3)

fig: Figure = obj_stream.plot_parallel(color='Fe')
fig.show()

# %%
#
# Create a ternary diagram for 3 composition variables

fig2 = obj_stream.plot_ternary(variables=['SiO2', 'Al2O3', 'Fe'], color='DHID')
fig2.show()

print('done')
