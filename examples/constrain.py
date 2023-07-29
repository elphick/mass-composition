"""
Constrain
=========

A simple example that demonstrates the constrain method of mass-composition.

It is possible that a MassComposition object is created from a Machine Learning Model estimation.
If either the ML model is over-fitted, or the features supplied to create the estimation are out of range,
improbable results can be generated. The constrain method will provide a way to manage these outliers.

"""

import pandas as pd

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import sample_data

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
df_data

# %%
#
# Construct a MassComposition object

obj_mc: MassComposition = MassComposition(df_data)
obj_mc.data.to_dataframe()

# %%
#
# Constraining by Clip
# --------------------
#
# Constraining by clip simpy clips the mass or composition values.  The simplest way to constrain is with
# a tuple of the limits.


obj_1: MassComposition = obj_mc.constrain(clip_mass=(85, 100))
obj_1.data.to_dataframe()

# %%
# Notice that the mass has been constrained for some records and the H2O has been modified accordingly.
#
# More granularity is possible by passing a dict[variable: tuple_of_limits]

# %%
obj_2: MassComposition = obj_mc.constrain(clip_mass={'mass_wet': (0, 100)})
obj_2.data.to_dataframe()

# %%
#
# Constraining Relative to Another Object
# ---------------------------------------
#
# Sometimes constraining relative to another object is useful.  This can be described as "constraining by recovery".
# The object os converted to absolute mass (where components are converted to mass units) and divided by the
# reference (other) object, also converted to mass units.  In mineral processing, this is known as recovery.
#
# First we'll make another object to act as our reference.

# %%
obj_other: MassComposition = obj_mc.add(obj_mc, name='feed')

obj_3: MassComposition = obj_mc.constrain(relative_mass=(0.0, 0.1), other=obj_other)
obj_3.data.to_dataframe()

# %%
#
# Here we constrain Fe to 10% recovery of 2 x the original object...

# %%
obj_4: MassComposition = obj_mc.constrain(relative_composition={'Fe': (0.0, 0.1)}, other=obj_other)
obj_4.data.to_dataframe()

# %%
# Arguments can be combined to perform multiple constraints in one call.

# %%
obj_5: MassComposition = obj_mc.constrain(clip_mass=(85, 100),
                                          relative_composition={'Fe': (0.0, 0.1)}, other=obj_other)
obj_5.data.to_dataframe()
