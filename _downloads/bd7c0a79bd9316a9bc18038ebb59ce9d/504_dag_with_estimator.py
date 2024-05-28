"""
DAG with Estimator
===================

Flowsheet can be used to apply an estimator in a process flowsheet.  This example demonstrates how to use a DAG
to define a flowsheet that applies a lump estimator to a feed stream.

The focus will not be on the model development, but rather on the simulation.  The model is a simple RandomForest
regressor that predicts the lump mass and composition from the feed stream.

.. note::
   This example uses the `estimator` extras.  ensure you have installed like ``poetry install -E estimator``.

"""
import logging

# This import at the top to guard against the estimator extras not being installed
from elphick.mass_composition.utils.sklearn import PandasPipeline

import pandas as pd
import plotly
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elphick.mass_composition import MassComposition, Stream
from elphick.mass_composition.dag import DAG
from elphick.mass_composition.datasets.sample_data import iron_ore_met_sample_data
from elphick.mass_composition.flowsheet import Flowsheet

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Load Data
# ---------
#
# We load some metallurgical data from a drill program, REF: A072391
# Since we are not concerned about the model performance in this example, we'll convert the categorical feature
# bulk_hole_no to an integer

df: pd.DataFrame = iron_ore_met_sample_data()

base_components = ['fe', 'p', 'sio2', 'al2o3', 'loi']
cols_x = ['dry_weight_lump_kg'] + [f'head_{comp}' for comp in base_components] + ['bulk_hole_no']
cols_y = ['lump_pct'] + [f'lump_{comp}' for comp in base_components]

df = df.loc[:, cols_x + cols_y].query('lump_pct>0').dropna(how='any')
df = df.rename(columns={'dry_weight_lump_kg': 'head_mass_dry'})
df['bulk_hole_no'] = df['bulk_hole_no'].astype('category').cat.codes

logger.info(df.shape)
df.head()

# %%
# Build a model
# -------------

X: pd.DataFrame = df[[col for col in df.columns if col not in cols_y]]
y: pd.DataFrame = df[cols_y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# The model needs to be wrapped in a PandasPipeline object to ensure that the column names are preserved.

pipe: PandasPipeline = PandasPipeline.from_pipeline(
    make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)))

pipe

# %%
# Test the model
# --------------
# The model can be called directly to predict the lump percentage and composition from the feed stream.
# We will pass in a dataframe with the same columns as the training data.

y_pred = pipe.fit(X_train.drop(columns=['head_mass_dry']), y_train).predict(X_test)
logger.info(f'Test score: {pipe.score(X_test, y_test)}')
y_pred.head()

# %%
# Create a Head MassComposition object
# ------------------------------------
# Now we will create a MassComposition object and use it to apply the model to the feed stream.

head: MassComposition = MassComposition(data=X_test.drop(columns=['bulk_hole_no']), name='head',
                                        mass_dry_var='head_mass_dry')
lump, fines = head.split_by_estimator(estimator=pipe, name_2='fines',
                                      mass_recovery_column='lump_pct', mass_recovery_max=100,
                                      extra_features=X_test['bulk_hole_no'])
lump.data.to_dataframe().head()

# %%
fines.data.to_dataframe().head()

# %%
# Define the DAG
# --------------
#
# First we define a simple DAG, where the feed stream is split into two streams, lump and fines.
# The lump estimator requires the usual mass-composition variables plus an addition feature/variable
# called `bulk_hole_no`. Since the `bulk_hole_no` is available in the feed stream, it is immediately accessible
# to the estimator.

head: MassComposition = MassComposition(data=X_test, name='head',
                                        mass_dry_var='head_mass_dry')

dag = DAG(name='A072391', n_jobs=1)
dag.add_input(name='head')
dag.add_step(name='screen', operation=Stream.split_by_estimator, streams=['head'],
             kwargs={'estimator': pipe, 'name_1': 'lump', 'name_2': 'fines',
                     'mass_recovery_column': 'lump_pct', 'mass_recovery_max': 100})
dag.add_output(name='lump', stream='lump')
dag.add_output(name='fines', stream='fines')
dag.run(input_streams={'head': head}, progress_bar=True)

fig = Flowsheet.from_dag(dag).plot_network()
fig

# %%
# More Complex DAG
# ----------------
# This DAG is to test a more complex flowsheet where the estimator may have all the features
# immediately available in the parent stream.
#
# .. note::
#    This example works, but it does so since all attribute (extra) variables are passed all the way around
#    the network in the current design.  This is to be changed in the future to allow for more efficient processing.
#    Once attributes are no longer passed, changes will be needed to the DAG to marshall
#    features from other streams in the network (most often the input stream).

dag = DAG(name='A072391', n_jobs=1)
dag.add_input(name='head')
dag.add_step(name='screen', operation=Stream.split_by_estimator, streams=['head'],
             kwargs={'estimator': pipe, 'name_1': 'lump', 'name_2': 'fines',
                     'mass_recovery_column': 'lump_pct', 'mass_recovery_max': 100})
dag.add_step(name='screen_2', operation=Stream.split_by_estimator, streams=['fines'],
             kwargs={'estimator': pipe, 'name_1': 'lump_2', 'name_2': 'fines_2',
                     'mass_recovery_column': 'lump_pct', 'mass_recovery_max': 100,
                     'allow_prefix_mismatch': True})
dag.add_output(name='lump', stream='lump_2')
dag.add_output(name='fines', stream='fines_2')
dag.add_output(name='stockpile', stream='lump')
dag.run(input_streams={'head': head}, progress_bar=True)

fs: Flowsheet = Flowsheet.from_dag(dag)

fig = fs.plot_network()
fig

# %%

fig = fs.table_plot(plot_type='sankey', sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=52,
                    sankey_vmax=70)
plotly.io.show(fig)
