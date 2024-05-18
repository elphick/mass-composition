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

import numpy as np
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

df: pd.DataFrame = iron_ore_met_sample_data()

base_components = ['fe', 'p', 'sio2', 'al2o3', 'loi']
cols_x = ['dry_weight_lump_kg'] + [f'head_{comp}' for comp in base_components]
cols_y = ['lump_pct'] + [f'lump_{comp}' for comp in base_components]

# %%
df = df.loc[:, ['sample_number'] + cols_x + cols_y].query('lump_pct>0').replace('-', np.nan).astype(float).dropna(
    how='any')
df = df.rename(columns={'dry_weight_lump_kg': 'head_mass_dry'}).set_index('sample_number')
df.index = df.index.astype(int)
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

head: MassComposition = MassComposition(data=X[[col for col in X.columns if 'head' in col]],
                                        mass_dry_var='head_mass_dry')
lump, fines = head.split_by_estimator(estimator=pipe, name_2='fines',
                                      mass_recovery_column='lump_pct', mass_recovery_max=100)

lump
# %%
fines

# %%
# Define the DAG
# --------------
#
# The DAG is defined by adding nodes to the graph.  Each node is an input, output or Stream operation
# (e.g. add, split, etc.).  The nodes are connected by the streams they operate on.

dag = DAG(name='A072391', n_jobs=2)
dag.add_input(name='head')
dag.add_step(name='screen', operation=Stream.split_by_estimator, streams=['head'],
             kwargs={'estimator': pipe, 'name_1': 'lump', 'name_2': 'fines',
                     'mass_recovery_column': 'lump_pct', 'mass_recovery_max': 100})
dag.add_output(name='lump', stream='lump')
dag.add_output(name='fines', stream='fines')

# %%
# Run the DAG
# -----------
#
# The dag is run by providing a Stream object for the input.

dag.run({'head': head}, progress_bar=True)

# %%
# Create a Flowsheet object from the dag, enabling all the usual network plotting and analysis methods.

fs: Flowsheet = Flowsheet.from_dag(dag)

fig = fs.plot_network()
fig

# %%

fig = fs.table_plot(plot_type='sankey', sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=52,
                    sankey_vmax=70)
plotly.io.show(fig)
