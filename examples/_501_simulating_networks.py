"""
Simulating a Network in Parallel
================================

While the ultimate objective is to process multiple fractionated samples together (with sample as a dim),
this pattern may be useful in the mean-time. It demonstrates how to process multiple samples in parallel,
with a progressbar to provide feedback.

The function my_simulator and the class TqdmParallel is defined in simulating_networks_tools.py, and are
 imported here to demonstrate.

"""
import pandas as pd
import plotly
from joblib import delayed

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import sample_data
from elphick.mass_composition.network import MCNetwork
from elphick.mass_composition.utils.parallel import TqdmParallel
from examples._simulating_network_functions import my_simulator

# %%
# Execute multiple simulations
# ----------------------------

df_data: pd.DataFrame = sample_data()
obj_mc: MassComposition = MassComposition(df_data, name='sample')
d_inputs: dict[int, MassComposition] = {1: obj_mc, 2: obj_mc.add(obj_mc), 3: obj_mc.add(obj_mc).add(obj_mc)}

results: list[tuple[int, MCNetwork]] = TqdmParallel(n_jobs=3, prefer="processes", total=len(d_inputs))(
    delayed(my_simulator)(item) for item in d_inputs.items()
)

d_results = {sid: mcn for sid, mcn in results}

# %%
# Print the results
print(d_results)

# %%
# View the network for a sample
# -----------------------------

fig = d_results[1].table_plot()
plotly.io.show(fig)
