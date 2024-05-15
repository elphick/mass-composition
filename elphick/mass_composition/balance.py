"""
For managing the mass balance across a mc-network.

DESIGN NOTE: Use a balance config file for the standard deviation definition rather than
appending more properties to the Flowsheet object (at least for now), since mass balancing
may be more for advanced users - keeps those properties in the scope of this class.

DESIGN NOTE: Early attempts to optimise error/cost in absolute space (mass/grade differences) while
constraining the mass balance (in metal units failed), due to needing a matric of constraints in the same space
as the input.  Instead, we'll apply a cost to the metal balance by each component.

Initially we will develop for dry balancing only.

"""
from functools import partial
from typing import Optional, List, Callable, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

from elphick.mass_composition.flowsheet import Flowsheet
from elphick.mass_composition.mc_node import NodeType


class MCBalance:
    def __init__(self, fs: Flowsheet):
        self.fs: Flowsheet = fs
        self.sd: pd.DataFrame = self.create_balance_config(best_measurements='input')

    def _create_cost_functions(self) -> Dict[str, Callable]:
        """Cost Functions to be minimised

        We penalise the following:
        1) differences between mass and absolute grades for each stream versus measured.
        2) differences across each node for component masses (in-out)

        If each record is minimised with the appropriate cost function individually we expect
        to balance each record as well as the aggregate.

        Returns:
            float which is the cost to be minimised
        """

        xm: pd.DataFrame = self.fs.to_dataframe().drop(columns=['mass_wet'])

        # xi: pd.DataFrame = df_x[[col for col in df_x.columns if col != 'mass_wet']]
        # sd: pd.DataFrame = self.sd

        stream_map: Dict = {n: i for i, n in enumerate(self.fs.get_stream_names())}
        nodes = [n for n in self.fs.graph.nodes.data() if n[1]['mc'].node_type == NodeType.BALANCE]
        node_ins_outs: List = []
        for n in nodes:
            inputs, outputs = self.fs.get_node_input_outputs(n[0])
            node_ins_outs.append(([stream_map[i.name] for i in inputs], [stream_map[o.name] for o in outputs]))

        def cost_fn(x: np.ndarray, xm: np.ndarray, sd: np.ndarray, node_relationships: List) -> float:
            """The numpy compliant cost function

            The columns of xm and sd are mass_dry, h2o, followed by any chemical analytes

            Args:
                x: The x values as a 1d array on which to calculate cost.
                xm: The 2D array (mxn) of measured x values: m=streams, n=components.
                sd: The 2D array (mxn) of sd values: m=streams, n=components.
                node_relationships: A list of input and output tuples of stream indexes (define node ins/outs)

            Returns:

            """

            cost_mass_grades = np.nan_to_num(((xm.ravel() - x) / (x * sd.ravel())) ** 2)

            # metal balance - convert to metal mass
            x_2d = x.reshape(xm.shape)
            # convert to mass units - first column is dry mass
            # ignore moisture (wet basis) for now...
            x_mass = np.hstack([x_2d[:, 0].reshape(-1, 1), x_2d[:, 1:] * x_2d[:, 0].reshape(-1, 1) / 100.0])
            cost_component_balance: List = []
            for ins, outs in node_relationships:
                mass_in_sum = x_mass[ins, :].sum(axis=0)
                x_mass_node = np.nan_to_num((mass_in_sum - x_mass[outs, :].sum(axis=0)) ** 2)
                cost_component_balance.append(x_mass_node.sum())

            costs = cost_mass_grades.sum() + sum(cost_component_balance)
            return costs

        # create one cost function per record
        d_fns: Dict = {}
        df_network: pd.DataFrame = self.fs.to_dataframe()
        cols = [col for col in df_network.columns if col != 'mass_wet']
        for i in self.fs.get_input_streams()[0].data.to_dataframe().index:
            df_x: pd.DataFrame = df_network.loc[i, :][cols]
            d_fns[i] = partial(cost_fn, xm=df_x.values, sd=self.sd.values, node_relationships=node_ins_outs)

        return d_fns

    def _create_constraints(self) -> List[LinearConstraint]:
        """Prepare the optimisation constraints

        The constraints ensure that the optimised component values stay within range post optimisation.

        The values are extracted from the MassComposition objects on the network.

        Returns:

        """
        xm: pd.DataFrame = self.fs.to_dataframe().drop(columns=['mass_wet'])

        streams = self.fs.streams_to_dict()
        linear_constraints: List[LinearConstraint] = []
        for i, (s, mc) in enumerate(streams.items()):
            cons = mc.constraints
            for col in xm.columns:
                cons_matrix = np.zeros([len(streams.keys()), xm.shape[1]])
                cons_matrix[i, :] = (xm.columns == col) * 1
                linear_constraints.append(LinearConstraint([cons_matrix.ravel()], [cons[col][0]], [cons[col][1]]))
        return linear_constraints

    def _get_constraints(self, x) -> Callable:
        """Prepare the constraint function

        NOTE: Parked - not used since now the metal balance is managed via the cost function
         rather than a constraint.

        When optimising np.ndarrays are used - we don't have the luxury of labels.
        Strategy:
        1) create an index of the streams in the order they fall
        2) iterate the ins/outs tuples and store the names
        3) prepare the function from the reshaped x array (using num_components) and the stream index number.

        Returns:

        """
        pass
        # stream_map: Dict = {n: i for i, n in enumerate(self.fs.get_edge_names())}
        # nodes = [n for n in self.fs.graph.nodes.data() if n[1]['mc'].node_type == NodeType.BALANCE]
        # node_ins_outs: List = []
        # for n in nodes:
        #     inputs, outputs = self.fs.get_node_input_outputs(n[0])
        #     node_ins_outs.append(([stream_map[i.name] for i in inputs], [stream_map[o.name] for o in outputs]))
        # num_components: int = len(self.fs.get_input_edges()[0].variables.chemistry.get_var_names()) + 2
        #
        # def constraint_fn(x: np.ndarray, node_relationships: List, num_comp: int) -> Callable:
        #     x_2d = x.reshape(len(x) // num_comp, num_comp)
        #     # convert to mass units - first column is dry mass
        #     # ignore moisture (wet basis) for now...
        #     x_mass = np.hstack([x_2d[:, 0].reshape(-1, 1), x_2d[:, 1:] * x_2d[:, 0].reshape(-1, 1) / 100.0])
        #     pass
        #
        # constraint_fn(x=x, node_relationships=node_ins_outs, num_comp=num_components)
        #
        # return partial(constraint_fn(node_relationships=node_ins_outs, num_comp=num_components))

    def create_balance_config(self,
                              best_measurements: Optional[str] = None,
                              best_locked: bool = False) -> pd.DataFrame:
        """Create a balance config file

        Args:
            best_measurements: The best measurements 'input'|'output', that will be given tighter SDs.
            best_locked: If True, the best measurements will be locked with very small SDs, otherwise SDs will be
            tighter than the nominal, but still able to flex when balanced (though less than other streams).

        Returns:
            A DataFrame containing one row per stream, and one column per component.  The values will be
            the SDs used to normalise the residuals in the cost function that is minimised during balancing.
        """

        df_sd: pd.DataFrame = self.fs.report().drop(columns=['mass_wet'])
        df_sd.loc[:, :] = 1.0
        if best_measurements:
            tight_sd: float = 0.001 if best_locked else 0.1
            if best_measurements == 'input':
                for strm in [e.name for e in self.fs.get_input_streams()]:
                    df_sd.loc[strm, :] = tight_sd
            elif best_measurements == 'output':
                for strm in [e.name for e in self.fs.get_output_streams()]:
                    df_sd.loc[strm, :] = tight_sd
            else:
                raise KeyError("best_measurements argument must be 'input'|'output'")
        return df_sd

    def optimise(self) -> pd.DataFrame:
        """Optimise to deliver balanced mass and component masses

        We'll prepare cost functions for each record in the dataset, and will iterate through them all,
        optimising in turn.  Later we can parallelize.

        Returns:

        """

        d_cost_fns: Dict[str, Callable] = self._create_cost_functions()
        constraints: List[LinearConstraint] = self._create_constraints()
        df_measured: pd.DataFrame = self.fs.to_dataframe().fillna(0)
        cols: List[str] = [col for col in df_measured.columns if col != 'mass_wet']

        chunks: List = []
        for k, fn in d_cost_fns.items():
            df_x0: pd.DataFrame = df_measured.loc[k, cols]
            res = minimize(fn, df_x0.values.ravel(), method='powell',  #method='nelder-mead',
                           options={'xatol': 1e-5, 'disp': True, 'maxfev': 100000},
                           constraints=constraints)
            df_res: pd.DataFrame = pd.DataFrame(res.x.reshape(df_x0.shape),
                                                index=df_x0.index, columns=df_x0.columns).assign(
                **{df_measured.index.names[0]: k})
            print(df_x0 - df_res)
            chunks.append(df_res)
        df_res: pd.DataFrame = pd.concat(chunks).reset_index().set_index(df_measured.index.names)
        df_res = df_res.loc[df_measured.index, :]  # sort to match the input
        return df_res
