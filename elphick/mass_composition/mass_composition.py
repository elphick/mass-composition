import logging
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterable, Callable, Set, Literal, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin

from elphick.mass_composition.config import read_yaml
from elphick.mass_composition.mc_status import Status
from elphick.mass_composition.plot import parallel_plot, comparison_plot
from elphick.mass_composition.utils import solve_mass_moisture
from elphick.mass_composition.utils.amenability import amenability_index
from elphick.mass_composition.utils.interp import mass_preserving_interp
from elphick.mass_composition.utils.pd_utils import weight_average, calculate_recovery, calculate_partition
from elphick.mass_composition.utils.sampling import random_int
from elphick.mass_composition.utils.sklearn import extract_feature_names, PandasPipeline

from elphick.mass_composition.variables import Variables, VariableGroups


class MassComposition:

    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None,
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None,
                 mass_units: Optional[str] = None,
                 composition_units: Optional[str] = None,
                 constraints: Optional[Dict[str, List]] = None,
                 config_file: Optional[Path] = None):
        """

        Args:
            data:
            name:
            mass_wet_var:
            mass_dry_var:
            moisture_var:
            chem_vars:
            mass_units:
            constraints:
            config_file:
        """

        self._logger = logging.getLogger(name=self.__class__.__name__)

        if config_file is None:
            config_file = Path(__file__).parent / './config/mc_config.yml'
        self.config = read_yaml(config_file)

        # _nodes can preserve relationships from math operations, and can be used to build a network.
        self._nodes: List[Union[str, int]] = [random_int(), random_int()]

        self._name: str = name
        self._mass_units = self.config['units']['mass'] if not mass_units else None
        self._composition_units = self.config['units']['composition_rel'] if not composition_units else None

        self._specified_columns: Dict = {'mass_wet_var': mass_wet_var,
                                         'mass_dry_var': mass_dry_var,
                                         'moisture_var': moisture_var,
                                         'chem_vars': chem_vars}

        self._data: Optional[xr.Dataset] = None
        self.variables: Optional[Variables] = None
        self.constraints: Optional[Dict[str, List]] = None
        self.status: Optional[Status] = None

        if data is not None:
            data = deepcopy(data)  # preserve the incoming data variable.
            self.set_data(data, constraints=constraints)

    @staticmethod
    def _strip_common_prefix(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        # Extract prefixes
        common_prefix = MassComposition.get_common_prefix(df.columns.to_list())

        res = df
        # Create a copy of the dataframe and strip the most common prefix from column names
        if common_prefix:
            res = df.copy()
            res.columns = [col.replace(common_prefix + '_', '') if col.startswith(common_prefix) else col for col in
                           df.columns]

        return res, common_prefix

    @staticmethod
    def get_common_prefix(columns: List[str]) -> str:
        prefixes = [col.split('_')[0] for col in columns]
        # Count the frequency of each prefix
        prefix_counter = Counter(prefixes)
        # Check if prefix_counter is not empty
        if prefix_counter:
            # Find the most common prefix
            common_prefix, freq = prefix_counter.most_common(1)[0]
            # Only return the prefix if its frequency is 3 or more
            if freq >= 3:
                return common_prefix
        return ""

    def set_data(self, data: Union[pd.DataFrame, xr.Dataset],
                 constraints: Optional[Dict[str, List]] = None):
        if isinstance(data, xr.Dataset):
            # we assume it is a compliant mc-xarray
            self._data = data
            self.variables = Variables(config=self.config['vars'],
                                       supplied=[str(v) for v in data.variables if v not in data.dims],
                                       specified_map=self._specified_columns)
        elif isinstance(data, pd.DataFrame):

            if sum(data.index.duplicated()) > 0:
                raise KeyError('The data has duplicate indexes.')
            if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 3:
                self._logger.warning('The data has more than 2 levels in the index, which can consume excessive '
                                     'memory for large datasets.  Is this is what you intend?  Depending on your'
                                     'requirements you may be able to process ths dataset with a single index.')

            # seek a prefix to self assign the name
            data, common_prefix = self._strip_common_prefix(data)
            if common_prefix:
                self._specified_columns = {k: v.replace(f"{common_prefix}_", '') for k, v in
                                           self._specified_columns.items()
                                           if v is not None}

            self.variables = Variables(config=self.config['vars'],
                                       supplied=list(data.columns),
                                       specified_map=self._specified_columns)

            # if interval pairs are passed as indexes then create the proper interval index
            data = self._create_interval_indexes(data=data)

            # rename the columns using the Variables class
            data.rename(columns=self.variables.vars.col_to_var(), inplace=True)

            # solve or validate the moisture balance
            data = self._solve_mass_moisture(data)

            xr_ds = self._dataframe_to_mc_dataset(data)

            self._data = xr_ds

            if not self._name:
                self.rename('unnamed') if not common_prefix else self.rename(common_prefix.strip('_'))
            else:
                self.rename(self._name)

        # explicitly define the constraints
        self.constraints: Dict = self.get_constraint_bounds(constraints=constraints)
        self.status = Status(self._check_constraints())

    def rename(self, new_name: str) -> 'MassComposition':
        self.name = new_name
        return self

    def get_constraint_bounds(self, constraints: Optional[Dict[str, List]]) -> Dict[str, List]:
        d_constraints: Dict = {}

        # populate from the defaults
        for v in self.variables.mass_moisture.get_var_names():
            if 'mass' in v:
                d_constraints[v] = self.config['constraints']['mass']
            else:
                d_constraints[v] = self.config['constraints']['composition']
        for col in self.variables.chemistry.get_var_names():
            d_constraints[col] = self.config['constraints']['composition']

        # modify the default dict based on any user passed constraints
        if constraints:
            for k, v in constraints.items():
                d_constraints[k] = v

        return d_constraints

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, name: Optional[str] = 'unnamed'):
        obj = cls()
        obj._data = ds
        obj.name = name
        return obj

    @property
    def name(self) -> str:
        return self._data.mc.name

    @name.setter
    def name(self, value):
        self._data.mc.rename(value)

    @property
    def data(self) -> xr.Dataset:

        moisture: xr.DataArray = xr.DataArray((self._data['mass_wet'] - self._data['mass_dry']) /
                                              self._data['mass_wet'] * 100, name='H2O',
                                              attrs={'units': '%',
                                                     'standard_name': 'H2O',
                                                     'mc_type': 'moisture',
                                                     'mc_col_orig': 'H2O'}
                                              )

        data: xr.Dataset = xr.merge(
            [self._data[self._data.attrs['mc_vars_mass']],
             moisture,
             self._data[self._data.attrs['mc_vars_chem']],
             self._data[self._data.attrs['mc_vars_attrs']]])
        return data

    def update_data(self, values: Union[pd.DataFrame, xr.Dataset, xr.DataArray]):
        if isinstance(values, xr.Dataset) or isinstance(values, xr.DataArray):
            values = values.to_dataframe()
        for v in values.columns:
            self._data[v].values = values[v].values
        self.status = Status(self._check_constraints())

    def set_parent_node(self, parent: 'MassComposition') -> 'MassComposition':
        self._nodes = [parent._nodes[1], self._nodes[1]]
        return self

    def set_child_node(self, child: 'MassComposition') -> 'MassComposition':
        self._nodes = [self._nodes[0], child._nodes[0]]
        return self

    def set_stream_nodes(self, nodes: Tuple[int, int]) -> 'MassComposition':
        self._nodes = nodes
        return self

    def to_xarray(self) -> xr.Dataset:
        """Returns the mc compliant xr.Dataset

        Returns:

        """
        return self._data

    def aggregate(self, group_var: Optional[str] = None,
                  group_bins: Optional[Union[int, Iterable]] = None,
                  as_dataframe: bool = True,
                  original_column_names: bool = False) -> Union[pd.DataFrame, xr.Dataset]:
        """Calculate the weight average.

        Args:
            group_var: Optional grouping variable
            group_bins: Optional bins to apply to the group_var
            as_dataframe: If True return a pd.DataFrame
            original_column_names: If True, and as_dataframe is True, will return with the original column names.

        Returns:

        """

        res: xr.Dataset = self._data.mc.aggregate(group_var=group_var,
                                                  group_bins=group_bins,
                                                  as_dataframe=as_dataframe,
                                                  original_column_names=original_column_names)

        return res

    def query(self, queries) -> 'MassComposition':
        res: MassComposition = deepcopy(self)
        res._data = res._data.query(queries=queries)
        return res

    def constrain(self,
                  clip_mass: Optional[Union[Tuple, Dict]] = None,
                  clip_composition: Optional[Union[Tuple, Dict]] = None,
                  relative_mass: Optional[Union[Tuple, Dict]] = None,
                  relative_composition: Optional[Union[Tuple, Dict]] = None,
                  other: Optional['MassComposition'] = None) -> 'MassComposition':

        """Constrain the mass-composition

        It is possible that a MassComposition object is created from a source that has improbable results.
        In this case this method can help improve the integrity of the mass-composition.

        Args:
            clip_mass: Limit the minimum and maximum values of the mass between a minimum and maximum absolute value.
            clip_composition: Limit the minimum and maximum values of the composition between a minimum and
                maximum absolute value.
            relative_mass: Constrain the mass recovery of the object to the other object
            relative_composition: Constrain the component recovery of the object to the other object
            other: The other object used for recovery calculation.  Must be provided if relative_mass or
            relative_composition are provided.



        Returns:
            Returns the new object constrained per the provided arguments.
        """

        xr_ds: xr.Dataset = self.data.copy()

        if clip_mass:
            if isinstance(clip_mass, Dict):
                for k, v in clip_mass.items():
                    xr_ds = self._clip(xr_ds=xr_ds, variables=[k], limits=v)
            else:
                xr_ds = self._clip(xr_ds=xr_ds, variables=xr_ds.mc.mc_vars_mass, limits=clip_mass)

        if clip_composition:
            if isinstance(clip_composition, Dict):
                for k, v in clip_composition.items():
                    xr_ds = self._clip(xr_ds=xr_ds, variables=[k], limits=v)
            else:
                xr_ds = self._clip(xr_ds=xr_ds, variables=xr_ds.mc.mc_vars_chem, limits=clip_composition)

        if relative_mass or relative_composition:
            if not object:
                raise ValueError("The other other argument must be provided to apply relative constraints.")

        if relative_mass:
            xr_relative: xr.Dataset = self.data[xr_ds.mc.mc_vars_mass] / other.data[xr_ds.mc.mc_vars_mass]
            if isinstance(relative_mass, Dict):
                for k, v in relative_mass.items():
                    xr_relative = self._clip(xr_ds=xr_relative, variables=[k], limits=v)
            else:
                xr_relative = self._clip(xr_ds=xr_relative, variables=xr_ds.mc.mc_vars_mass, limits=relative_mass)

            # convert back to relative composition (mass/grades)
            xr_ds = other.data[xr_ds.mc.mc_vars_mass] * xr_relative
            xr_ds = xr.merge([xr_ds, self.data[self.data.mc.mc_vars_chem], self.data[self.data.mc.mc_vars_attrs]])
            xr_ds = self._copy_all_attrs(xr_ds, self.data)

        if relative_composition:
            xr_relative: xr.Dataset = self.compare(other=other, comparisons='recovery', explicit_names=False,
                                                   as_dataframe=False)
            if isinstance(relative_composition, Dict):
                for k, v in relative_composition.items():
                    xr_relative = self._clip(xr_ds=xr_relative, variables=[k], limits=v)
            else:
                xr_relative = self._clip(xr_ds=xr_relative, variables=self.data.mc.mc_vars_chem,
                                         limits=relative_composition)

            # convert back to relative composition (mass/grades)
            xr_ds = other.data.mc.mul(xr_relative)
            xr_ds = xr.merge([xr_ds, self.data[self.data.mc.mc_vars_attrs]])
            xr_ds = self._copy_all_attrs(xr_ds, self.data)

        res: MassComposition = MassComposition().from_xarray(xr_ds, name=self.name)

        return res

    def compare(self, other: 'MassComposition', comparisons: Union[str, List[str]] = 'recovery',
                explicit_names: bool = True, as_dataframe: bool = True) -> Union[pd.DataFrame, xr.Dataset]:

        comparisons = [comparisons] if isinstance(comparisons, str) else comparisons
        valid_comparisons: Set = {'recovery', 'difference', 'divide', 'all'}

        def set_explicit_names(xrds, comparison) -> xr.Dataset:
            xrds = xrds.rename_vars(
                {col: f"{self.name}_{col}_{self.config['comparisons'][comparison]}_{other.name}" for col in
                 xrds.data_vars})
            return xrds

        cols = [col for col in self.data.data_vars if col not in self.data.mc.mc_vars_attrs]

        chunks: List[xr.Dataset] = []
        if 'recovery' in comparisons or comparisons == ['all']:
            ds: xr.Dataset = self.data.mc.composition_to_mass()[cols] / other.data.mc.composition_to_mass()[cols]
            ds = set_explicit_names(ds, comparison='recovery') if explicit_names else ds
            chunks.append(ds)
        if 'difference' in comparisons or comparisons == ['all']:
            ds: xr.Dataset = self.data[cols] - other.data[cols]
            ds = set_explicit_names(ds, comparison='difference') if explicit_names else ds
            chunks.append(ds)
        if 'divide' in comparisons or comparisons == ['all']:
            ds: xr.Dataset = self.data[cols] / other.data[cols]
            ds = set_explicit_names(ds, comparison='divide') if explicit_names else ds
            chunks.append(ds)

        if not chunks:
            raise ValueError(f"The comparison argument is not valid: {valid_comparisons}")

        res: xr.Dataset = xr.merge(chunks)
        res: pd.DataFrame = res.to_dataframe() if as_dataframe else res

        return res

    def binned_mass_composition(self, cutoff_var: str,
                                bin_width: float,
                                cumulative: bool = True,
                                direction: str = 'descending',
                                as_dataframe: bool = True,
                                ) -> Union[xr.Dataset, pd.DataFrame]:
        """A.K.A "The Grade-Tonnage" curve.

        Mass and grade by bins for a cut-off variable.

        Args:
            cutoff_var: The variable that defines the bins
            bin_width: The width of the bin
            cumulative: If True, the results are cumulative weight averaged.
            direction: 'ascending'|'descending', if cumulative is True, the direction of accumulation
            as_dataframe: If True return a pd.DataFrame

        Returns:

        """

        if cutoff_var not in list(self._data.variables):
            raise KeyError(f'{cutoff_var} is not found in the data')

        bins = np.arange(np.floor(min(self._data[cutoff_var].values)),
                         np.ceil(max(self._data[cutoff_var].values)) + bin_width,
                         bin_width)

        res: xr.Dataset = self.aggregate(group_var=cutoff_var, group_bins=bins, as_dataframe=False)

        if cumulative:
            res = res.mc.data().mc.cumulate(direction=direction)

        if as_dataframe:
            res = res.mc.data().to_dataframe()
        else:
            res = res.mc.data()

        return res

    def ideal_incremental_separation(self, discard_from: Literal["lowest", "highest"] = "lowest") -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the mass-composition and recovery of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_composition, ideal_incremental_recovery.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.

        Returns:
            A pandas DataFrame
        """
        self._check_one_dim_interval()

        sample: pd.DataFrame = self.data.to_dataframe()

        is_decreasing: bool = sample.index.is_monotonic_decreasing
        if discard_from == "lowest":
            sample.sort_index(ascending=True, inplace=True)
            new_index: pd.Index = pd.Index(sample.index.left)
        else:
            sample.sort_index(ascending=False, inplace=True)
            new_index: pd.Index = pd.Index(sample.index.right)
        new_index.name = f"{sample.index.name}_cut-point"

        aggregated_chunks: List = []
        recovery_chunks: List = []
        head: pd.DataFrame = sample.pipe(weight_average)

        for i, indx in enumerate(sample.index):
            tmp_composition: pd.DataFrame = sample.iloc[i:, :].pipe(weight_average)
            aggregated_chunks.append(tmp_composition)
            recovery_chunks.append(tmp_composition.pipe(calculate_recovery, df_ref=head))

        res_composition: pd.DataFrame = pd.concat(aggregated_chunks).assign(attribute="composition").set_index(
            new_index)
        res_recovery: pd.DataFrame = pd.concat(recovery_chunks).assign(attribute="recovery").set_index(
            new_index)

        if is_decreasing:
            res_composition.sort_index(ascending=False, inplace=True)
            res_recovery.sort_index(ascending=False, inplace=True)

        res: pd.DataFrame = pd.concat([res_composition, res_recovery]).reset_index().set_index(
            [new_index.name, 'attribute'])

        return res

    def ideal_incremental_composition(self, discard_from: Literal["lowest", "highest"] = "lowest") -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the mass-composition of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_separation, ideal_incremental_recovery.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.

        Returns:
            A pandas DataFrame
        """
        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from).query(
            'attribute=="composition"').droplevel('attribute')
        return df

    def ideal_incremental_recovery(self, discard_from: Literal["lowest", "highest"] = "lowest",
                                   apply_closure: bool = True) -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the recovery of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_separation, ideal_incremental_composition.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            apply_closure: If True, Add the missing record (zero recovery) that closes the recovery envelope.

        Returns:
            A pandas DataFrame
        """

        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from).query(
            'attribute=="recovery"').droplevel('attribute').rename(columns={'mass_dry': 'mass'}).drop(
            columns=["mass_wet", 'H2O'])
        if apply_closure:
            # add zero recovery record to close the envelope.
            indx = np.inf if df.index.min() == 0.0 else 0.0
            indx_name: str = df.index.name
            df = pd.concat([df, pd.Series(0, index=df.columns, name=indx).to_frame().T]).sort_index(ascending=True)
            df.index.name = indx_name
        return df

    def split(self,
              fraction: float,
              name_1: Optional[str] = None,
              name_2: Optional[str] = None) -> Tuple['MassComposition', 'MassComposition']:
        """Split the object by mass

        A simple mass split maintaining the same composition

        See also: split_by_partition, split_by_function, split_by_estimator

        Args:
            fraction: A constant in the range [0.0, 1.0]
            name_1: The name of the reference stream created by the split
            name_2: The name of the complement stream created by the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """

        xr_ds_1, xr_ds_2 = self._data.mc.split(fraction=fraction)

        out: MassComposition = MassComposition(name=xr_ds_1.mc.name, constraints=self.constraints)
        out.set_data(data=xr_ds_1, constraints=self.constraints)
        comp: MassComposition = MassComposition(name=xr_ds_2.mc.name, constraints=self.constraints)
        comp.set_data(data=xr_ds_2, constraints=self.constraints)

        self._post_process_split(out, comp, name_1, name_2)

        return out, comp

    def split_by_partition(self,
                           partition_definition: Callable,
                           name_1: Optional[str] = None,
                           name_2: Optional[str] = None) -> Tuple['MassComposition', 'MassComposition']:
        """Partition the object along a given dimension.

        This method applies the defined separation resulting in two new objects.

        See also: split, split_by_function, split_by_estimator

        Args:
            partition_definition: A partition function that defines the efficiency of separation along a dimension
            name_1: The name of the reference stream created by the split
            name_2: The name of the complement stream created by the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        out = deepcopy(self)
        comp = deepcopy(self)

        xr_ds_1, xr_ds_2 = self._data.mc.split_by_partition(partition_definition=partition_definition)

        out._data = xr_ds_1
        comp._data = xr_ds_2

        self._post_process_split(out, comp, name_1, name_2)

        return out, comp

    def split_by_function(self,
                          split_function: Callable,
                          name_1: Optional[str] = None,
                          name_2: Optional[str] = None) -> Tuple['MassComposition', 'MassComposition']:
        """Split an object using a function.

        This method applies the function to self, resulting in two new objects. The object returned with name_1
        is the result of the function.  The object returned with name_2 is the complement.

        See also: split, split_by_estimator, split_by_partition

        Args:
            split_function: Any function that transforms the dataframe from a MassComposition object into a new
             dataframe with values representing a new (output) stream.  The returned dataframe structure must be
             identical to the input dataframe.
            name_1: The name of the stream created by the function
            name_2: The name of the complement stream created by the split, which is calculated automatically.

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        out_data: pd.DataFrame = split_function(self.data.to_dataframe())
        out: MassComposition = MassComposition(name=name_1, constraints=self.constraints, data=out_data)
        comp: MassComposition = self.sub(other=out, name=name_2)

        self._post_process_split(out, comp, name_1, name_2)

        return out, comp

    def split_by_estimator(self,
                           estimator: PandasPipeline,
                           name_1: Optional[str] = None,
                           name_2: Optional[str] = None,
                           extra_features: Optional[pd.DataFrame] = None,
                           allow_prefix_mismatch: bool = False,
                           mass_recovery_column: Optional[str] = None,
                           mass_recovery_max: float = 1.0) -> Tuple['MassComposition', 'MassComposition']:
        """Split an object using a sklearn estimator.

        This method applies the function to self, resulting in two new objects. The object returned with name_1
        is the result of the estimator.predict() method.  The object returned with name_2 is the complement.

        See also: split, split_by_function, split_by_partition

        Args:
            estimator: Any sklearn estimator that transforms the dataframe from a MassComposition object into a new
             dataframe with values representing a new (output) stream using the predict method.  The returned
             dataframe structure must be identical to the input dataframe.
            name_1: The name of the stream created by the estimator.
            name_2: The name of the complement stream created by the split, which is calculated automatically.
            extra_features: Optional additional features to pass to the estimator as features.
            allow_prefix_mismatch: If True, allow feature names to be different and log an info message. If False,
             raise an error when feature names are different.
            mass_recovery_column: If provided, this indicates that the model has estimated mass recovery, not mass
             explicitly.  This will execute a transformation of the predicted `dry` mass recovery to dry mass.
            mass_recovery_max: The maximum mass recovery value, used to scale the mass recovery to mass.  Only
             applicable if mass_recovery_column is provided.  Should be either 1.0 or 100.0.

        Returns:
            tuple of two MassComposition objects, the first the output of the estimator, the other the complement
        """
        # Extract feature names from the estimator, and get the actual features
        feature_names: list[str] = list(extract_feature_names(estimator))
        features: pd.DataFrame = self._get_features(feature_names, allow_prefix_mismatch=allow_prefix_mismatch,
                                                    extra_features=extra_features)

        # Apply the estimator
        estimates: pd.DataFrame = estimator.predict(X=features)
        if isinstance(estimates, np.ndarray):
            raise NotImplementedError("The estimator must return a DataFrame")

        # Detect a possible prefix from the estimate columns
        features_prefix: str = self.get_common_prefix(features.columns.to_list())
        estimates_prefix: str = self.get_common_prefix(estimates.columns.to_list())

        # If there is a prefix, check that it matches name_1, subject to allow_prefix_mismatch
        if estimates_prefix and not allow_prefix_mismatch and name_1 and not name_1 == estimates_prefix:
            raise ValueError(f"Common prefix mismatch: {features_prefix} and name_1: {name_1}")

        # assign the output names, based on specified names, allow for prefix mismatch
        name_1 = name_1 if name_1 else estimates_prefix

        if mass_recovery_column:
            # Transform the mass recovery to mass by applying the mass recovery to the dry mass of the input stream
            if mass_recovery_max not in [1.0, 100.0]:
                raise ValueError(f"mass_recovery_max must be either 1.0 or 100.0, not {mass_recovery_max}")
            if mass_recovery_column not in estimates.columns:
                raise KeyError(f"mass_recovery_column: {mass_recovery_column} not found in the estimates.")

            dry_mass_var: str = self.data.mass_dry.name
            estimates[mass_recovery_column] = estimates[mass_recovery_column] * self.data[
                dry_mass_var].values / mass_recovery_max
            estimates.rename(columns={mass_recovery_column: dry_mass_var}, inplace=True)

        if estimates_prefix:
            col_name_map: dict[str, str] = {f: f.replace(estimates_prefix + '_', "") for f in estimates.columns}
            estimates.rename(columns=col_name_map, inplace=True)

        out: MassComposition = MassComposition(name=name_1, constraints=self.constraints, data=estimates)
        comp: MassComposition = self.sub(other=out, name=name_2)

        self._post_process_split(out, comp, name_1, name_2)

        return out, comp

    def _get_features(self, feature_names: List[str], allow_prefix_mismatch: bool,
                      extra_features: Optional[pd.DataFrame] = None, ) -> pd.DataFrame:
        """
        This method checks if the feature names required by an estimator are present in the data. If not, it tries to
        match the feature names by considering a common prefix. If a match is found, the columns in the data are renamed
        accordingly. If a match is not found and `allow_prefix_mismatch` is False, an error is raised. If
        `allow_prefix_mismatch` is True, the method proceeds with the mismatched feature names.

        If `extra_features` is provided, these features are added to the data.

        Args:
            feature_names (List[str]): A list of feature names required by the estimator.
            allow_prefix_mismatch (bool): If True, allows the feature names in the data and the estimator to be different.
            extra_features (Optional[pd.DataFrame]): Additional features to be added to the data.

        Returns:
            pd.DataFrame: The data with the correct feature names.

        Raises:
            ValueError: If `allow_prefix_mismatch` is False and the feature names in the data and the estimator do not match.
        """

        # Create a mapping of lower-case feature names to original feature names
        feature_name_map = {name.lower(): name for name in feature_names}

        df_features: pd.DataFrame = self.data.to_dataframe()
        if extra_features is not None:
            df_features = pd.concat([df_features, extra_features], axis=1)

        missing_features = set(f.lower() for f in feature_names) - set(c.lower() for c in df_features.columns)

        if missing_features:
            prefix: str = f"{self.name}_"
            common_prefix: str = self.get_common_prefix(feature_names)
            if common_prefix and common_prefix + '_' != prefix and allow_prefix_mismatch:
                prefix = common_prefix + '_'

            # create a map to support renaming the columns
            prefixed_feature_map: dict[str, str] = {f: feature_name_map.get(f"{prefix}{f.lower()}") for f in
                                                    df_features.columns if
                                                    feature_name_map.get(f"{prefix}{f.lower()}") is not None}
            df_features.rename(columns=prefixed_feature_map, inplace=True)
            missing_features = set(f.lower() for f in feature_names) - set(c.lower() for c in df_features.columns)

            if missing_features:
                raise ValueError(f"Missing features: {missing_features}, with mc.name: {self.name}, prefix: {prefix}"
                                 f" and allow_prefix_mismatch set to {allow_prefix_mismatch}.")

        # Return the dataframe with the selected features
        df_features: pd.DataFrame = df_features[feature_names]

        return df_features

    def calculate_partition(self, ref: 'MassComposition') -> pd.DataFrame:
        """Calculate the partition of the ref stream relative to self"""
        self._check_one_dim_interval()
        return calculate_partition(df_feed=self.data.to_dataframe(), df_ref=ref.data.to_dataframe(),
                                   col_mass_dry='mass_dry')

    # def resample(self, dim: str, num_intervals: int = 50, edge_precision: int = 8) -> 'MassComposition':
    #     res = deepcopy(self)
    #     res._data = self._data.mc.resample(dim=dim, num_intervals=num_intervals, edge_precision=edge_precision)
    #     return res

    def resample_1d(self, interval_edges: Union[Iterable, int],
                    precision: Optional[int] = None,
                    include_original_edges: bool = False) -> 'MassComposition':
        """Resample a 1D fractional dim/index

        Args:
            interval_edges: The values of the new grid (interval edges).  If an int, will up-sample by that factor, for
             example the value of 10 will automatically define edges that create 10 x the resolution (up-sampled).
            precision: Optional integer for the number of decimal places to round the grid values to.
            include_original_edges: If True include the original edges in the grid.

        Returns:
            A new object interpolated onto the new grid
        """

        # TODO: add support for supplementary variables

        df_upsampled: pd.DataFrame = mass_preserving_interp(self.data.to_dataframe(),
                                                            interval_edges=interval_edges, precision=precision,
                                                            include_original_edges=include_original_edges)

        obj: MassComposition = MassComposition(df_upsampled, name=self.name)
        obj._nodes = self._nodes
        obj.constraints = self.constraints
        return obj

    def add(self, other: 'MassComposition', name: Optional[str] = None) -> 'MassComposition':
        """Add two objects

        Adds other to self, with optional name of the returned object
        Args:
            other: object to add to self
            name: name of the returned object

        Returns:

        """
        res: MassComposition = self.__add__(other)
        if name is not None:
            res._data.mc.rename(name)
        return res

    def sub(self, other: 'MassComposition', name: Optional[str] = None) -> 'MassComposition':
        """Subtract two objects

        Subtracts other from self, with optional name of the returned object
        Args:
            other: object to subtract from self
            name: name of the returned object

        Returns:

        """
        res: MassComposition = self.__sub__(other)
        if name is not None:
            res._data.mc.rename(name)
        return res

    def div(self, other: 'MassComposition', name: Optional[str] = None) -> 'MassComposition':
        """Divide two objects

        Divides self by other, with optional name of the returned object
        Args:
            other: the denominator (or reference) object
            name: name of the returned object

        Returns:

        """
        res: MassComposition = self.__truediv__(other)
        if name is not None:
            res._data.mc.rename(name)
        return res

    def plot_bins(self,
                  variables: List[str],
                  cutoff_var: str,
                  bin_width: float,
                  cumulative: bool = True,
                  direction: str = 'descending',
                  ) -> go.Figure:
        """Plot "The Grade-Tonnage" curve.

        Mass and grade by bins for a cut-off variable.

        Args:
            variables: List of variables to include in the plot
            cutoff_var: The variable that defines the bins
            bin_width: The width of the bin
            cumulative: If True, the results are cumulative weight averaged.
            direction: 'ascending'|'descending', if cumulative is True, the direction of accumulation
        """

        bin_data: pd.DataFrame = self.binned_mass_composition(cutoff_var=cutoff_var,
                                                              bin_width=bin_width,
                                                              cumulative=cumulative,
                                                              direction=direction,
                                                              as_dataframe=True)

        id_var: str = bin_data.index.name

        df: pd.DataFrame = bin_data[variables].reset_index()
        # convert the interval to the left edge TODO: make flexible
        df[id_var] = df[id_var].apply(lambda x: x.left)
        var_cutoff: str = id_var.replace('_bins', '_cut-off')
        df.rename(columns={id_var: var_cutoff}, inplace=True)

        df = df.melt(id_vars=[var_cutoff], var_name='component')

        fig = px.line(df, x=var_cutoff, y='value', facet_row='component')
        fig.update_yaxes(matches=None)
        fig.update_layout(title=self.name)

        return fig

    def plot_intervals(self,
                       variables: List[str],
                       cumulative: bool = True,
                       direction: str = 'descending',
                       show_edges: bool = True,
                       min_x: Optional[float] = None) -> go.Figure:
        """Plot "The Grade-Tonnage" curve.

        Mass and grade by bins for a cut-off variable.

        Args:
            variables: List of variables to include in the plot
            cumulative: If True, the results are cumulative weight averaged.
            direction: 'ascending'|'descending', if cumulative is True, the direction of accumulation
            show_edges: If True, show the edges on the plot.  Applicable to cumulative plots only.
            min_x: Optional minimum value for the x-axis, useful to set reasonable visual range with a log
            scaled x-axis when plotting size data
        """

        res: xr.Dataset = self.data

        plot_kwargs: Dict = dict(line_shape='vh')
        if cumulative:
            res = res.mc.data().mc.cumulate(direction=direction)
            plot_kwargs = dict(line_shape='spline')

        interval_data: pd.DataFrame = res.mc.to_dataframe()

        df_intervals: pd.DataFrame = self._intervals_to_columns(interval_index=interval_data.index)
        df = pd.concat([df_intervals, interval_data], axis='columns')
        x_var: str = interval_data.index.name
        if not cumulative:
            # append on the largest fraction right edge for display purposes
            df_end: pd.DataFrame = df.loc[df.index.max(), list(df_intervals.columns) + variables].to_frame().T
            df_end[df_intervals.columns[0]] = df_end[df_intervals.columns[1]]
            df_end[df_intervals.columns[1]] = np.inf
            df = pd.concat([df_end, df], axis='index')
            df[interval_data.index.name] = df[df_intervals.columns[0]]
        else:
            if direction == 'ascending':
                x_var = df_intervals.columns[1]
            elif direction == 'descending':
                x_var = df_intervals.columns[0]

        if 'size' in x_var:
            if not min_x:
                min_x = interval_data.index.min().right / 2.0
            # set zero to the minimum x value (for display only) to enable the tooltips on that point.
            df.loc[df[x_var] == df[x_var].min(), x_var] = min_x
            hover_data = {'component': True,  # add other column, default formatting
                          x_var: ':.3f',  # add other column, customized formatting
                          'value': ':.2f'
                          }
            plot_kwargs = {**plot_kwargs,
                           **dict(log_x=True,
                                  range_x=[min_x, interval_data.index.max().right],
                                  hover_data=hover_data)}

        df = df[[x_var] + variables].melt(id_vars=[x_var], var_name='component')

        if cumulative and show_edges:
            plot_kwargs['markers'] = True

        fig = px.line(df, x=x_var, y='value', facet_row='component', **plot_kwargs)
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("component=", "")))
        fig.update_yaxes(matches=None)
        fig.update_layout(title=self.name)

        return fig

    def plot_grade_recovery(self, target_analyte,
                            discard_from: Literal["lowest", "highest"] = "lowest",
                            title: Optional[str] = None,
                            ) -> go.Figure:
        """The grade-recovery plot.

        The grade recovery curve is generated by assuming an ideal separation (for the chosen property, or dimension)
        at each fractional interval.  It defines the theoretical maximum performance, which can only be improved if
        liberation is improved by comminution.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        Args:
            target_analyte: The analyte of value.
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            title: Optional plot title

        Returns:
            A plotly.GraphObjects figure
        """
        title = title if title is not None else 'Ideal Grade - Recovery'

        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from)
        df_recovery: pd.DataFrame = df.loc[(slice(None), 'recovery'), [target_analyte, 'mass_dry']].droplevel(
            'attribute').rename(
            columns={'mass_dry': 'Yield', target_analyte: f"{target_analyte}_recovery"})
        df_composition: pd.DataFrame = df.loc[(slice(None), 'composition'), :].droplevel('attribute').drop(
            columns=['mass_wet', 'mass_dry', 'H2O'])

        df_plot: pd.DataFrame = pd.concat([df_recovery, df_composition], axis=1).reset_index()
        fig = px.line(df_plot, x=target_analyte,
                      y=f"{target_analyte}_recovery",
                      hover_data=df_plot.columns,
                      title=title)
        # fig.update_layout(xaxis_title=f"Grade of {target_analyte}", yaxis_title=f"Recovery of {target_analyte}",
        #                   title=title)

        return fig

    def plot_amenability(self, target_analyte: str,
                         discard_from: Literal["lowest", "highest"] = "lowest",
                         gangue_analytes: Optional[str] = None,
                         title: Optional[str] = None,
                         ) -> go.Figure:
        """The yield-recovery plot.

        The yield recovery curve provides an understanding of the amenability of a sample.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        Args:
            target_analyte: The analyte of value.
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            gangue_analytes: The analytes to be rejected
            title: Optional plot title

        Returns:
            A plotly.GraphObjects figure
        """
        title = title if title is not None else 'Amenability Plot'
        df: pd.DataFrame = self.ideal_incremental_recovery(discard_from=discard_from)
        amenability_indices: pd.Series = amenability_index(df, col_target=target_analyte, col_mass_recovery='mass')

        analytes = [col for col in df.columns if col != "mass"] if gangue_analytes is None else [
            target_analyte + gangue_analytes]

        mass_rec: pd.DataFrame = df["mass"]
        df = df[analytes]

        fig = go.Figure()
        for analyte in analytes:
            fig.add_trace(
                go.Scatter(x=mass_rec, y=df[analyte], mode="lines",
                           name=f"{analyte} ({round(amenability_indices[analyte], 2)})",
                           customdata=df.index.values,
                           hovertemplate='<b>Recovery: %{y:.3f}</b><br>Cut-point: %{customdata:.3f} '))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name='y=x',
                                 line=dict(shape='linear', color='gray', dash='dash'),
                                 ))
        fig.update_layout(xaxis_title='Yield (Mass Recovery)', yaxis_title='Recovery', title=title,
                          hovermode='x')
        return fig

    def plot_parallel(self, color: Optional[str] = None,
                      vars_include: Optional[List[str]] = None,
                      vars_exclude: Optional[List[str]] = None,
                      title: Optional[str] = None,
                      include_dims: Optional[Union[bool, List[str]]] = True,
                      plot_interval_edges: bool = False) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multidimensional data like mass-composition data

        Args:
            color: Optional color variable
            vars_include: Optional List of variables to include in the plot
            vars_exclude: Optional List of variables to exclude in the plot
            title: Optional plot title
            include_dims: Optional boolean or list of dimension to include in the plot.  True will show all dims.
            plot_interval_edges: If True, interval edges will be plotted instead of interval mid

        Returns:

        """
        df = self.data.mc.to_dataframe()

        if not title and hasattr(self, 'name'):
            title = self.name

        fig = parallel_plot(data=df, color=color, vars_include=vars_include, vars_exclude=vars_exclude, title=title,
                            include_dims=include_dims, plot_interval_edges=plot_interval_edges)
        return fig

    def plot_comparison(self, other: 'MassComposition',
                        color: Optional[str] = None,
                        vars_include: Optional[List[str]] = None,
                        vars_exclude: Optional[List[str]] = None,
                        facet_col_wrap: int = 3,
                        trendline: bool = False,
                        trendline_kwargs: Optional[Dict] = None,
                        title: Optional[str] = None) -> go.Figure:
        """Create an interactive parallel plot

        Useful to compare the difference in component values between two objects.

        Args:
            other: the object to compare with self.
            color: Optional color variable
            vars_include: Optional List of variables to include in the plot
            vars_exclude: Optional List of variables to exclude in the plot
            trendline: If True and trendlines
            trendline_kwargs: Allows customising the trendline: ref: https://plotly.com/python/linear-fits/
            title: Optional plot title
            facet_col_wrap: The number of subplot columns per row.

        Returns:

        """
        df_self: pd.DataFrame = self.data.to_dataframe()
        df_other: pd.DataFrame = other.data.to_dataframe()

        if vars_include is not None:
            missing_vars = set(vars_include).difference(set(df_self.columns))
            if len(missing_vars) > 0:
                raise KeyError(f'var_subset provided contains variable not found in the data: {missing_vars}')
            df_self = df_self[vars_include]
        if vars_exclude:
            df_self = df_self[[col for col in df_self.columns if col not in vars_exclude]]
        df_other = df_other[df_self.columns]
        # Supplementary variables are the same for each stream and so will be unstacked.
        supp_cols: List[str] = [col for col in df_self.columns if col in self.variables.supplementary.get_col_names()]
        if supp_cols:
            df_self.set_index(supp_cols, append=True, inplace=True)
            df_other.set_index(supp_cols, append=True, inplace=True)

        index_names = list(df_self.index.names)
        cols = list(df_self.columns).copy()

        df_self = df_self[cols].assign(name=self.name).reset_index().melt(id_vars=index_names + ['name'])
        df_other = df_other[cols].assign(name=other.name).reset_index().melt(id_vars=index_names + ['name'])

        df_plot: pd.DataFrame = pd.concat([df_self, df_other])
        df_plot = df_plot.set_index(index_names + ['name', 'variable'], drop=True).unstack(['name'])
        df_plot.columns = df_plot.columns.droplevel(0)
        df_plot.reset_index(level=list(np.arange(-1, -len(index_names) - 1, -1)), inplace=True)

        # set variables back to standard order
        variable_order: Dict = {col: i for i, col in enumerate(cols)}
        df_plot = df_plot.sort_values(by=['variable'], key=lambda x: x.map(variable_order))

        fig: go.Figure = comparison_plot(data=df_plot, x=self.name, y=other.name, facet_col_wrap=facet_col_wrap,
                                         color=color, trendline=trendline, trendline_kwargs=trendline_kwargs)
        fig.update_layout(title=title)
        return fig

    def plot_ternary(self, variables: List[str], color: Optional[str] = None,
                     title: Optional[str] = None) -> go.Figure:
        """Plot a ternary diagram

            variables: List of 3 components to plot
            color: Optional color variable
            title: Optional plot title

        """

        df = self.data.to_dataframe()
        vars_missing: List[str] = [v for v in variables if v not in df.columns]
        if vars_missing:
            raise KeyError(f'Variable/s not found in the dataset: {vars_missing}')

        cols: List[str] = variables
        if color is not None:
            cols.append(color)

        if color:
            fig = px.scatter_ternary(df[cols], a=variables[0], b=variables[1], c=variables[2], color=color)
        else:
            fig = px.scatter_ternary(df[cols], a=variables[0], b=variables[1], c=variables[2])

        if not title and hasattr(self, 'name'):
            title = self.name

        fig.update_layout(title=title)

        return fig

    def __str__(self) -> str:
        res: str = f'\n{self.name}\n'
        res += str(self.data)
        return res

    def __add__(self, other: 'MassComposition') -> 'MassComposition':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """
        xr_sum: xr.Dataset = self._data.mc.add(other._data)

        res: MassComposition = MassComposition(name=xr_sum.mc.name, constraints=self.constraints)
        res.set_data(data=xr_sum, constraints=self.constraints)

        other._nodes = [other._nodes[0], self._nodes[1]]
        res._nodes = [self._nodes[1], random_int()]

        return res

    def __sub__(self, other: 'MassComposition') -> 'MassComposition':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        xr_sub: xr.Dataset = self._data.mc.sub(other._data)

        res: MassComposition = MassComposition(name=xr_sub.mc.name, constraints=self.constraints)
        res.set_data(data=xr_sub, constraints=self.constraints)

        res._nodes = [self._nodes[1], random_int()]
        return res

    def __truediv__(self, other: 'MassComposition') -> 'MassComposition':
        """Divide self by the supplied object

        Perform the division with the mass-composition variables only and then append any attribute variables.
        Args:
            other: denominator object, self will be divided by this object

        Returns:

        """

        xr_div: xr.Dataset = self._data.mc.div(other._data)

        res: MassComposition = MassComposition(name=xr_div.mc.name, constraints=self.constraints)
        res.set_data(data=xr_div, constraints=self.constraints)

        return res

    def __eq__(self, other):
        if isinstance(other, MassComposition):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def _check_cols_in_data_cols(cols: List[str], cols_data: List[str]):
        for col in cols:
            if (col is not None) and (col not in cols_data):
                msg: str = f"{col} not in the data columns: {cols_data}"
                logging.getLogger(__name__).error(msg)
                raise IndexError(msg)

    @staticmethod
    def _copy_all_attrs(xr_to: xr.Dataset, xr_from: xr.Dataset) -> xr.Dataset:
        xr_to.attrs.update(xr_from.attrs)
        da: xr.DataArray
        for new_da, da in zip(xr_to.values(), xr_from.values()):
            new_da.attrs.update(da.attrs)
        return xr_to

    @staticmethod
    def _clip(xr_ds: xr.Dataset, variables: List[str], limits: Tuple) -> xr.Dataset:
        if len(variables) == 1:
            variables = variables[0]
        xr_ds[variables] = xr_ds[variables].where(xr_ds[variables] > limits[0], limits[0])
        xr_ds[variables] = xr_ds[variables].where(xr_ds[variables] < limits[1], limits[1])
        return xr_ds

    def _post_process_split(self, obj_1, obj_2, name_1, name_2):
        if name_1:
            obj_1._data.mc.rename(name_1)
        if name_2:
            obj_2._data.mc.rename(name_2)
        obj_1._nodes = [self._nodes[1], random_int()]
        obj_2._nodes = [self._nodes[1], random_int()]
        obj_1._name = name_1
        obj_2._name = name_2

        return obj_1, obj_2

    def _intervals_to_columns(self, interval_index: pd.IntervalIndex) -> pd.DataFrame:
        """Reconstruct columns from an interval index

        Uses the left and right names stored in the xr.Dataset attrs

        Args:
            interval_index: The IntervalIndex to convert to named columns of edges

        Returns:

        """
        base_name: str = str(interval_index.name)
        if base_name in self._data.attrs['mc_interval_edges'].keys():
            d_edge_names = self._data.attrs['mc_interval_edges'][base_name]
        else:
            d_edge_names = {'left': 'left', 'right': 'right'}
        df_intervals: pd.DataFrame = pd.DataFrame(index=interval_index).reset_index()
        df_intervals[f'{base_name}_{d_edge_names["left"]}'] = df_intervals[base_name].apply(lambda x: x.left)
        df_intervals[f'{base_name}_{d_edge_names["right"]}'] = df_intervals[base_name].apply(lambda x: x.right)
        df_intervals.set_index(base_name, inplace=True)
        return df_intervals

    def _create_interval_indexes(self, data: pd.DataFrame) -> pd.DataFrame:

        if (data.index.names is not None) and (data.index.names[0] is not None):
            for pair in self.config['intervals']['suffixes']:
                suffix_candidates: Dict = {n: n.split('_')[-1].lower() for n in data.index.names}
                suffixes: Dict = {k: v for k, v in suffix_candidates.items() if v in pair}
                if suffixes:
                    indexes_orig: List = data.index.names
                    data = data.reset_index()
                    num_intervals: int = int(len(suffixes.keys()) / 2)
                    for i in range(0, num_intervals):
                        keys = list(suffixes.keys())[i: i + 2]
                        base_name: str = '_'.join(keys[0].split('_')[:-1])
                        data[base_name] = pd.arrays.IntervalArray.from_arrays(left=data[keys[0]], right=data[keys[1]],
                                                                              closed=self.config['intervals']['closed'])
                        # verbose but need to preserve index order...
                        new_indexes: List = []
                        index_edge_names: Dict = {base_name: {'left': keys[0].split('_')[-1],
                                                              'right': keys[1].split('_')[-1]}}
                        for index in indexes_orig:
                            if index not in keys:
                                new_indexes.append(index)
                            if (index in keys) and (base_name not in new_indexes):
                                new_indexes.append(base_name)

                        # push the left and right names (suffixes) to the dataset attrs
                        # (series attrs are lost when set to an index)
                        data.attrs = index_edge_names
                        data.set_index(new_indexes, inplace=True)
                        data.drop(columns=keys, inplace=True)

        return data

    def _solve_mass_moisture(self, data) -> pd.DataFrame:

        d_var_map: Dict = self.variables.mass_moisture.property_to_var()
        d_var_exists: Dict = {k: v in data.columns for k, v in d_var_map.items()}
        d_mass_var_exists: Dict = {k: v in data.columns for k, v in self.variables.mass.property_to_var().items()}

        if sum(list(d_var_exists.values())) == 0:
            raise KeyError(f"Insufficient data supplied to solve mass-moisture: {d_var_exists}")
        if sum(list(d_mass_var_exists.values())) == 0:
            raise KeyError(f"At least one mass variable must be supplied to solve mass-moisture: {d_mass_var_exists}")
        if sum(list(d_var_exists.values())) == 3:
            # TODO: add mass-moisture balance integrity check.
            self._logger.info(
                'The mass-moisture variables are over-specified and not (yet) checked for balance. '
                'Moisture is ignored and the mass variables assumed to be correct.')

        # assume zero moisture
        if sum(list(d_var_exists.values())) == 1:
            data[d_var_map['moisture']] = 0.0
            self._logger.info('Zero moisture has been assumed.')

        if not d_var_exists['mass_wet']:
            data[d_var_map['mass_wet']] = solve_mass_moisture(mass_dry=data[d_var_map['mass_dry']],
                                                              moisture=data[d_var_map['moisture']])

        if not d_var_exists['mass_dry']:
            data[d_var_map['mass_dry']] = solve_mass_moisture(mass_wet=data[d_var_map['mass_wet']],
                                                              moisture=data[d_var_map['moisture']])

        # drop the moisture column, since it is now redundant, work with mass, moisture is dependent property
        if d_var_exists['moisture']:
            data.drop(columns=d_var_map['moisture'], inplace=True)

        return data

    def _dataframe_to_mc_dataset(self, data):
        # create the xr.Dataset, dims from the index.
        xr_ds: xr.Dataset = data.to_xarray()
        # move the attrs to become coords - HOLD - this creates merging problems in the data property, reconsider.
        # xr_ds = xr_ds.set_coords(cols_attrs)
        # add the dataset attributes
        ds_attrs: Dict = {'mc_name': self._name,
                          'mc_vars_mass': self.variables.mass.get_var_names(),
                          'mc_vars_chem': self.variables.chemistry.get_var_names(),
                          'mc_vars_attrs': self.variables.supplementary.get_var_names(),
                          'mc_interval_edges': data.attrs}
        xr_ds.attrs = ds_attrs
        # add the variable attributes
        for v in self.variables.xr.variables:
            xr_ds[v.name].attrs = {
                'units': self._mass_units if v.group == VariableGroups.MASS else self._composition_units,
                'standard_name': ' '.join(
                    v.name.split('_')[::-1]).title() if v.group == VariableGroups.MASS else v.name,
                'mc_type': (VariableGroups.MASS if v.group == VariableGroups.MASS else VariableGroups.CHEMISTRY).value,
                'mc_col_orig': v.column_name}
        return xr_ds

    def _check_constraints(self) -> pd.DataFrame:
        """Determine if all records are within the constraints"""
        # execute column-wise to manage memory
        df: pd.DataFrame = self.data[self.constraints.keys()].to_dataframe()
        chunks = []
        for variable, bounds in self.constraints.items():
            chunks.append(df.loc[(df[variable] < bounds[0]) | (df[variable] > bounds[1]), variable])
        oor: pd.DataFrame = pd.concat(chunks, axis='columns')
        return oor

    def _check_one_dim_interval(self):
        if len(self.data.dims) > 1:
            raise NotImplementedError(f"This object is {len(self.data.dims)} dimensional. "
                                      f"Only 1D interval objects are valid")
        index_var: str = str(list(self.data.dims.keys())[0])
        if not isinstance(self.data[index_var].data[0], pd.Interval):
            raise NotImplementedError(f"The dim {index_var} of this object is not a pd.Interval. "
                                      f" Only 1D interval objects are valid")
