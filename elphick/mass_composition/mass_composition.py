import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterable

import numpy as np
import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import plotly.express as px
import yaml

from elphick.mass_composition.utils import solve_mass_moisture
from elphick.mass_composition.utils.components import is_compositional
from elphick.mass_composition.utils.viz import plot_parallel

# noinspection PyUnresolvedReferences
import elphick.mass_composition.mcxarray  # keep this "unused" import - it helps


class MassComposition:

    def __init__(self,
                 data: pd.DataFrame,
                 name: Optional[str] = 'unnamed',
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None,  # ignored for now
                 mass_units: Optional[str] = 'mass units',
                 config_file: Optional[Path] = None):

        if sum(data.index.duplicated()) > 0:
            raise KeyError('The data has duplicate indexes.')

        data: pd.DataFrame = data.copy()
        self._logger = logging.getLogger(name=self.__class__.__name__)

        self._input_columns: List[str] = list(data.columns)
        self._mass_units = mass_units

        var_args: Dict = {k: v for k, v in locals().items() if '_var' in k}

        if config_file is None:
            config_file = Path(__file__).parent / './config/mc_config.yaml'
        self.config = read_yaml(config_file)

        # if interval pairs are passed as indexes then create the proper interval index
        data = self._create_interval_indexes(data=data)

        input_variables: Dict = self._detect_var_types(var_args=var_args, cols_data=list(data.columns))

        # solve or validate the moisture balance
        # TODO: this assumes 2 of the 3 vars are supplied - fix this
        data, col_map = self._solve_mass_moisture(data, input_variables)
        cols_mass = [self.config['vars']['mass_wet'], self.config['vars']['mass_dry']]
        col_map = {**col_map, **{v: k for k, v in input_variables['chemistry'].items()}}

        data.rename(columns=input_variables['chemistry'], inplace=True)

        cols_chem = list(input_variables['chemistry'].values())
        cols_attrs = input_variables['attrs']

        data = data[cols_mass + cols_chem + cols_attrs]

        # create the xr.Dataset, dims from the index.
        xr_ds: xr.Dataset = data.to_xarray()
        # move the attrs to become coords
        xr_ds = xr_ds.set_coords(cols_attrs)

        # add the dataset attributes
        ds_attrs: Dict = {'mc_name': name,
                          'mc_vars_mass': cols_mass,
                          'mc_vars_chem': cols_chem,
                          'mc_vars_attrs': cols_attrs,
                          'mc_history': [f'Created with name: {name}']}
        xr_ds.attrs = ds_attrs

        # add the variable attributes

        for var_name, var_data in {'mass_wet': xr_ds[cols_mass[0]], 'mass_dry': xr_ds[cols_mass[1]]}.items():
            var_data.attrs = {'units': self._mass_units,
                              'standard_name': ' '.join(var_name.split('_')[::-1]).title(),
                              'mc_type': 'mass',
                              'mc_col_orig': input_variables[var_name]}

        for in_analyte, analyte in input_variables['chemistry'].items():
            xr_ds[analyte].attrs = {'units': '%',
                                    'standard_name': analyte,
                                    'mc_type': 'chemistry',
                                    'mc_col_orig': in_analyte}

        for var_attr in input_variables['attrs']:
            xr_ds[var_attr].attrs = {'standard_name': var_attr,
                                     'mc_type': 'attribute',
                                     'mc_col_orig': var_attr}

        self._data = xr_ds

    @staticmethod
    def _solve_mass_moisture(data, input_variables) -> Tuple[pd.DataFrame, Dict[str, str]]:
        col_map: Dict[str, str] = {}
        if input_variables['mass_wet'] is None:
            col_map['mass_wet'] = 'mass_wet'
            data['mass_wet'] = solve_mass_moisture(mass_dry=data[input_variables['mass_dry']],
                                                   moisture=data[input_variables['moisture']])
        else:
            col_map['mass_wet'] = input_variables['mass_wet']

        if input_variables['mass_dry'] is None:
            col_map['mass_dry'] = 'mass_dry'
            data['mass_dry'] = solve_mass_moisture(mass_wet=data[input_variables['mass_wet']],
                                                   moisture=data[input_variables['moisture']])
        else:
            col_map['mass_dry'] = input_variables['mass_dry']

        if input_variables['moisture'] is None:
            col_map['H2O'] = 'H2O'
        else:
            # drop the moisture column, since it is now redundant, work with mass, moisture is dependent property
            data.drop(columns=input_variables['moisture'], inplace=True)
            col_map['H2O'] = input_variables['moisture']

        data.rename(columns={v: k for k, v in col_map.items()}, inplace=True)

        return data, col_map

    @property
    def name(self) -> str:
        return self._data.mc.name

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
            [self._data[self._data.attrs['mc_vars_mass']], moisture, self._data[self._data.attrs['mc_vars_chem']],
             self._data[self._data.attrs['mc_vars_attrs']]])
        return data

    def to_xarray(self) -> xr.Dataset:
        """Returns the mc compliant xr.Dataset

        Returns:

        """
        return self._data

    def __str__(self) -> str:
        res: str = f'\n{self.name}\n'
        res += str(self.data)
        return res

    def aggregate(self, group_var: Optional[str] = None,
                  group_bins: Optional[Union[int, Iterable]] = None,
                  as_dataframe: bool = True,
                  original_column_names: bool = False) -> Union[xr.Dataset, pd.DataFrame]:
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

    def _detect_var_types(self, var_args: Dict, cols_data: List[str]) -> Dict:

        # TODO: migrate candidates to config file
        res: Dict = {}

        variables = self._input_columns
        self._check_cols_in_data_cols(cols=list(var_args.values()), cols_data=cols_data)

        # detect the mass variables

        mass_wet_candidates: List[str] = [var for var in variables if var.lower() in ['mass_wet', 'wet_mass', 'wmt']]
        if var_args['mass_wet_var'] is not None:
            res['mass_wet'] = var_args['mass_wet_var']
        elif len(mass_wet_candidates) == 1:
            res['mass_wet'] = mass_wet_candidates[0]
        else:
            res['mass_wet'] = None
            # raise IndexError('Mass wet variable not detected. Consider setting mass_wet_var.')

        mass_dry_candidates: List[str] = [var for var in variables if var.lower() in ['mass_dry', 'dry_mass', 'dmt']]
        if var_args['mass_dry_var'] is not None:
            res['mass_dry'] = var_args['mass_dry_var']
        elif len(mass_dry_candidates) == 1:
            res['mass_dry'] = mass_dry_candidates[0]
        else:
            res['mass_dry'] = None
            # raise IndexError('Mass wet variable not detected. Consider setting mass_dry_var.')

        # detect the moisture variable
        # TODO: some regex work to broaden the matches
        moisture_candidates: List[str] = [var for var in variables if var.lower() in ['h2o', 'moisture']]
        if var_args['moisture_var'] is not None:
            res['moisture'] = var_args['moisture_var']
        elif len(moisture_candidates) == 1:
            res['moisture'] = moisture_candidates[0]
        else:
            res['moisture'] = None
            # raise IndexError('Moisture variable not detected, and cannot be calculated from supplied mass data.'
            #                  'Consider setting var_moisture.')

        # detect the chemistry variable
        chem_ignore: List[str] = ['H2O'] + self.config['components']['chemistry']['ignore']
        chem_ignore = list(set(chem_ignore + [c.lower() for c in chem_ignore] + [c.upper() for c in chem_ignore]))
        chemistry_var_candidates: Dict[str, str] = {k: v for k, v in
                                                    is_compositional(list(variables), strict=False).items() if
                                                    v not in chem_ignore}
        if var_args['chem_vars'] is not None:
            res['chemistry'] = var_args['chem_vars']
        elif len(chemistry_var_candidates.keys()) > 0:
            res['chemistry'] = chemistry_var_candidates
        else:
            raise IndexError('Chemistry variables not detected. Consider setting vars_chemistry.')

        # report any remaining variables as attrs
        vars_known_dims: List[str] = [v for k, v in res.items() if (k != 'chemistry') and v is not None]
        if isinstance(res['chemistry'], Dict):
            vars_known_dims.extend(list(res['chemistry'].keys()))
        else:
            vars_known_dims.extend(list(res['chemistry']))
        vars_attrs: List[str] = list(set(list(variables)).difference(set(vars_known_dims)))
        res['attrs'] = vars_attrs

        return res

    @staticmethod
    def _check_cols_in_data_cols(cols: List[str], cols_data: List[str]):
        for col in cols:
            if (col is not None) and (col not in cols_data):
                msg: str = f"{col} not in the data columns: {cols_data}"
                logging.error(msg)
                raise IndexError(msg)

    def split(self, fraction: float) -> tuple['MassComposition', 'MassComposition']:
        """Split the object at the specified mass fraction.

        Args:
            fraction: The mass fraction that defines the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        out = deepcopy(self)
        comp = deepcopy(self)

        xr_ds_1, xr_ds_2 = self._data.mc.split(fraction=fraction)
        out._data = xr_ds_1
        comp._data = xr_ds_2

        return out, comp

    def __add__(self, other: 'MassComposition') -> 'MassComposition':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """
        xr_sum: xr.Dataset = self._data.mc.add(other._data)

        res = deepcopy(self)
        res._data = xr_sum

        return res

    def __sub__(self, other: 'MassComposition') -> 'MassComposition':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        xr_sub: xr.Dataset = self._data.mc.sub(other._data)

        res = deepcopy(self)
        res._data = xr_sub

        return res

    def __truediv__(self, other: 'MassComposition') -> 'MassComposition':
        """Divide self by the supplied object

        Perform the division with the mass-composition variables only and then append any attribute variables.
        Args:
            other: denominator object, self will be divided by this object

        Returns:

        """

        xr_div: xr.Dataset = self._data.mc.div(other._data)

        res = deepcopy(self)
        res._data = xr_div

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

    def plot_parallel(self, color: Optional[str] = None,
                      var_subset: Optional[List[str]] = None,
                      title: Optional[str] = None,
                      include_dims: Optional[Union[bool, List[str]]] = True,
                      plot_interval_edges: bool = False) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multidimensional data like mass-composition data

        Args:
            color: Optional color variable
            var_subset: List of variables to include in the plot
            title: Optional plot title
            include_dims: Optional boolean or list of dimension to include in the plot.  True will show all dims.
            plot_interval_edges: If True, interval edges will be plotted instead of interval mid

        Returns:

        """
        df = self.data.mc.to_dataframe()

        if var_subset is not None:
            missing_vars = set(var_subset).difference(set(df.columns))
            if len(missing_vars) > 0:
                raise KeyError(f'var_subset provided contains variable not found in the data: {missing_vars}')
            df = df[var_subset]

        if include_dims is True:
            df.reset_index(inplace=True)
        elif isinstance(include_dims, List):
            for d in include_dims:
                df.reset_index(d, inplace=True)

        interval_cols: Dict[str, int] = {col: i for i, col in enumerate(df.columns) if df[col].dtype == 'interval'}

        for col, pos in interval_cols.items():
            if plot_interval_edges:
                df.insert(loc=pos + 1, column=f'{col}_left', value=df[col].array.left)
                df.insert(loc=pos + 2, column=f'{col}_right', value=df[col].array.right)
                df.drop(columns=col, inplace=True)
            else:
                df[col] = df[col].array.mid

        if not title and hasattr(self, 'name'):
            title = self.name

        fig = plot_parallel(data=df, color=color, title=title)
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

    def _create_interval_indexes(self, data: pd.DataFrame) -> pd.DataFrame:

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
                    data[base_name] = pd.IntervalIndex.from_arrays(left=data[keys[0]],
                                                                   right=data[keys[1]],
                                                                   closed=self.config['intervals']['closed'])
                    # verbose but need to preserve index order...
                    new_indexes: List = []
                    for index in indexes_orig:
                        if index not in keys:
                            new_indexes.append(index)
                        if (index in keys) and (base_name not in new_indexes):
                            new_indexes.append(base_name)
                    data.set_index(new_indexes, inplace=True)
                    data.drop(columns=keys, inplace=True)

        return data


def read_yaml(file_path):
    with open(file_path, "r") as f:
        d_config: Dict = yaml.safe_load(f)
        if 'MC' != list(d_config.keys())[0]:
            msg: str = f'config file {file_path} is not a MassComposition config file - no MC key'
            logging.error(msg)
            raise KeyError(msg)
        return d_config['MC']
