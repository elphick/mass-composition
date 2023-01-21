from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import plotly.express as px

from mass_composition.utils import solve_mass_moisture
from mass_composition.utils.components import is_compositional
from mass_composition.utils.viz import plot_parallel

# noinspection PyUnresolvedReferences
import mass_composition.mcxarray  # keep this "unused" import - it helps


class MassComposition:

    def __init__(self,
                 data: pd.DataFrame,
                 name: Optional[str] = 'unnamed',
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None,
                 mass_units: Optional[str] = 'mass units'):

        self._input_columns: List[str] = list(data.columns)
        self._mass_units = mass_units

        # TODO - try to kill the need to store these properties
        self._mass_wet_var_in = mass_wet_var
        self._mass_dry_var_in = mass_dry_var
        self._moisture_var_in = moisture_var
        self._chem_vars_in = chem_vars

        input_variables: Dict = self._detect_var_types()

        # # solve or validate the moisture balance
        if input_variables['mass_wet'] is not None:
            mass_wet: Optional[pd.Series] = data[input_variables['mass_wet']]
        else:
            mass_wet = None
        if input_variables['mass_dry'] is not None:
            mass_dry: Optional[pd.Series] = data[input_variables['mass_dry']]
        else:
            mass_dry = None
        if input_variables['moisture'] is not None:
            moisture: Optional[pd.Series] = data[input_variables['moisture']]
        else:
            moisture = None

        res: pd.Series = solve_mass_moisture(mass_wet=mass_wet,
                                             mass_dry=mass_dry,
                                             moisture=moisture)
        if mass_wet is None:
            mass_wet = res
        elif mass_dry is None:
            mass_dry = res

        # create the xr.Dataset

        # Design Decision - make mass concrete and moisture part of the dependent property
        d_da_mass: Dict[str: xr.DataArray] = {}
        for var_name, var_data in {'mass_wet': mass_wet, 'mass_dry': mass_dry}.items():
            tmp_da: xr.DataArray = xr.DataArray(var_data, name=var_name,
                                                attrs={'units': self._mass_units,
                                                       'standard_name': ' '.join(var_name.split('_')[::-1]).title(),
                                                       'mc_type': 'mass',
                                                       'mc_col_orig': var_data.name})
            d_da_mass[var_name] = tmp_da

        _data_mass: xr.Dataset = xr.Dataset(d_da_mass)

        d_da_chem: Dict[str: xr.DataArray] = {}
        for in_analyte, analyte in input_variables['chemistry'].items():
            tmp_da: xr.DataArray = xr.DataArray(data[in_analyte], name=analyte,
                                                attrs={'units': '%',
                                                       'standard_name': analyte,
                                                       'mc_type': 'chemistry',
                                                       'mc_col_orig': in_analyte})
            d_da_chem[analyte] = tmp_da

        _data_chem: xr.Dataset = xr.Dataset(d_da_chem)

        d_da_attrs: Dict[str: xr.DataArray] = {}
        for var_attr in input_variables['attrs']:
            tmp_da: xr.DataArray = xr.DataArray(data[var_attr], name=var_attr,
                                                attrs={'standard_name': var_attr,
                                                       'mc_type': 'attribute',
                                                       'mc_col_orig': var_attr})
            d_da_attrs[var_attr] = tmp_da

        _data_attrs: xr.Dataset = xr.Dataset(d_da_attrs)

        _data: xr.Dataset = xr.merge([_data_mass, _data_chem, _data_attrs])

        d_column_attrs: Dict = {'mc_name': name,
                                'mc_vars_mass': list(_data_mass.keys()),
                                'mc_vars_chem': list(_data_chem.keys()),
                                'mc_vars_attrs': list(_data_attrs.keys()),
                                'mc_history': [f'Created with name: {name}']}

        self._data = _data.assign_attrs(d_column_attrs)

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
                  as_dataframe: bool = True,
                  original_column_names: bool = False) -> Union[xr.Dataset, pd.DataFrame]:
        """Calculate the weight average.

        Args:
            group_var: Optional grouping variable
            as_dataframe: If True return a pd.DataFrame
            original_column_names: If True, and as_dataframe is True, will return with the original column names.

        Returns:

        """

        res: xr.Dataset = self._data.mc.aggregate(group_var=group_var,
                                                  as_dataframe=as_dataframe,
                                                  original_column_names=original_column_names)

        return res

    def _detect_var_types(self) -> Dict:

        # TODO: migrate candidates to config file
        res: Dict = {}

        variables = self._input_columns

        # detect the mass variables
        mass_wet_candidates: List[str] = [var for var in variables if var.lower() in ['mass_wet', 'wet_mass', 'wmt']]
        if self._mass_wet_var_in is not None:
            res['mass_wet'] = self._mass_wet_var_in
        elif len(mass_wet_candidates) == 1:
            res['mass_wet'] = mass_wet_candidates[0]
        else:
            res['mass_wet'] = None
            # raise IndexError('Mass wet variable not detected. Consider setting mass_wet_var.')

        mass_dry_candidates: List[str] = [var for var in variables if var.lower() in ['mass_dry', 'dry_mass', 'dmt']]
        if self._mass_dry_var_in is not None:
            res['mass_dry'] = self._mass_dry_var_in
        elif len(mass_dry_candidates) == 1:
            res['mass_dry'] = mass_dry_candidates[0]
        else:
            res['mass_dry'] = None
            # raise IndexError('Mass wet variable not detected. Consider setting mass_dry_var.')

        # detect the moisture variable
        # TODO: some regex work to broaden the matches
        moisture_candidates: List[str] = [var for var in variables if var.lower() in ['h2o', 'moisture']]
        if self._moisture_var_in is not None:
            res['moisture'] = self._moisture_var_in
        elif len(moisture_candidates) == 1:
            res['moisture'] = moisture_candidates[0]
        else:
            res['moisture'] = None
            # raise IndexError('Moisture variable not detected, and cannot be calculated from supplied mass data.'
            #                  'Consider setting var_moisture.')

        # detect the chemistry variable
        chemistry_var_candidates: Dict[str, str] = {k: v for k, v in
                                                    is_compositional(list(variables), strict=False).items() if
                                                    v not in ['H2O']}
        if self._chem_vars_in is not None:
            res['chemistry'] = self._chem_vars_in
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

        xr_sum: xr.Dataset = self._data.mc.sub(other._data)

        res = deepcopy(self)
        res._data = xr_sum

        return res

    def plot_parallel(self, color: Optional[str] = None, composition_only: bool = False,
                      title: Optional[str] = None) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multidimensional data like mass-composition data

        Args:
            color: Optional color variable
            composition_only: if True will limit the plot to composition components only
            title: Optional plot title

        Returns:

        """
        df = self.data.to_dataframe()
        if composition_only:
            df = df[self._data.mc_vars_chem]

        if not title and hasattr(self, 'name'):
            title = self.name

        fig = plot_parallel(data=df, color=color, title=title)
        return fig

    def plot_ternary(self, variables: List[str], color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
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
