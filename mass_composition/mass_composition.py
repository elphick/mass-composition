from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import plotly.express as px

from mass_composition.utils import solve_mass_moisture
from mass_composition.utils.components import is_compositional
from mass_composition.utils.transform import mass_to_composition, composition_to_mass
from mass_composition.utils.viz import plot_parallel


class MassComposition:
    def __init__(self,
                 data: pd.DataFrame,
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None
                 ):

        self.input_columns: List[str] = list(data.columns)
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

        # create the xr object

        # Design Decision - make mass concrete and moisture part of the dependent property
        self._data_mass: xr.Dataset = xr.Dataset(pd.concat([mass_wet, mass_dry], axis='columns'))
        # TODO: resolve how chem can be a dict or a list - try for consistency going forward
        df_chem: pd.DataFrame = data[input_variables['chemistry']].copy()
        if isinstance(input_variables['chemistry'], Dict):
            df_chem.rename(columns=input_variables['chemistry'])
        self._data_chem: xr.Dataset = xr.Dataset(df_chem)

        self._data_attrs: xr.Dataset = xr.Dataset(data[input_variables['attrs']].copy())

        # noinspection PyTypeChecker
        self.mass_wet_var: str = mass_wet.name
        # noinspection PyTypeChecker
        self.mass_dry_var: str = mass_dry.name
        # noinspection PyTypeChecker
        self.mass_vars: List[str] = [mass_wet.name, mass_dry.name]
        self.chem_vars: List[str] = list(self._data_chem.to_dataframe().columns)
        self.attr_vars: List[str] = list(self._data_attrs.to_dataframe().columns)

        self.chemistry_var_map = input_variables['chemistry']  # {in: symbols}

    @property
    def data(self):
        moisture: xr.DataArray = (self._data_mass[self.mass_wet_var] - self._data_mass[self.mass_dry_var]) / \
                                 self._data_mass[self.mass_wet_var] * 100
        moisture.name = 'H2O'

        data: xr.Dataset = xr.merge([self._data_mass, moisture, self._data_chem, self._data_attrs])
        return data

    def as_mass(self) -> xr.Dataset:
        """Mass and Composition converted to mass units

        Used for math operations
        Returns:

        """
        res: xr.Dataset = xr.merge([self._data_mass, self._data_chem * self._data_mass[self.mass_dry_var] / 100])
        return res

    def __str__(self) -> str:
        res: str = '\n'
        # res += f'mass_vars: {self.mass_vars}\n'
        # res += f'moisture_var: {self.moisture_var}\n'
        # res += f'chem_vars: {self.chem_vars}\n'
        # res += f'attr_vars: {self.attr_vars}\n'
        # res += '\n'
        res += str(self.data)
        return res

    def aggregate(self, group_var: Optional[str] = None) -> xr.Dataset:
        """Calculate the weight average of this dataset.
        """
        if group_var is None:

            xr_mass: xr.Dataset = composition_to_mass(mass_wet=self._data_mass[self.mass_wet_var],
                                                      mass_dry=self._data_mass[self.mass_dry_var],
                                                      composition=self._data_chem[self.chem_vars]).sum()

        else:

            xr_mass: xr.Dataset = composition_to_mass(mass_wet=self._data_mass[self.mass_wet_var],
                                                      mass_dry=self._data_mass[self.mass_dry_var],
                                                      composition=self._data_chem[self.chem_vars],
                                                      attributes=self._data_attrs).groupby(group_var).sum()

        res: xr.Dataset = mass_to_composition(mass_wet=xr_mass[self.mass_wet_var],
                                              mass_dry=xr_mass[self.mass_dry_var],
                                              component_mass=xr_mass[self.chem_vars])

        # If the Dataset is 0D add a placeholder index
        if len(res.dims) == 0:
            res = res.expand_dims('index')
        return res

    def _detect_var_types(self) -> Dict:

        # TODO: migrate candidates to config file
        res: Dict = {}

        variables = self.input_columns

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
        out._data_mass = out._data_mass * fraction
        res = out
        comp = deepcopy(self)
        comp._data_mass = comp._data_mass * (1 - fraction)
        return res, comp

    def __add__(self, other: 'MassComposition') -> 'MassComposition':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """

        xr_component_mass: xr.Dataset = self.as_mass() + other.as_mass()

        xr_composition: xr.Dataset = mass_to_composition(mass_wet=xr_component_mass[self.mass_wet_var],
                                                         mass_dry=xr_component_mass[self.mass_dry_var],
                                                         component_mass=xr_component_mass[self.chem_vars],
                                                         attributes=self._data_attrs)

        res: MassComposition = MassComposition(data=xr_composition.to_dataframe())

        return res

    def __sub__(self, other: 'MassComposition') -> 'MassComposition':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        xr_component_mass: xr.Dataset = self.as_mass() - other.as_mass()

        xr_composition: xr.Dataset = mass_to_composition(mass_wet=xr_component_mass[self.mass_wet_var],
                                                         mass_dry=xr_component_mass[self.mass_dry_var],
                                                         component_mass=xr_component_mass[self.chem_vars],
                                                         attributes=self._data_attrs)

        res: MassComposition = MassComposition(data=xr_composition.to_dataframe())

        return res

    def plot_parallel(self, color: Optional[str] = None, composition_only: bool = False,
                      title: Optional[str] = None) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multi-dimensional data like mass-composition data

        Args:
            color: Optional color variable
            composition_only: if True will limit the plot to composition components only
            title: Optional plot title

        Returns:

        """
        df = self.data.to_dataframe()
        if composition_only:
            df = df[self.chem_vars]

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
