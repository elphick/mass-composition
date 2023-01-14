from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import plotly.express as px

from mass_composition.utils import solve_mass_moisture
from mass_composition.utils.components import is_compositional
from mass_composition.utils.viz import plot_parallel


class MassComposition:
    def __init__(self,
                 data: Union[xr.Dataset, pd.DataFrame],
                 mass_vars: Optional[List[str]] = None,
                 moisture_var: Optional[str] = None,
                 chem_vars: Optional[List[str]] = None
                 ):

        if isinstance(data, pd.DataFrame):
            data = data.to_xarray()

        self.data = data

        self.mass_vars = None
        self.moisture_var = None
        self.chem_vars = None

        d_var_types: Dict = self._detect_var_types()

        self.mass_vars = d_var_types['mass']
        self.moisture_var = d_var_types['moisture']
        self.chem_vars = list(d_var_types['chemistry'].keys())
        self.attr_vars = d_var_types['attrs']

        # # solve or validate the moisture balance
        self._solve_mass_moisture()

        self.mc_vars = [v for v in list(self.data.data_vars) if v not in self.attr_vars]

        self.chemistry_var_map = d_var_types['chemistry']  # {in: symbols}

    def _solve_mass_moisture(self):
        # TODO: avoid the hard-code names
        if not self.moisture_var and len(self.mass_vars) == 2:
            # TODO: fix - relies on wet being to the left or dry - need to be explicit.
            moisture: xr.DataArray = solve_mass_moisture(mass_wet=self.data[self.mass_vars[0]],
                                                         mass_dry=self.data[self.mass_vars[1]])
            self.data = xr.merge([self.data[self.mass_vars], moisture.to_dataset(),
                                  self.data[self.chem_vars], self.data[self.attr_vars]])
            self.moisture_var = 'H2O'
        elif len(self.mass_vars) == 1:
            var_to_solve: str = self.mass_vars[0]
            if var_to_solve == 'mass_wet':
                mass_dry: xr.DataArray = solve_mass_moisture(mass_wet=self.data['mass_wet'],
                                                             moisture=self.data[self.moisture_var[0]])
                self.data = xr.merge([self.data['mass_wet'], mass_dry.to_dataset(),
                                      self.data[self.moisture_var[0]],
                                      self.data[self.chem_vars], self.data[self.attr_vars]])
            elif var_to_solve == 'mass_dry':
                mass_wet: xr.DataArray = solve_mass_moisture(mass_dry=self.data['mass_dry'],
                                                             moisture=self.data[self.moisture_var[0]])
                self.data = xr.merge([mass_wet.to_dataset(), self.data['mass_dry'],
                                      self.data[self.moisture_var[0]],
                                      self.data[self.chem_vars], self.data[self.attr_vars]])
            self.mass_vars = ['mass_wet', 'mass_dry']

    def __str__(self) -> str:
        res: str = '\n'
        res += f'mass_vars: {self.mass_vars}\n'
        res += f'moisture_var: {self.moisture_var}\n'
        res += f'chem_vars: {self.chem_vars}\n'
        res += f'attr_vars: {self.attr_vars}\n'
        res += '\n'
        res += str(self.data)
        return res

    def convert_chem_to_symbols(self):
        """Convert chemical components to standard symbols
        """
        self.data = self.data.rename(self.chemistry_var_map)
        self.chem_vars = list(self.chemistry_var_map.values())
        self.mc_vars = [v for v in list(self.data.data_vars) if v not in self.attr_vars]

    def aggregate(self, group_var: Optional[str] = None) -> xr.Dataset:
        """Calculate the weight average of this dataset.
        """
        if group_var is None:
            res: xr.Dataset = self.data.mc.composition_to_mass().sum().mc.mass_to_composition()
        else:
            res: xr.Dataset = self.data.mc.composition_to_mass().groupby(group_var).sum().mc.mass_to_composition()

        # If the Dataset is 0D add a placeholder index
        if len(res.dims) == 0:
            res = res.expand_dims('index')
        return res

    def _detect_var_types(self) -> Dict:

        res: Dict = {}

        variables = list(self.data.data_vars)

        # detect the mass variables
        mass_var_candidates: List[str] = [v for v in variables if 'mass' in v]
        if self.mass_vars is not None:
            res['mass'] = self.mass_vars
        elif len(mass_var_candidates) in [1, 2]:
            res['mass'] = mass_var_candidates
        else:
            raise IndexError('Mass variable/s not detected. Consider setting vars_mass.')

        # detect the moisture variable
        # TODO: some regex work to broaden the matches
        moisture_var_candidates: List[str] = [var for var in variables if var.lower() in ['h2o', 'moisture']]
        if self.moisture_var is not None:
            res['moisture'] = self.moisture_var
        elif len(moisture_var_candidates) == 1:
            res['moisture'] = moisture_var_candidates
        elif len(res['mass']) == 2:
            res['moisture'] = []
        else:
            raise IndexError('Moisture variable not detected, and cannot be calculated from supplied mass data.'
                             'Consider setting var_moisture.')

        # detect the chemistry variable
        chemistry_var_candidates: Dict[str, str] = {k: v for k, v in
                                                    is_compositional(list(variables), strict=False).items() if
                                                    v not in ['H2O']}
        if self.chem_vars is not None:
            res['chemistry'] = self.chem_vars
        elif len(chemistry_var_candidates.keys()) > 0:
            res['chemistry'] = chemistry_var_candidates
        else:
            raise IndexError('Chemistry variables not detected. Consider setting vars_chemistry.')

        # report any remaining variables as attrs
        vars_known_dims: List[str] = res['mass'] + res['moisture'] + list(res['chemistry'].keys())
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
        out.data[self.mass_vars] = out.data[self.mass_vars] * fraction
        res = out
        comp = deepcopy(self)
        comp.data[self.mass_vars] = comp.data[self.mass_vars] * (1 - fraction)
        return res, comp

    def __add__(self, other: 'MassComposition') -> 'MassComposition':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """

        xr_mass: xr.Dataset = self.data.mc.composition_to_mass()[self.mc_vars] + other.data.mc.composition_to_mass()[
            self.mc_vars]
        xr_composition: xr.Dataset = xr_mass.mc.mass_to_composition()

        # add back the attributes
        for v in self.attr_vars:
            xr_composition[v] = self.data[v]

        res: MassComposition = MassComposition(data=xr_composition)

        return res

    def __sub__(self, other: 'MassComposition') -> 'MassComposition':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        xr_mass: xr.Dataset = self.data.mc.composition_to_mass()[self.mc_vars] - other.data.mc.composition_to_mass()[
            self.mc_vars]
        xr_composition: xr.Dataset = xr_mass.mc.mass_to_composition()

        # add back the attributes
        for v in self.attr_vars:
            xr_composition[v] = self.data[v]

        res: MassComposition = MassComposition(data=xr_composition)
        return res

    def plot_parallel(self, color: Optional[str] = None, composition_only: bool = False, title: Optional[str] = None) -> go.Figure:
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
