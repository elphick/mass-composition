from copy import deepcopy
from typing import Dict, List, Optional, Union

import xarray as xr

import plotly.graph_objects as go
import plotly.express as px

from mcxarray.utils.components import is_compositional


@xr.register_dataset_accessor("mc")
class MassComposition:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

        self.mass_vars = None
        self.moisture_var = None
        self.chem_vars = None

        d_var_types: Dict = self._detect_var_types()

        self.mass_vars = d_var_types['mass']
        self.moisture_var = d_var_types['moisture']
        self.chem_vars = list(d_var_types['chemistry'].keys())
        self.attr_vars = d_var_types['attrs']

        self.mc_vars = [v for v in list(self._obj.data_vars) if v not in self.attr_vars]

        self.chemistry_var_map = d_var_types['chemistry']  # {in: symbols}

    def convert_chem_to_symbols(self) -> xr.Dataset:
        """Convert chemical components to standard symbols
        """
        return self._obj.rename(self.chemistry_var_map)

    def aggregate(self, group_var: Optional[str] = None) -> xr.Dataset:
        """Calculate the weight average of this dataset.
        """
        if group_var is None:
            res: xr.Dataset = self.composition_to_mass().sum().mc.mass_to_composition()
        else:
            res: xr.Dataset = self.composition_to_mass().groupby(group_var).sum().mc.mass_to_composition()

        # If the Dataset is 0D add a placeholder index
        if len(res.dims) == 0:
            res = res.expand_dims('index')
        return res

    def _detect_var_types(self) -> Dict:

        res: Dict = {}

        variables = self._obj.data_vars

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
        moisture_var_candidates: List[str] = [var for var in variables if var.lower() in ['h20', 'moisture']]
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
        chemistry_var_candidates: Dict[str, str] = is_compositional(list(variables), strict=False)
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

    def composition_to_mass(self) -> xr.Dataset:
        """Transform composition to mass

        :return:
        """

        # select just the chem variables
        ds_chem: xr.Dataset = self._obj[self.chem_vars]
        # TODO: improve reference to DRY mass
        ds_mass: xr.Dataset = ds_chem * self._obj[self.mass_vars[-1]] / 100
        # TODO: tweak unit attributes.
        # insert into the rest of the dataset
        ds_res: xr.Dataset = self._obj.copy()
        ds_res[self.chem_vars] = ds_mass
        return ds_res

    def mass_to_composition(self) -> xr.Dataset:
        """Transform mass to composition.

        Assumes incoming Dataset is mass

        :return:
        """
        # select just the chem variables
        ds_mass: xr.Dataset = self._obj[self.chem_vars]
        ds_chem: xr.Dataset = ds_mass / self._obj[self.mass_vars[-1]] * 100
        # insert into the rest of the dataset
        ds_res: xr.Dataset = self._obj.copy()
        ds_res[self.chem_vars] = ds_chem
        return ds_res

    def split(self, fraction: float, return_complement: bool = True) -> Union[
        xr.Dataset, tuple[xr.Dataset, xr.Dataset]]:
        """Split the object at the specified mass fraction.

        Args:
            fraction: The mass fraction that defines the split
            return_complement: If True, return a tuple of objects, the latter being the complement

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        out = deepcopy(self)
        out._obj[self.mass_vars] = out._obj[self.mass_vars] * fraction
        res = out._obj
        if return_complement:
            comp = deepcopy(self)
            comp._obj[self.mass_vars] = comp._obj[self.mass_vars] * (1 - fraction)
            res = (res, comp._obj)
        return res

    def add(self, other: xr.Dataset) -> xr.Dataset:
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.

        Args:
            other: object to add to self

        Returns:

        """

        res: xr.Dataset = self.composition_to_mass()[self.mc_vars] + MassComposition(other).composition_to_mass()[
            self.mc_vars]
        res = MassComposition(res).mass_to_composition()

        for v in self.attr_vars:
            res[v] = self._obj[v]
        return res

    def minus(self, other: xr.Dataset) -> xr.Dataset:
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        res: xr.Dataset = self.composition_to_mass()[self.mc_vars] - MassComposition(other).composition_to_mass()[
            self.mc_vars]
        res = MassComposition(res).mass_to_composition()

        for v in self.attr_vars:
            res[v] = self._obj[v]
        return res

    def plot_parallel(self, color: Optional[str] = None, composition_only: bool = False) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multi-dimensional data like mass-composition data

        Args:
            color: Optional color variable
            composition_only: if True will limit the plot to composition components only

        Returns:

        """
        df = self._obj.to_dataframe()
        if composition_only:
            df = df[self.chem_vars]

        # Kudos: https://stackoverflow.com/questions/72125802/parallel-coordinate-plot-in-plotly-with-continuous-
        # and-categorical-data

        categorical_columns = df.select_dtypes(include=['category', 'object'])
        col_list = []

        for col in df.columns:
            if col in categorical_columns:  # categorical columns
                values = df[col].unique()
                value2dummy = dict(zip(values, range(
                    len(values))))  # works if values are strings, otherwise we probably need to convert them
                df[col] = [value2dummy[v] for v in df[col]]
                col_dict = dict(
                    label=col,
                    tickvals=list(value2dummy.values()),
                    ticktext=list(value2dummy.keys()),
                    values=df[col],
                )
            else:  # continuous columns
                col_dict = dict(
                    range=(df[col].min(), df[col].max()),
                    label=col,
                    values=df[col],
                )
            col_list.append(col_dict)
        # fig = go.Figure(data=go.Parcoords(dimensions=col_list)

        if color is None:
            # fig: go.Figure = px.parallel_coordinates(col_list)
            fig = go.Figure(data=go.Parcoords(dimensions=col_list))

        else:
            # if color in ].dtype == 'object':
            # df[color] = pd.Categorical(df[color])
            # df[f'{color}_id'] = df[color].cat.codes
            # fig: go.Figure = px.parallel_coordinates(col_list, color=color)
            fig = go.Figure(data=go.Parcoords(dimensions=col_list, line=dict(color=df[color])))

        return fig

    def plot_ternary(self, variables: List[str], color: Optional[str] = None) -> go.Figure:
        """Plot a ternary diagram
            variables: List of 3 components to plot
            color: Optional color variable

        """

        df = self._obj.to_dataframe()
        if color:
            fig = px.scatter_ternary(df, a=variables[0], b=variables[1], c=variables[2], color=color)
        else:
            fig = px.scatter_ternary(df, a=variables[0], b=variables[1], c=variables[2])

        return fig
