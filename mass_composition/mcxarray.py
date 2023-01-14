from copy import deepcopy
from typing import Dict, List, Optional, Union

import xarray as xr

import plotly.graph_objects as go
import plotly.express as px

from mass_composition.mass_composition import MassComposition
from mass_composition.utils.components import is_compositional
from mass_composition.utils.viz import plot_parallel


@xr.register_dataset_accessor("mc")
class MassCompositionXR:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

        self._mc_obj = MassComposition(xarray_obj)

    def convert_chem_to_symbols(self) -> xr.Dataset:
        """Convert chemical components to standard symbols
        """
        self._mc_obj.convert_chem_to_symbols()
        return self._mc_obj.data

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

        variables = list(self._obj.data_vars)

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
        ds_chem: xr.Dataset = self._mc_obj.data[self._mc_obj.chem_vars]
        # TODO: improve reference to DRY mass
        ds_mass: xr.Dataset = ds_chem * self._mc_obj.data[self._mc_obj.mass_vars[-1]] / 100
        # TODO: tweak unit attributes.
        # insert into the rest of the dataset
        ds_res: xr.Dataset = self._mc_obj.data.copy()
        ds_res[self._mc_obj.chem_vars] = ds_mass
        return ds_res

    def mass_to_composition(self) -> xr.Dataset:
        """Transform mass to composition.

        Assumes incoming Dataset is mass

        :return:
        """
        # select just the chem variables
        ds_mass: xr.Dataset = self._mc_obj.data[self._mc_obj.chem_vars]
        ds_chem: xr.Dataset = ds_mass / self._mc_obj.data[self._mc_obj.mass_vars[-1]] * 100
        # insert into the rest of the dataset
        ds_res: xr.Dataset = self._mc_obj.data.copy()
        ds_res[self._mc_obj.chem_vars] = ds_chem
        return ds_res

    def split(self, fraction: float) -> tuple[xr.Dataset, xr.Dataset]:
        """Split the object at the specified mass fraction.

        Args:
            fraction: The mass fraction that defines the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        out = deepcopy(self)
        out._obj[self.mass_vars] = out._obj[self.mass_vars] * fraction
        res = out._obj
        comp = deepcopy(self)
        comp._obj[self.mass_vars] = comp._obj[self.mass_vars] * (1 - fraction)
        return res, comp._obj

    def add(self, other: xr.Dataset) -> xr.Dataset:
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.

        Args:
            other: object to add to self

        Returns:

        """

        res: xr.Dataset = self.composition_to_mass()[self.mc_vars] + MassCompositionXR(other).composition_to_mass()[
            self.mc_vars]
        res = MassCompositionXR(res).mass_to_composition()

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

        res: xr.Dataset = self.composition_to_mass()[self.mc_vars] - MassCompositionXR(other).composition_to_mass()[
            self.mc_vars]
        res = MassCompositionXR(res).mass_to_composition()

        for v in self.attr_vars:
            res[v] = self._obj[v]
        return res


