import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union, Iterable, List

import pandas as pd
import xarray as xr

from elphick.mc.mass_composition.utils import solve_mass_moisture


class CompositionContext(Enum):
    ABSOLUTE = 'mass'
    RELATIVE = "percent"


@xr.register_dataset_accessor("mc")
class MassCompositionAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

        self._logger = logging.getLogger(name=self.__class__.__name__)

        self.mc_vars = self._obj.mc_vars_mass + self._obj.mc_vars_chem
        self.mc_vars_mass = self._obj.mc_vars_mass
        self.mc_vars_chem = self._obj.mc_vars_chem
        self.mc_vars_attrs = self._obj.mc_vars_attrs
        self._chem: xr.Dataset = self._obj[self.mc_vars_chem]
        self._mass: xr.Dataset = self._obj[self.mc_vars_mass]

    @property
    def name(self):
        return self._obj.attrs['mc_name']

    @property
    def history(self):
        return self._obj.attrs['mc_history']

    def data(self):

        moisture: xr.DataArray = xr.DataArray((self._obj['mass_wet'] - self._obj['mass_dry']) /
                                              self._obj['mass_wet'] * 100, name='H2O',
                                              attrs={'units': '%',
                                                     'standard_name': 'H2O',
                                                     'mc_type': 'moisture',
                                                     'mc_col_orig': 'H2O'}
                                              )

        data: xr.Dataset = xr.merge(
            [self._obj[self._obj.mc_vars_mass], moisture, self._obj[self._obj.mc_vars_chem],
             self._obj[self._obj.mc_vars_attrs]])
        return data

    @property
    def composition_context(self) -> CompositionContext:

        units = [da.attrs['units'] for _, da in self._obj[self._obj.mc_vars_chem].items()]
        if len(set(units)) == 1:
            if units[0] == '%':
                res = CompositionContext.RELATIVE
            elif units[0] == self._obj['mass_dry'].attrs['units']:
                res = CompositionContext.ABSOLUTE
            else:
                raise KeyError("Chemistry units do not conform")
        else:
            raise KeyError("Chemistry units are inconsistent")
        return res

    def rename(self, new_name: str):
        self._obj.attrs['mc_name'] = new_name
        self.log_to_history(f'Renamed to {new_name}')

    def log_to_history(self, msg: str):
        self._logger.info(msg)
        self._obj.attrs['mc_history'].append(msg)

    def aggregate(self, group_var: Optional[str] = None,
                  group_bins: Optional[Union[int, Iterable]] = None,
                  as_dataframe: bool = False,
                  original_column_names: bool = False) -> Union[xr.Dataset, pd.DataFrame]:
        """Calculate the weight average of this dataset.

        Args:
            group_var: Optional grouping variable
            group_bins: Optional bins to apply to the group_var
            as_dataframe: If True return a pd.DataFrame
            original_column_names: If True, and as_dataframe is True, will return with the original column names.

        Returns:

        """

        if group_var is None:
            res = mc_aggregate(self._obj)

        else:
            if group_var not in self._obj.dims:
                if group_bins is None:
                    res = self._obj.groupby(group_var).map(mc_aggregate)
                else:
                    res = self._obj.groupby_bins(group_var, bins=group_bins).map(mc_aggregate)

            elif group_var in self._obj.dims:
                # TODO: consider a better way - maybe reset the index?
                # sum across all dims other than the one of interest...
                other_dims = [gv for gv in self._obj.dims if gv != group_var]
                res: xr.Dataset = self.composition_to_mass().sum(other_dims,
                                                                 keep_attrs=True).mc.mass_to_composition()
            else:
                raise KeyError(f'{group_var} not found in dataset')

            res.mc.rename(f'Aggregate of {self.name} with group {group_var}')

        ds_moisture = solve_mass_moisture(mass_wet=res[res.mc_vars_mass[0]],
                                          mass_dry=res[res.mc_vars_mass[1]]).to_dataset()
        ds_moisture['H2O'].attrs = {'units': '%',
                                    'standard_name': 'H2O',
                                    'mc_type': 'moisture',
                                    'mc_col_orig': 'H2O'}

        res: xr.Dataset = xr.merge([res[res.mc_vars_mass], ds_moisture, res[res.mc_vars_chem]])

        # If the Dataset is 0D add a placeholder index
        if len(res.dims) == 0:
            res = res.expand_dims('index')

        if as_dataframe:
            res: pd.DataFrame = self.to_dataframe(original_column_names=original_column_names,
                                                  ds=res)

        return res

    def cumulate(self, direction: str) -> xr.Dataset:
        """Cumulate along the dims

        Expected use case in only for Datasets that have been reduced to 1D.

        Args:
            direction: 'ascending'|'descending'

        Returns:

        """

        valid_dirs: List[str] = ['ascending', 'descending']
        if direction not in valid_dirs:
            raise KeyError(f'Invalid direction provided.  Valid arguments are: {valid_dirs}')

        if len(self._obj.dims) > 1:
            raise NotImplementedError('Datasets > 1D have not been tested.')

        # convert to mass, then cumsum, then convert back to relative composition (grade)
        mass: xr.Dataset = self.composition_to_mass()

        index_var = list(mass.indexes)[0]
        if direction == 'descending':
            mass = mass.sortby(variables=index_var, ascending=False)

        mass_cum: xr.Dataset = mass.cumsum(keep_attrs=True)
        # put the coords back
        mass_cum = mass_cum.assign_coords(**mass.coords)
        res: xr.Dataset = mass_cum.mc.mass_to_composition()

        if direction == 'descending':
            # put back to ascending order
            res = res.sortby(variables=index_var, ascending=True)

        return res

    def composition_to_mass(self) -> xr.Dataset:
        """Transform composition to mass

        :return:
        """

        if self.composition_context == CompositionContext.ABSOLUTE:
            raise AssertionError('The dataset composition context is already absolute (mass units)')

        xr.set_options(keep_attrs=True)

        dsm: xr.Dataset = self._obj.copy()
        dsm_chem: xr.DataArray = dsm[self._obj.mc_vars_chem] * self._obj['mass_dry'] / 100

        dsm[self._obj.mc_vars_chem] = (dsm[self._obj.mc_vars_chem] * self._obj['mass_dry'] / 100)

        for da in dsm.values():
            if da.attrs['mc_type'] == 'chemistry':
                da.attrs['units'] = dsm['mass_wet'].attrs['units']

        self.log_to_history(f'Converted to {self.composition_context}, dropped attr variables')

        xr.set_options(keep_attrs='default')

        return dsm

    def mass_to_composition(self) -> xr.Dataset:
        """Transform mass to composition.

        :return:
        """

        if self.composition_context == CompositionContext.RELATIVE:
            raise AssertionError('The dataset composition context is already relative (% units)')

        xr.set_options(keep_attrs=True)

        dsc: xr.Dataset = self._obj.copy()
        dsc[self._obj.mc_vars_chem] = (dsc[self._obj.mc_vars_chem] / self._obj['mass_dry'] * 100)

        for da in dsc.values():
            if da.attrs['mc_type'] == 'chemistry':
                da.attrs['units'] = '%'

        self.log_to_history(f'Converted to {self.composition_context}')

        xr.set_options(keep_attrs='default')

        return dsc

    def split(self, fraction: float) -> tuple[xr.Dataset, xr.Dataset]:
        """Split the object at the specified mass fraction.

        Args:
            fraction: The mass fraction that defines the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """

        xr.set_options(keep_attrs=True)

        out = deepcopy(self)
        out._obj[self._obj.mc_vars_mass] = out._obj[self._obj.mc_vars_mass] * fraction
        out.log_to_history(f'Split from object [{self.name}] @ fraction: {fraction}')
        out.rename(f'({fraction} * {self.name})')
        res = out._obj

        comp = deepcopy(self)
        comp._obj[self._obj.mc_vars_mass] = comp._obj[self._obj.mc_vars_mass] * (1 - fraction)
        comp.log_to_history(f'Split from object [{self.name}] @ 1 - fraction {fraction}: {1 - fraction}')
        comp.rename(f'({1 - fraction} * {self.name})')

        xr.set_options(keep_attrs='default')

        return res, comp._obj

    def add(self, other: xr.Dataset) -> xr.Dataset:
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.

        Args:
            other: object to add to self

        Returns:

        """

        xr.set_options(keep_attrs=True)

        # initially just add the mass and composition, not the attr vars
        xr_self: xr.Dataset = self.composition_to_mass()[self.mc_vars]
        xr_other: xr.Dataset = other.mc.composition_to_mass()[self.mc_vars]

        res: xr.Dataset = xr_self + xr_other

        res = self._math_post_process(other, res, xr_self, 'added')

        xr.set_options(keep_attrs='default')

        return res

    def sub(self, other: xr.Dataset) -> xr.Dataset:
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        xr.set_options(keep_attrs=True)

        # initially just add the mass and composition, not the attr vars
        xr_self: xr.Dataset = self.composition_to_mass()[self.mc_vars]
        xr_other: xr.Dataset = other.mc.composition_to_mass()[self.mc_vars]

        res: xr.Dataset = xr_self - xr_other

        res = self._math_post_process(other, res, xr_self, 'subtracted')

        xr.set_options(keep_attrs='default')

        return res

    def _math_post_process(self, other, res, xr_self, operator_string):
        # update attrs
        res.attrs.update(self._obj.attrs)
        da: xr.DataArray
        for new_da, da in zip(res.values(), xr_self.values()):
            new_da.attrs.update(da.attrs)
        # merge in the attr vars
        res = xr.merge([res, self._obj[self._obj.mc_vars_attrs]])
        res.mc.log_to_history(f'Object called {other.mc.name} has been {operator_string}.')
        res.mc.rename(f'({self.name} - {other.mc.name})')
        # convert back to relative composition
        res = res.mc.mass_to_composition()
        return res

    def column_map(self):
        res: Dict = {var: da.attrs['mc_col_orig'] for var, da in self._obj.items()}
        return res

    def to_dataframe(self, original_column_names: bool = False, ds: Optional[xr.Dataset] = None) -> pd.DataFrame:

        if ds is None:
            ds = self._obj.mc.data()

        df: pd.DataFrame = ds.to_dataframe()

        # order with mass, chem then attr columns last
        col_order: List[str] = self.mc_vars_mass + ['H2O'] + self.mc_vars_chem + self.mc_vars_attrs
        df = df[[col for col in col_order if col in df.columns]]

        if original_column_names:
            df.rename(columns=self.column_map(), inplace=True)
        return df


def mc_aggregate(xr_ds: xr.Dataset) -> xr.Dataset:
    """A standalone function to aggregate

    Sum the mass vars and weight average the chem vars by the mass_dry.

    Args:
        xr_ds: A MassComposition compliant xr.Dataset

    Returns:

    """
    chem: xr.Dataset = xr_ds.mc._chem.weighted(weights=xr_ds.mc._mass['mass_dry'].fillna(0)).mean(keep_attrs=True)
    mass: xr.Dataset = xr_ds.mc._mass.sum(keep_attrs=True)
    res: xr.Dataset = xr.merge([mass, chem])
    res.attrs['mc_vars_attrs'] = []
    res.mc.rename(f'Aggregate of {xr_ds.mc.name}')
    return res
