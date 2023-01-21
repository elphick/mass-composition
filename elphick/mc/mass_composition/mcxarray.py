import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union

import pandas as pd
import xarray as xr


class CompositionContext(Enum):
    ABSOLUTE = 'mass'
    RELATIVE = "percent"


@xr.register_dataset_accessor("mc")
class MassCompositionXR:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

        self._logger = logging.getLogger(name=self.__class__.__name__)

        self.mc_vars = self._obj.mc_vars_mass + self._obj.mc_vars_chem

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
                  as_dataframe: bool = False,
                  original_column_names: bool = False) -> Union[xr.Dataset, pd.DataFrame]:
        """Calculate the weight average of this dataset.

        Args:
            group_var: Optional grouping variable
            as_dataframe: If True return a pd.DataFrame
            original_column_names: If True, and as_dataframe is True, will return with the original column names.

        Returns:

        """
        pass
        """Calculate the weight average of this dataset.

        attr vars will be dropped in this operation
        TODO: consider how to add back in aggregated attr vars, categorical/continuous
        """

        if group_var is None:
            xr_ds_base = self._obj[self.mc_vars]
            xr_ds_base.attrs['mc_vars_attrs'] = []
            res: xr.Dataset = xr_ds_base.mc.composition_to_mass().sum(keep_attrs=True).mc.mass_to_composition()
            res.mc.rename(f'Aggregate of {self.name}')

        else:
            xr_ds_base = self._obj[self.mc_vars + [group_var]]
            res: xr.Dataset = xr_ds_base.mc.composition_to_mass().groupby(group_var).sum(
                keep_attrs=True).mc.mass_to_composition()
            res.mc.rename(f'Aggregate of {self.name} with group {group_var}')

        # If the Dataset is 0D add a placeholder index
        if len(res.dims) == 0:
            res = res.expand_dims('index')

        if as_dataframe:
            res: pd.DataFrame = self.to_dataframe(original_column_names=original_column_names,
                                                  ds=res)

        return res

    def composition_to_mass(self) -> xr.Dataset:
        """Transform composition to mass

        :return:
        """

        if self.composition_context == CompositionContext.ABSOLUTE:
            raise AssertionError('The dataset composition context is already absolute (mass units)')

        xr.set_options(keep_attrs=True)

        dsm: xr.Dataset = self._obj.copy()
        dsm[self._obj.mc_vars_chem] = dsm[self._obj.mc_vars_chem] * self._obj['mass_dry'] / 100

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
        dsc[self._obj.mc_vars_chem] = dsc[self._obj.mc_vars_chem] / self._obj['mass_dry'] * 100

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
        if original_column_names:
            df.rename(columns=self.column_map(), inplace=True)
        return df
