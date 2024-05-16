import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union, Iterable, List, Tuple, Callable

import numpy as np
import pandas as pd
import xarray as xr

from elphick.mass_composition.utils import solve_mass_moisture
from elphick.mass_composition.utils.interp import interp_monotonic
from elphick.mass_composition.utils.size import mean_size


class CompositionContext(Enum):
    ABSOLUTE = 'mass'
    RELATIVE = "percent"
    NONE = None


@xr.register_dataset_accessor("mc")
class MassCompositionAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        """MassComposition xarray Accessor

        Args:
            xarray_obj:
        """
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
            res = CompositionContext.NONE
        return res

    def rename(self, new_name: str):
        self._obj.attrs['mc_name'] = new_name

    def aggregate(self, group_var: Optional[str] = None,
                  group_bins: Optional[Union[int, Iterable]] = None,
                  as_dataframe: bool = False,
                  original_column_names: bool = False,
                  column_formats: Optional[Dict] = None) -> Union[xr.Dataset, pd.DataFrame]:
        """Calculate the weight average of this dataset.

        Args:
            group_var: Optional grouping variable
            group_bins: Optional bins to apply to the group_var
            as_dataframe: If True return a pd.DataFrame
            original_column_names: If True, and as_dataframe is True, will return with the original column names.
            column_formats: If not None, and as_dataframe is True, will format the dataframe per the dict.

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
            res: pd.DataFrame = self.to_dataframe(original_column_names=original_column_names, ds=res)
            if len(res) == 1:
                res.index = pd.Index([self.name], name='name')

        return res

    def cumulate(self, direction: str) -> xr.Dataset:
        """Cumulate along the dims

        Expected use case is only for Datasets that have been reduced to 1D.

        Args:
            direction: 'ascending'|'descending'

        Returns:

        """

        valid_dirs: List[str] = ['ascending', 'descending']
        if direction not in valid_dirs:
            raise KeyError(f'Invalid direction provided.  Valid arguments are: {valid_dirs}')

        d_dir: Dict = {'ascending': True, 'descending': False}

        if len(self._obj.dims) > 1:
            raise NotImplementedError('Datasets > 1D have not been tested.')

        index_var: str = str(list(self._obj.dims.keys())[0])
        if not isinstance(self._obj[index_var].data[0], pd.Interval):
            self._logger.warning("Unexpected use of the cumulate method on non-fractional data.  "
                                 "Consider setting the index (dim) as intervals.")

        interval_index = pd.Index(self._obj[index_var])
        if not (interval_index.is_monotonic_increasing or interval_index.is_monotonic_decreasing):
            raise ValueError('Index is not monotonically increasing or decreasing')

        in_data_ascending: bool = True
        if interval_index.is_monotonic_decreasing:
            in_data_ascending = False

        # convert to mass, then cumsum, then convert back to relative composition (grade)
        mass: xr.Dataset = self.composition_to_mass()

        mass = mass.sortby(variables=index_var, ascending=d_dir[direction])

        mass_cum: xr.Dataset = mass.cumsum(keep_attrs=True)
        # put the coords back
        mass_cum = mass_cum.assign_coords(**mass.coords)
        res: xr.Dataset = mass_cum.mc.mass_to_composition()

        # put back to original order
        res = res.sortby(variables=index_var, ascending=in_data_ascending)

        return res

    def composition_to_mass(self) -> xr.Dataset:
        """Transform composition to mass

        :return:
        """

        if self.composition_context == CompositionContext.ABSOLUTE:
            raise AssertionError('The dataset composition context is already absolute (mass units)')

        xr.set_options(keep_attrs=True)

        dsm: xr.Dataset = self._obj.copy()

        if 'H2O' in dsm.variables:
            dsm['H2O'] = self._obj['mass_wet'] - self._obj['mass_dry']

        if self.composition_context == CompositionContext.RELATIVE:
            dsm[self._obj.mc_vars_chem] = dsm[self._obj.mc_vars_chem] * self._obj['mass_dry'] / 100

            for da in dsm.values():
                if da.attrs['mc_type'] == 'chemistry':
                    da.attrs['units'] = dsm['mass_wet'].attrs['units']

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
        if 'H2O' in dsc.variables:
            dsc['H2O'] = (self._obj['mass_wet'] - self._obj['mass_dry']) / self._obj['mass_wet'] * 100

        if self.composition_context == CompositionContext.ABSOLUTE:
            dsc[self._obj.mc_vars_chem] = dsc[self._obj.mc_vars_chem] / self._obj['mass_dry'] * 100

            for da in dsc.values():
                if da.attrs['mc_type'] == 'chemistry':
                    da.attrs['units'] = '%'

        xr.set_options(keep_attrs='default')

        return dsc

    def split(self, fraction: float) -> Tuple[xr.Dataset, xr.Dataset]:
        """Split the object by mass

        A simple mass split maintaining the same composition

        Args:
            fraction: A constant in the range [0.0, 1.0]

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """

        xr.set_options(keep_attrs=True)

        if not (isinstance(fraction, float)):
            raise TypeError("The fraction provided is not a float")

        if not (fraction >= 0.0) and (fraction <= 1.0):
            raise ValueError("The fraction provided must be between [0.0, 1.0] - mass cannot be created nor destroyed.")

        out = deepcopy(self)
        comp = deepcopy(self)

        # split by mass
        out._obj[self._obj.mc_vars_mass] = out._obj[self._obj.mc_vars_mass] * fraction
        out.rename(f'({fraction} * {self.name})')

        comp._obj[self._obj.mc_vars_mass] = comp._obj[self._obj.mc_vars_mass] * (1 - fraction)
        comp.rename(f'({1 - fraction} * {self.name})')

        xr.set_options(keep_attrs='default')

        return out._obj, comp._obj

    def split_by_partition(self, partition_definition: Callable) -> Tuple[xr.Dataset, xr.Dataset]:
        """Partition the object along a given dimension.

        This method applies the defined partition resulting in two new objects.

        See also: split

        Args:
            partition_definition: A partition function that defines the efficiency of separation along a dimension

        Returns:
            tuple of two datasets, the first defined by the function, the other the complement
        """

        xr.set_options(keep_attrs=True)

        out = deepcopy(self)
        comp = deepcopy(self)

        if not isinstance(partition_definition, Callable):
            raise TypeError("The definition is not a callable function")
        if 'dim' not in partition_definition.keywords.keys():
            raise NotImplementedError("The callable function passed does not have a dim")

        dim = partition_definition.keywords['dim']
        partition_definition.keywords.pop('dim')
        if isinstance(self._obj[dim].data[0], pd.Interval):
            if dim == 'size':
                x = mean_size(pd.arrays.IntervalArray(self._obj[dim].data))
            else:
                x = pd.arrays.IntervalArray(self._obj[dim].data).mid
        else:
            # assume the index is already the mean, not a passing or retained value.
            self._logger.warning('The provided callable is being applied across a dimension that is '
                                 'not an interval. This is not typical usage.  It is assumed that the '
                                 'dimension data represents the centre/mean, and not an edge like '
                                 'retained or passing.')
        pn = partition_definition(x)
        if not ((dim in self._obj.dims) and (len(self._obj.dims) == 1)):
            # TODO: Set the dim to match the partition if it does not already
            # obj_mass = obj_mass.swap_dims(dim=)
            pass

        out._obj = out.mul(pn / 100)
        comp._obj = self.sub(out._obj)

        xr.set_options(keep_attrs='default')

        return out._obj, comp._obj

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

    def mul(self, value: Union[float, np.ndarray]) -> xr.Dataset:
        """Multiply self and retain attrs

        Multiply the mass-composition variables only by the value then append any attribute variables.
        NOTE: does not multiply two objects together.  Used for separation (partition) operations.

        Args:
            value: the multiplier, a scalr or array of floats.

        Returns:

        """

        xr.set_options(keep_attrs=True)

        # initially just add the mass and composition, not the attr vars
        xr_self: xr.Dataset = self.composition_to_mass()[self.mc_vars]
        res: xr.Dataset = xr_self * value

        res = self._math_post_process(None, res, xr_self, 'multiplied')

        # when two objects (edges) are added, their to-nodes must be set to the same value (merged)
        # then the nodes for the summed object will be from that merged node to the next, new node.
        # res.attrs['nodes'] = [xr_self.attrs['nodes'][1], xr_other.attrs['nodes'][1] + 2]

        xr.set_options(keep_attrs='default')

        return res

    def div(self, other: xr.Dataset) -> xr.Dataset:
        """Divide self by the supplied object

        Perform the division with the mass-composition variables only and then append any attribute variables.
        Args:
            other: denominator object, self will be divided by this object

        Returns:

        """

        xr.set_options(keep_attrs=True)

        # initially just divide the mass and composition, not the attr vars
        xr_self: xr.Dataset = self.composition_to_mass()[self.mc_vars]
        xr_other: xr.Dataset = other.mc.composition_to_mass()[self.mc_vars]

        res: xr.Dataset = xr_self / xr_other

        res = self._math_post_process(other, res, xr_self, 'divided')

        xr.set_options(keep_attrs='default')

        return res

    def _math_post_process(self, other, res, xr_self, operator_string: str):
        """Manages cleanup after math operations

        Math operations occur in absolute composition (mass) space, and these operations lose the attrs
        This method adds back the attrs for the Dataset, DataArrays.  It also converts the result back
        to relative compositional space.  Attr variables are also added back.
        Args:
            other: The other object taking part in the operation
            res: The result object
            xr_self: The xarray of the result object - used to recover attrs - we just use self instead?
            operator_string: string representing the operation

        Returns:

        """
        # update attrs
        res.attrs.update(self._obj.attrs)
        da: xr.DataArray
        for new_da, da in zip(res.values(), xr_self.values()):
            new_da.attrs.update(da.attrs)
        # merge in the attr vars
        res = xr.merge([res, self._obj[self._obj.mc_vars_attrs]])
        if operator_string == 'added':
            res.mc.rename(f'({self.name} + {other.mc.name})')
            res = res.mc.mass_to_composition()
        elif operator_string == 'subtracted':
            res.mc.rename(f'({self.name} - {other.mc.name})')
            res = res.mc.mass_to_composition()
        elif operator_string == 'divided':
            res.mc.rename(f'({self.name} / {other.mc.name})')
            # division returns relative, not absolute - do not convert back to relative composition
        elif operator_string == 'multiplied':
            res.mc.rename(f'({self.name} partitioned)')
            res = res.mc.mass_to_composition()
        else:
            raise NotImplementedError('Unexpected operator string')

        # protect grades from nans and infs - push them to zero
        if res.mc_vars_chem:
            res[res.mc_vars_chem] = res[res.mc_vars_chem].where(res[res.mc_vars_chem].map(np.isfinite), 0.0)

        return res

    def column_map(self):
        res: Dict = {var: da.attrs['mc_col_orig'] for var, da in self._obj.items()}
        return res

    def to_dataframe(self, original_column_names: bool = False,
                     ds: Optional[xr.Dataset] = None) -> pd.DataFrame:

        if ds is None:
            ds = self._obj.mc.data()

        df: pd.DataFrame = ds.to_dataframe()

        # order with mass, chem then attr columns last
        col_order: List[str] = self.mc_vars_mass + ['H2O'] + self.mc_vars_chem + self.mc_vars_attrs
        df = df[[col for col in col_order if col in df.columns]]

        if original_column_names:
            df.rename(columns=self.column_map(), inplace=True)
        return df

    # def resample(self, dim: str, num_intervals: int = 50, edge_precision: int = 8) -> xr.Dataset:
    #     if len(self._obj.dims) > 1:
    #         raise NotImplementedError("Not yet tested on datasets > 1D")
    #
    #     # define the new coordinates
    #     right_edges = pd.arrays.IntervalArray(self._obj[dim].data).right
    #     new_coords = np.round(np.geomspace(right_edges.min(), right_edges.max(), num_intervals), edge_precision)
    #     xr_upsampled: xr.Dataset = interp_monotonic(self._obj, coords={'size': new_coords},
    #                                                 include_original_coords=True)
    #     return xr_upsampled


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
