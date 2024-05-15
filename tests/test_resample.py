import numpy as np
import pandas as pd
import xarray as xr

# noinspection PyUnresolvedReferences
from .fixtures import demo_data, size_assay_data
from elphick.mass_composition import MassComposition


# TODO: reinstate tests after replacing resample with resample_1d

# def test_resample_head(size_assay_data):
#     # compare the aggregations (head) of the original and up-sampled samples
#     df_data: pd.DataFrame = size_assay_data
#     mc_size: MassComposition = MassComposition(df_data, name='size sample')
#     mc_upsampled: MassComposition = mc_size.resample(dim='size')
#
#     pd.testing.assert_frame_equal(mc_size.aggregate(), mc_upsampled.aggregate())
#
#
# def test_resample_fraction_validation(size_assay_data):
#     # aggregate the up-sampled fractions according to the original fractions.
#     # then test that aggregated up-sampled fractions align with the fractions of the original object
#
#     df_data: pd.DataFrame = size_assay_data
#     mc_size: MassComposition = MassComposition(df_data, name='size sample')
#
#     mc_upsampled: MassComposition = mc_size.resample(dim='size')
#     xr_upsampled = mc_upsampled.data
#
#     bins = [0] + list(pd.arrays.IntervalArray(mc_size.data['size'].data[::-1]).right)
#     original_sizes: pd.Series = pd.cut(
#         pd.Series(pd.arrays.IntervalArray(mc_upsampled.data['size'].data).mid, name='original_size'),
#         bins=bins, right=False, precision=8)
#     original_sizes.index = pd.arrays.IntervalArray(xr_upsampled['size'].data, closed='left')
#     original_sizes.index.name = 'size'
#     xr_upsampled = xr.merge([xr_upsampled, original_sizes.to_xarray()])
#     mc_upsampled2: MassComposition = MassComposition(xr_upsampled.to_dataframe(), name='Upsampled Sample')
#
#     df_check: pd.DataFrame = mc_upsampled2.aggregate(group_var='original_size').sort_index(ascending=False)
#     df_check.index = pd.IntervalIndex(df_check.index)
#     df_check.index.name = 'size'
#
#     pd.testing.assert_frame_equal(df_check, mc_size.data.to_dataframe())
#

def test_fractal():
    """Kudos Ben Chi - Fractal GeoAnalytics"""

    axis = 0

    for i in ((2, 3, 2, 2), (4, 3, 2), (8, 3), (1, 24), (24)):
        A = np.arange(0, 24).reshape(i)
        pre_0 = list(A.shape)
        pre_0[axis] = 1
        B = np.cumsum(A, 0)
        C = np.diff(B, axis=axis, prepend=np.zeros(pre_0))

        assert np.isclose(A, C).all()

        print('Success with \n')
        print(A.shape)
