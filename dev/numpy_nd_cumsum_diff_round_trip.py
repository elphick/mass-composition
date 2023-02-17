"""
Cumsum-Diff Round Trip
======================

The cumulative sum space is monotonic, and hence can be used for more robust interpolations/resampling.

However, we then need to transform back into the non-cumulative space.

cumsum in N-D is trivial - getting back to non-cum is less trivial.

This script is to work on demonstrating a generalisable N-D solution.

The test data will initially be 2D

a -> original
b -> a.cumsum
c -> b.diff

We want a == c

Greg Elphick
29/01/2023
"""

# %%

import numpy as np
from scipy.ndimage.interpolation import shift

# TODO: resolve the bug when passing a single analyte only
# %%

# create some test data

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [10, 12, 14, 16]])

print(a, '\n')

# cumulative sum
# shape stays the same

b = a.cumsum().reshape(a.shape)
print(b)

c = np.diff(b)
print(c, '\n')

# append the first col (the missing one)
first_col = a[:, 0].reshape(a.shape[0], 1)
c1 = np.hstack([first_col, c])
print(c1, '\n')

# but that solution above relies on a, we want to be self-sufficient and rely only on b
# we need to subtract the first col from the shifted last col.  The fill val is zero,
# so it delivers the first element after the subtraction.
first_col_2 = b[:, 0] - shift(b[:, -1], 1, cval=0)
first_col_2 = first_col_2.reshape(b.shape[0], 1)
c2 = np.hstack([first_col_2, c])
print(c2, '\n')

assert np.isclose(a, c2).all()

# %%
#
# So we have a solution for 2D.
# Now how do we generalise this to N-D?  Well 3D at the very least, but ND would be great.
