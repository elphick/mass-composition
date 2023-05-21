import numpy as np
from pandas.arrays import IntervalArray


def mean_size(size_intervals: IntervalArray) -> np.ndarray:
    """Geometric mean size

    Size calculations are performed using the geometric mean, not the arithmetic mean

    NOTE: If geometric mean is used for the pan fraction (0.0mm retained) it will return zero, which is an
    edge size not mean size.  So the mean ratio of the geometric mean to the arithmetic mean for all other
    fractions is used for the bottom fraction.


    Args:
        size_intervals: A pandas IntervalArray

    Returns:

    """

    intervals = size_intervals.copy()
    res = np.array((intervals.left * intervals.right) ** 0.5)

    geomean_mean_ratio: float = float(np.mean((res[0:-1] / intervals.mid[0:-1])))

    if np.isclose(size_intervals.min().left, 0.0):
        res[-1] = size_intervals.min().mid * geomean_mean_ratio

    return res
