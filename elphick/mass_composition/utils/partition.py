import numpy as np
import pandas as pd


def perfect(x: np.ndarray, d50: float) -> np.ndarray:
    """A perfect partition
    
    Args:
        x: The input dimension, e.g. size or density
        d50: The cut-point

    Returns:

    """
    pn: np.ndarray = np.where(x >= d50, 100.0, 0.0)
    return pn


if __name__ == '__main__':
    da = np.arange(0, 10)
    PN = perfect(da, d50=6.3)
    df = pd.DataFrame([da, PN], index=['da', 'pn']).T
    print(df)
