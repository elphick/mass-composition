import numpy as np


def rosin_rammler(d: np.ndarray[float] = np.array([53.8, 38.1, 26.7, 18.8, 13.3]),
                  a: float = 10,
                  b: float = 1.42):
    """The Rosin-Rammler equation

    The Rosin-Rammler equation used to determine the size distribution is as follows:

    mp = 100 * (1 - exp * (-(d/a)**b))

    where:
    mp = mass fraction passing (%)
    d = particle diameter
    a = size at which (100/e) = 36.8% of particles retained
    b = constant = slope of the plot of ln(100/wp) vs ln x

    REF: https://help.syscad.net/Size_Distribution_Definition#Rosin-Rammler

    Returns:

    """

    mp = 100 * (1 - np.exp(-(d / a) ** b))
    return mp


def modified_rosin_rammler(d: np.ndarray[float] = np.array([53.8, 38.1, 26.7, 18.8, 13.3]),
                           d50: float = 10,
                           m: float = 1.42):
    """The Modified Rosin-Rammler equation

    The Rosin-Rammler equation used to determine the size distribution is as follows:

    mp = 100 * (1 - exp * (--0.693147(d/d50)**m))

    where:
    mp = mass fraction passing (%)
    d = particle diameter
    d50 = size at which 50% of particles retained by mass
    m = sharpness constant

    REF: https://help.syscad.net/Size_Distribution_Definition#Rosin-Rammler

    Returns:

    """

    mp = 100 * (1 - np.exp(-0.693147 * (d / d50) ** m))
    return mp


def gaudin_schuhmann(d: np.ndarray[float] = np.array([53.8, 38.1, 26.7, 18.8, 13.3]),
                     k: float = 350,
                     m: float = 0.71):
    """The Gaudin-Schuhmann equation

    The Gaudin-Schuhmann equation used to determine the size distribution is as follows:

    mp = 100 * (d/k)**m

    where:
    mp = mass fraction passing (%)
    k = size modulus - size when Wp = 100
    m = distribution modulus = slope of the log-log plot Wp vs x

    REF: https://help.syscad.net/Size_Distribution_Definition#Rosin-Rammler

    Returns:

    """

    mp = 100 * (d / k) ** m
    return mp


def lynch(d: np.ndarray[float] = np.array([53.8, 38.1, 26.7, 18.8, 13.3]),
          d50: float = 10.0,
          m: float = 1.42):
    """The Lynch equation

    The Lynch equation used to determine the size distribution is as follows:

    mp = 100 - (100 * (d/k)**m)

    where:
    mp = mass fraction passing (%)
    d50 = size at which 50% of particles retained by mass
    m = sharpness constant

    REF: https://help.syscad.net/Size_Distribution_Definition#Rosin-Rammler

    Returns:

    """

    alpha = 1.54 * m - 0.47
    mp = 100 - (100 * ((np.exp(alpha) - 1) / (np.exp(alpha * d / d50) + np.exp(alpha) - 2)))
    return mp
