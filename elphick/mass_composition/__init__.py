from __future__ import annotations

from .mass_composition import MassComposition
from .mc_xarray import MassCompositionAccessor

import importlib.metadata
__version__ = importlib.metadata.version('mass-composition')