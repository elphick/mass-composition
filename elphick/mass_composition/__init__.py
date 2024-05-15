from __future__ import annotations

from .mass_composition import MassComposition
from .stream import Stream
from .flowsheet import Flowsheet
from .mc_xarray import MassCompositionAccessor

from importlib import metadata

try:
    __version__ = metadata.version('mass-composition')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
