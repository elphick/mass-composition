from __future__ import annotations

from .mass_composition import MassComposition
from .mc_xarray import MassCompositionAccessor

# adding this fails the pipeline unit tests
# REF: https://github.com/Nuitka/Nuitka/issues/1793
#
# import importlib.metadata
# __version__ = importlib.metadata.version('mass-composition')