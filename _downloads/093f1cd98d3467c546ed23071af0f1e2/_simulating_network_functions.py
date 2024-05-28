"""
Network Simulation Tools
========================

This is not a runnable example, but are the functions and classes used in the simulating_networks.py example.

Including the my_simulator function in the main script can cause parallel processing issues (at least when
trying to run both as a script and an ipython/Sphinx example).

"""

import time
from random import random

from elphick.mass_composition import MassComposition
from elphick.mass_composition.flowsheet import Flowsheet


def my_simulator(args) -> tuple[int, Flowsheet]:
    mc: MassComposition
    sid, mc = args
    mc.name = 'feed'
    fraction = random()
    time.sleep(fraction * 10)
    lump, fines = mc.split(fraction, name_1='lump', name_2='fines')
    fs: Flowsheet = Flowsheet.from_streams(streams=[mc, lump, fines], name=f'Sample {sid}')
    return sid, fs



