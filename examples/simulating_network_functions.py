"""
Network Simulation Tools
========================

This is not a runnable example, but are the functions and classes used in the simulating_networks.py example.

Including the my_simulator function in the main script can cause parallel processing issues (at least when
trying to run both as a script and an ipython/Sphinx example).

"""

from random import random
import time

from joblib import Parallel
from tqdm import tqdm

from elphick.mass_composition import MassComposition
from elphick.mass_composition.network import MCNetwork


def my_simulator(args) -> tuple[int, MCNetwork]:
    mc: MassComposition
    sid, mc = args
    mc.name = 'feed'
    fraction = random()
    time.sleep(fraction * 10)
    lump, fines = mc.split(fraction, name_1='lump', name_2='fines')
    mcn: MCNetwork = MCNetwork.from_streams(streams=[mc, lump, fines], name=f'Sample {sid}')
    return sid, mcn


class TqdmParallel(Parallel):
    def __init__(self, *args, **kwargs):
        self._tqdm = tqdm(total=kwargs['total'])
        kwargs.pop('total')
        super().__init__(*args, **kwargs)

    def __call__(self, iterable):
        iterable = list(iterable)
        self._tqdm.total = len(iterable)
        result = super().__call__(iterable)
        self._tqdm.close()
        return result

    def _print(self, msg, *msg_args):
        return

    def print_progress(self):
        self._tqdm.update()

    def _dispatch(self, batch):
        job_idx = super()._dispatch(batch)
        return job_idx

    def _collect(self, output):
        return super()._collect(output)
