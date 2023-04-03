import random
from typing import Final

MAX_NODES: Final[int] = 100


def random_int():
    # sample without replacement and return a single int
    return random.sample(range(MAX_NODES), 1)[0]
