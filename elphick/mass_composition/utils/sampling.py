import random
from typing import Final

MAX_NODES: Final[int] = int(1e10)


def random_int() -> int:
    # ideally sample without replacement and return a single int
    # sadly multiple calls mean it is not without replacement, managing temporarily with a large range
    return random.sample(range(MAX_NODES), 1)[0]
