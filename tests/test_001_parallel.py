import time
import pytest
from elphick.mass_composition.utils.parallel import TqdmParallel

def test_init_and_call():
    # Create a TqdmParallel object
    tp = TqdmParallel(n_jobs=1, total=5, desc="Processing")

    # Define a simple function to apply to the iterable
    def multiply_by_two(n):
        time.sleep(1)
        return n * 2

    # Create an iterable
    iterable = tuple([(i, multiply_by_two, {}) for i in range(5)])

    # Process the iterable with the TqdmParallel object
    result = tp(iterable)

    # Check if the result is as expected
    assert result == [0, 2, 4, 6, 8]