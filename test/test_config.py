import pytest
from pathlib import Path
from elphick.mass_composition.mass_composition import read_yaml
import os


# from testfixtures import TempDirectory
# d = TempDirectory('config')

@pytest.fixture(scope="module")
def script_loc(request):
    """Return the directory of the currently running test script"""

    # uses .join instead of .dirname, so we get a LocalPath object instead of
    # a string. LocalPath.join calls normpath for us when joining the path
    return Path(request.fspath.join('..'))


def test_config(script_loc):
    read_yaml(script_loc / 'config/test_mc_config.yml')


def test_not_config(script_loc):
    with pytest.raises(KeyError):
        read_yaml(script_loc / 'config/not_mc_config.yml')
