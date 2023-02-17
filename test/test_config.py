import pytest

from elphick.mass_composition.mass_composition import read_yaml


def test_config():
    read_yaml('../elphick/mass_composition\config\mc_config.yaml')


def test_not_config():
    with pytest.raises(KeyError):
        read_yaml('not_mc_config.yaml')
