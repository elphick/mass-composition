import pytest

from elphick.mass_composition.mass_composition import read_yaml


def test_config():
    read_yaml('config/test_mc_config.yml')


def test_not_config():
    with pytest.raises(KeyError):
        read_yaml('config/not_mc_config.yml')
