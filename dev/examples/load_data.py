"""
Example for loading data
"""
import pandas as pd

from elphick.mass_composition.datasets import Downloader

df: pd.DataFrame = Downloader().load_data()

print('done')