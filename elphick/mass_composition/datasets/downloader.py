import webbrowser
from pathlib import Path
from typing import Dict

import pandas as pd
import pooch
from pooch import Unzip, Pooch


class Downloader:
    def __init__(self):
        """Instantiate a Downloader
        """

        self.register: pd.DataFrame = pd.read_csv(Path(__file__).parent / 'register.csv', index_col=False)

        self.dataset_hashes: Dict = self._create_register_dict()

        self.downloader: Pooch = pooch.create(path=pooch.os_cache("elphick/mass_composition"),
                                              base_url="https://github.com/elphick/mass_composition/raw/main/docs"
                                                       "/source/_static/",
                                              version=None,
                                              version_dev=None,
                                              registry={**self.dataset_hashes})

    def load_data(self, datafile: str = '231575341_size_by_assay.zip', show_report: bool = False) -> pd.DataFrame:
        """
        Load the 231575341_size_by_assay data as a pandas.DataFrame.
        """
        fnames = self.downloader.fetch(datafile, processor=Unzip())
        if show_report:
            webbrowser.open(str(Path(fnames[0]).with_suffix('.html')))
        data = pd.read_csv(Path(fnames[0]).with_suffix('.csv'))
        return data

    def _create_register_dict(self) -> Dict:
        df_reg: pd.DataFrame = self.register[['target', 'target_sha256']]
        df_reg.loc[:, 'target'] = df_reg['target'].apply(lambda x: Path(x).name)
        return df_reg.set_index('target').to_dict()['target_sha256']
