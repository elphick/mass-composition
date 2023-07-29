import shutil
import zipfile
from string import Template
from zipfile import ZIP_DEFLATED
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml
from pooch import file_hash
from ydata_profiling import ProfileReport

from elphick.mass_composition.utils.hash_utils import check_hash, read_hash_file, write_hash


class DatasetCurator:
    def __init__(self, dir_datasets: Path = Path('../../datasets/')):
        """Instantiate

        Args:
            dir_datasets: The directory containing the csv datasets, where each dataset file/s is contained
            within a parent folder. The directory structure expected is:

            - dir_datasets
              - datatype ['tabular', 'raster', ...]
                - dataset_name
                  - datafile.ext, ['csv', 'parquet', ...]
                  - datafile.yaml, the metadata file (flat key: value format)
        """

        self.dir_datasets: Path = dir_datasets
        self._datasets: Optional[List[Dict]] = None
        self.valid_formats: List[str] = ['.csv', '.parquet']
        self._process()
        self._create_downloader()

    def register(self) -> pd.DataFrame:
        """A tabular register of the Datasets

        Returns:

        """

        records: List[pd.Series] = []

        datasets: List[str] = [d.name for d in self.dir_datasets.iterdir() if d.is_dir()]
        for ds in datasets:
            record: Dict = {'dataset': ds}
            files: List[Path] = list((self.dir_datasets / ds).glob('*'))
            datafile: Optional[Path] = None
            for file_format in self.valid_formats:
                candidates = list((self.dir_datasets / ds).glob(f'*{file_format}'))
                if len(candidates) == 1:
                    datafile = candidates[0]
                    break
                elif len(candidates) > 1:
                    raise IndexError(
                        f"Multiple datafiles found: {candidates}.  Only a single datafile is supported.")
            record['datafile'] = str(datafile)
            record['bytes'] = datafile.stat().st_size
            record['metadata'] = datafile.with_suffix('.yaml') in files
            record['report'] = datafile.with_suffix('.html') in files
            record['archive'] = datafile.with_suffix('.zip') in files
            record['datafile_md5'] = read_hash_file(datafile.with_suffix('.md5'))
            record['target_filepath'] = datafile.with_suffix('.zip') if record['archive'] else record['datafile']
            record['target'] = Path(record['target_filepath']).name
            record['target_sha256'] = file_hash(record['target_filepath'])
            records.append(pd.Series(record))

        df_register: pd.DataFrame = pd.concat(records, axis=1).T
        return df_register

    def _process(self):
        """Process the datasets

        The dataset directory is walked and each csv dataset is processed:
        1) Profiling - The profile report is stored with the same name as the dataset, but with a html suffix.
        2) The file is zipped.

        Artefacts from the process method are stored alongside the source file.

        :return:
        """

        data_files: List[Path] = list(self.dir_datasets.rglob('**/*.csv'))
        table_content: List = []

        for f in data_files:
            with open(f.with_suffix('.yaml'), 'r') as file:
                meta: Dict = yaml.load(file, Loader=yaml.SafeLoader)

            meta['maintainer'] = 'Greg Elphick'

            if not check_hash(f):  # don't re-profile if the file is unchanged
                df: pd.DataFrame = pd.read_csv(f)
                report = ProfileReport(df, title=meta['title'], dataset=meta, minimal=True,
                                       sample=dict(name="Top 10 samples", caption='', data=df.head(10)))
                report.to_file(f.with_suffix('.html'))
                write_hash(f)

            files_to_zip: List[Path] = [fn for fn in Path(f.parent).glob('*.*') if fn.suffix != '.zip']

            with zipfile.ZipFile(f.with_suffix('.zip'), mode='w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
                for filename in files_to_zip:
                    archive.write(filename, arcname=filename.name)

            for file_to_copy in [f.with_suffix('.html'), f.with_suffix('.zip')]:
                shutil.copy(file_to_copy, Path(f'../../docs/source/_static/'))

            rpt_filepath: str = (Path('../_static/') / Path(meta['filename']).with_suffix('.html').name).as_posix()
            meta['link'] = f"`link <{rpt_filepath}>`_"
            download_filepath: str = (Path('../_static/') / Path(meta['filename']).with_suffix('.zip').name).as_posix()
            meta['download'] = f":download:`zip <{download_filepath}>`"
            table_content.append(meta)

        with open('table_content.yaml', 'w') as fw:
            yaml.dump({'datasets': table_content}, fw)
        shutil.copy('table_content.yaml', Path(f'../../docs/source/datasets/'))
        self._datasets = [str(Path(ds['filename']).stem) for ds in table_content]

    def _create_downloader(self):

        self.register().to_csv('register.csv')
        self.register().to_csv('../../elphick/mass_composition/datasets/register.csv')

        with open('load_method.py.template', 'r') as tf:
            src: Template = Template(tf.read())

        output: str = "from elphick.mass_composition.datasets import Downloader\n" \
                      "import pandas as pd\n\n"
        for i, r in self.register().iterrows():
            output += src.substitute({'dataset': str(r.dataset).lower(),
                                      'datafile': str(Path(r.target).name)})

        with open(Path(__file__).parent / '../../elphick/mass_composition/datasets/datasets.py', 'w') as f:
            f.writelines(output)


if __name__ == '__main__':
    obj: DatasetCurator = DatasetCurator()
    df = obj.register()
    print('done')
