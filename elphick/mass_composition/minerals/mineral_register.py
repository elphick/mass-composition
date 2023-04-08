from pathlib import Path

import pandas as pd
from periodictable.formulas import formula, Formula


class Minerals:
    """Minerals

    REF: https://rruff.info/ima/
    """

    def __init__(self, filepath: Path = Path('RRUFF_Export_20230407_183346.csv')):
        self.filepath: Path = filepath
        register: pd.DataFrame = pd.read_csv(filepath)
        register['formula'] = register['IMA Chemistry (concise)'].apply(lambda x: x.replace('_', ''))
        register.columns = [col.lower().replace(" ", "_").replace("(", "").replace(")", "") for col in register.columns]
        self.data: pd.DataFrame = register

    def get_mineral(self, mineral_name: str) -> Formula:
        record: pd.Series = self.data.query("mineral_name_plain==@mineral_name").iloc[0]
        obj_formula: Formula = formula(record['formula'], name=record['mineral_name_plain'])
        return obj_formula


if __name__ == '__main__':

    water = formula('H2O')

    obj: Minerals = Minerals()
    print(obj.data.head())

    hematite: Formula = obj.get_mineral('Hematite')
    kaolinite: Formula = obj.get_mineral('Kaolinite')
    quartz: Formula = obj.get_mineral('Quartz')


    url_mindat: str = r"https://www.mindat.org/search.php?name=%20Hematite"

    print('done')
