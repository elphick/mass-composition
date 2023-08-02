from typing import Tuple, Optional

from elphick.mass_composition import MassComposition
from elphick.mass_composition.datasets.sample_data import sample_data
from elphick.mass_composition.utils.sampling import random_int


class Stream(MassComposition):
    def __init__(self, nodes: Optional[Tuple[int, int]] = None, **kwargs):
        """

        Args:
            nodes: (u, v) representing (from_node, to_node)
            **kwargs: The key word arguments for the Mass Composition object

            data:
            name:
            mass_wet_var:
            mass_dry_var:
            moisture_var:
            chem_vars:
            mass_units:
            constraints:
            config_file:
        """
        if nodes:
            self.nodes: Tuple[int, int] = nodes
        else:
            self.nodes: Tuple[int, int] = (random_int(), random_int())

        super(MassComposition, self).__init__(**kwargs)


if __name__ == '__main__':
    obj: Stream = Stream(kwargs={'data': sample_data()})
    print('done')
