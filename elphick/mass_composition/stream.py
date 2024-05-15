from typing import Optional, Callable

from elphick.mass_composition import MassComposition


class Stream(MassComposition):
    def __init__(self, mc: MassComposition, **kwargs):
        """
        Args:
            mc: MassComposition object to be used as a stream.
            **kwargs: The key word arguments for the Mass Composition object
        """
        # Filter out all private properties
        mc_dict = {k: v for k, v in mc.__dict__.items() if
                   not (k.startswith('_') or k in ['config', 'variables', 'status'])}

        super().__init__(**mc_dict, **kwargs)
        self.set_data(data=mc._data, constraints=mc.constraints)
        self._nodes = mc._nodes
        self.variables = mc.variables

    @property
    def source_node(self):
        return self._nodes[0]

    @property
    def destination_node(self):
        return self._nodes[1]

    @classmethod
    def from_mass_composition(cls, mc: MassComposition, **kwargs):
        return cls(mc=mc, **kwargs)

    def split(self, fraction: float,
              name_1: Optional[str] = None, name_2: Optional[str] = None) -> tuple['Stream', 'Stream']:
        """
        Splits the stream into two streams.

        Args:
            fraction: The fraction of the stream to be assigned to the first stream.
            name_1: The name of the first stream.
            name_2: The name of the second stream.

        Returns:
            A tuple of two Stream objects.
        """
        mc1, mc2 = super().split(fraction, name_1, name_2)
        return Stream.from_mass_composition(mc1), Stream.from_mass_composition(mc2)

    def apply_partition(self, definition: Callable,
                        name_1: Optional[str] = None, name_2: Optional[str] = None) -> tuple['Stream', 'Stream']:
        """
        Partition the object along a given dimension.

        This method applies the defined separation resulting in two new objects.

        See also: split

        Args:
            definition: A partition function that defines the efficiency of separation along a dimension
            name_1: The name of the reference stream created by the split
            name_2: The name of the complement stream created by the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement


        """
        mcs = super().apply_partition(definition, name_1, name_2)
        return (Stream.from_mass_composition(mc) for mc in mcs)

    def add(self, other: 'Stream', name: Optional[str] = None) -> 'Stream':
        """
        Adds the stream to another stream.

        Args:
            other: The other stream.
            name: The name of the new stream.

        Returns:
            A Stream object.
        """
        mc = super().add(other, name)
        return Stream.from_mass_composition(mc)

    def sub(self, other: 'Stream', name: Optional[str] = None) -> 'Stream':
        """Subtract two streams

        Subtracts other from self, with optional name of the returned object
        Args:
            other: stream to subtract from self
            name: name of the returned stream

        Returns:

        """
        res: MassComposition = self.__sub__(other)
        if name is not None:
            res._data.mc.rename(name)
        return Stream.from_mass_composition(res)
