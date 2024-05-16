from typing import Optional, Callable, Generator

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

    def split_by_partition(self, partition_definition: Callable,
                           name_1: Optional[str] = None, name_2: Optional[str] = None) -> Generator[
        'Stream', None, None]:
        """
        Partition the object along a given dimension.

        This method applies the defined separation resulting in two new objects.

        See also: split, split_by_function

        Args:
            partition_definition: A partition function that defines the efficiency of separation along a dimension
            name_1: The name of the reference stream created by the split
            name_2: The name of the complement stream created by the split

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement


        """
        streams = super().split_by_partition(partition_definition, name_1, name_2)
        return (Stream.from_mass_composition(stream) for stream in streams)

    def split_by_function(self, split_function: Callable,
                          name_1: Optional[str] = None,
                          name_2: Optional[str] = None) -> Generator['Stream', None, None]:
        """Split an object using a function.

        This method applies the function to self, resulting in two new objects. The object returned with name_1
        is the result of the function.  The object returned with name_2 is the complement.

        See also: split, split_by_estimator, split_by_partition

        Args:
            split_function: Any function that transforms the dataframe from a MassComposition object into a new
             dataframe with values representing a new (output) stream.  The returned dataframe structure must be
             identical to the input dataframe.
            name_1: The name of the stream created by the function
            name_2: The name of the complement stream created by the split, which is calculated automatically.

        Returns:
            A generator of two Streams,


        """
        streams = super().split_by_function(split_function, name_1, name_2)
        return (Stream.from_mass_composition(stream) for stream in streams)

    def split_by_estimator(self, estimator: 'sklearn.base.BaseEstimator',
                           name_1: Optional[str] = None,
                           name_2: Optional[str] = None) -> Generator['Stream', None, None]:
        """Split an object using a sklearn estimator.

        This method applies the function to self, resulting in two new objects. The object returned with name_1
        is the result of the estimator.predict() method.  The object returned with name_2 is the complement.

        See also: split, split_by_function, split_by_partition

        Args:
            estimator: Any sklearn estimator that transforms the dataframe from a MassComposition object into a new
             dataframe with values representing a new (output) stream using the predict method.  The returned
             dataframe structure must be identical to the input dataframe.
            name_1: The name of the stream created by the estimator.
            name_2: The name of the complement stream created by the split, which is calculated automatically.

        Returns:
            tuple of two datasets, the first with the mass fraction specified, the other the complement
        """
        streams = super().split_by_estimator(estimator, name_1, name_2)
        return (Stream.from_mass_composition(stream) for stream in streams)

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
