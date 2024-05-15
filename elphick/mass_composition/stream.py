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
