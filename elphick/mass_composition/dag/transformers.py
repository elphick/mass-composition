"""
The fundamental object of interest is a MassComposition (mc) object.
The mc object is represented as an edge in the graph.
Since a DAG is defined by steps (nodes) not edges, this translates to methods of the mc object

Design decision - define steps a.k.a. transformers that apply the operations of the mc object

By doing this we can mimic the pattern implemented by the skdag package.

"""
from abc import ABC

from sklearn.base import BaseEstimator, TransformerMixin

from elphick.mass_composition import MassComposition
from sklearn.utils.metaestimators import _BaseComposition, available_if


class MCTransformer(TransformerMixin, _BaseComposition):
    def __init__(self):
        pass


class Feed(MCTransformer):
    def __init__(self):
        super().__init__()
        self.mc_: MassComposition

    def fit(self, x: MassComposition, y: [None, MassComposition]):
        self.mc_ = x

    def transform(self, x: [None, MassComposition] = None) -> MassComposition:
        return self.mc_


class Split(MCTransformer):
    def __init__(self, *, fraction: float = 0.5, name_1: str = None, name_2: str = None):
        super().__init__()
        self.fraction = fraction
        self.name_1 = name_1
        self.name_2 = name_2

        self.mc_: MassComposition

    def fit(self, x: MassComposition, y: [None, MassComposition]):
        self.mc_ = x.split(y, name_1=self.name_1, name_2=self.name_2)

    def transform(self, x: [None, MassComposition] = None) -> MassComposition:
        return self.mc_


class Combine(MCTransformer):
    def __init__(self, *, name: str = None):
        super().__init__()
        self.name = name
        self.mc_: MassComposition

    def fit(self, x: MassComposition, y: [None, MassComposition]):
        self.mc_ = x.add(y, name=self.name)

    def transform(self, x: [None, MassComposition] = None) -> MassComposition:
        return self.mc_
