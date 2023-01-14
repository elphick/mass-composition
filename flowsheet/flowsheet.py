import networkx as nx


class Flowsheet(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    @classmethod
    def from_streams(cls) -> 'Flowsheet':
        pass


if __name__ == '__main__':
    obj_fs: Flowsheet = Flowsheet.from_streams()
