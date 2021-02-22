from Neurosetta.inputs import swc_input
from Neurosetta.outputs import navis_output, graph_output

class rosettaNEURON:

    """ general core class used to move between neuron types """

    def __init__(self,neuron):
        if isinstance(neuron,str):
            self.swcTable = swc_input(neuron)

    def to_navis(self):
        navis_n = navis_output(self)
        return navis_n

    def to_graph(self,directed):
        graph_n = graph_output(self,directed)
        return graph_n