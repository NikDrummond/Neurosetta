import numpy as np
import graph_tool.all as gt
import vaex as vx 

# Main core class

class Stone(object):
    """
    Core class
    """
    
    def __init__(self, name, id, units):
        self.name = name
        self.id = id
        self.units = units
# Child Graph class

class Graph(Stone):
    """
    Core graph class
    """
    def __init__(self,name,id,dimensions):
        super().__init__(name,id,dimensions)


# Child Image Class

class Image(Stone):
    """
    Core Image object
    """
    
    def __init__(self, name, id, dimensions):
        super().__init__(name,id)
        self.dimensions = dimensions

# Grand-child graph classes

class neu_Tree(Graph):
    """
    Tree graph representation of a neuron based on graph_tool.Graph object
    """

    def __init__(self,name,id,dimensions,graph):
        super().__init__(name,id, dimensions)
        self.graph = graph

class neu_Mesh(Graph):
    """
    Mesh neuron representation
    """

    def __init__(self,name,id,dimensions,mesh):
        super().__init__(name,id,dimensions)
        self.mesh = mesh

# Grand-child image classes

class neu_denseImage(Image):
    """
    Dense image representation
    """

    def __init__(self,name,id,image, dimensions):
        super().__init__(name,id, dimensions)
        self.image = image

class neu_sparseImage(Image):
    """
    Spare image representation
    """

    def __init__(self,name,id,image, dimensions): 
        super().__init__(name,id,dimensions)
        self.image = image