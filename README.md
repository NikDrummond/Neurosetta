Neurosetta - The rosetta stone for neuron data types

Convert neurons form different sources (navis, pymaid, natverse) to unified python data type and back, based on .swc file conventions.

Allows for output to networkx directed and undirected graph objects as well.

The core class used adheres stricktly to the SWC file documentation found [here](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) for the standardised swc files, as described on the [neuromorpho](www.neuromorpho.org) site.

To DOs:

    add natverse integration
    Sort so connectors are returned as a separate table
    Graph outputs

Additional benefits - this will work for the table structure of the Skeletor package. 

Quick Install:

  `pip3 install git+git://github.com/NikDrummond/Neurosetta@master`



