import navis

def navis_output(rosetta):
    """Generate SWC table for given neuron.
    Parameters
    ----------
    rosetta :     neuron in format of 'rosettaNEURON' Class
    Returns
    -------
    n :           navis.TreeNeuron format
    """

    nodes = rosetta.swcTable.copy()
    nodes.columns = ['treenode_id',
                    'label',
                    'x',
                    'y',
                    'z',
                    'radius',
                    'parent_id']
    neuron = navis.TreeNeuron(nodes)


    return neuron

def nxgraph_output(rosetta, directed = False):
    """ Generate an network x graph (undirected).
    Parameters
    ----------
    rosetta :     neuron in format of 'rosettaNEURON' Class

    directed:     bool
                  if False(default) produces an undirected graph, if True, produces a directed graph

    Returns
    -------
    n :           networkx.Graph | networkx.Digraph
    """

    df = rosetta.swcTable.copy()
    df.set_index('sample_number',inplace = True)
    nodes = [(df['sample_number'][i],
                {'coords':df.loc[i,['x','y','z']].values})
                for i in range(len(df['sample_number']))]
    
    edges = df[df.parent_sample >=0][['parent_sample','sample_number']].values
    weights = np.sqrt(np.sum((df.loc[edges[:, 0], ['x', 'y', 'z']].values.astype(float)
                                  - df.loc[edges[:, 1], ['x', 'y', 'z']].values.astype(float)) ** 2, axis=1))
    # turn into dict
    edge_dict = np.array([{'weight': w} for w in weights])
    # Add weights to dictionary
    edges = np.append(edges, edge_dict.reshape(len(edges), 1), axis=1)
    
    edges = list(map(tuple,edges))

        if directed == False:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
    # need to flip the edge list here    
    elif directed == True:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

    return G


