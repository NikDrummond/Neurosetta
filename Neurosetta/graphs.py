import graph_tool.all as gt
import numpy as np
from scipy.spatial.distance import squareform, pdist
import hdbscan
from typing import List


from .core import Tree_graph


def g_has_property(g_property:str, g:gt.Graph, t:str | bool = None)->bool:
    """
    Check if a property is within a graph
    """

    # if t is specifically vertex or edge
    if t is not None:
        # check if vertex property
        if t == "v":
            return ("v", g_property) in g.properties
        elif t == "e":
            return ("e", g_property) in g.properties
    else:
        return (("v", g_property) in g.properties) | (("e", g_property) in g.properties)


# function to get node coordinates from a graph


def g_vert_coords(g: gt.Graph, subset: str | bool = None) -> np.ndarray[float]:
    """
    return spatial coordinates of verticies in an np.array
    """

    # return an np.array of coordinates from a graph

    # check input type
    if (isinstance(g, Tree_graph)) | (isinstance(g, gt.Graph)):
        if isinstance(g, Tree_graph):
            g = g.graph
    else:
        raise TypeError(
            "Input type not supported - must be Neurosetta.Tree_graph or graph_tool.Graph"
        )

    # check coordinates are a propety
    if not g_has_property("coordinates", g, t="v"):
        raise AttributeError("Coordinates property missing from graph")

    # if we have no subset, return all
    if subset is None:
        coords = np.array([g.vp["coordinates"][i] for i in g.get_vertices()])
    else:
        coords = np.array([g.vp["coordinates"][i] for i in subset])

    return coords


def get_g_distances(g: gt.Graph, inplace:bool =False, name:str ="weight") -> None | gt.PropertyMap:
    """
    create edge property map of edge lengths for a graph with corrdinates vertex property
    """
    # check input type
    if not (isinstance(g, gt.Graph)):
        raise TypeError(
            "Input type not supported - must be Neurosetta.Tree_graph or graph_tool.Graph"
        )

    # check coordinates are a propety
    if not g_has_property("coordinates", g, t="v"):
        raise AttributeError("Coordinates property missing from graph")

    # generate distance/weight for graph
    # add edge weights based on distance
    eprop_w = g.new_ep("double")
    # get length of each edge
    eprop_w.a = [
        np.linalg.norm(g.vp["coordinates"][i[0]].a - g.vp["coordinates"][i[1]].a)
        for i in g.iter_edges()
    ]

    if inplace:
        g.ep[name] = eprop_w
    else:
        return eprop_w


def g_leaf_inds(g: gt.Graph) -> np.ndarray[int]:
    """
    Returns a numpy array of leaf node indicies
    """
    # graph leaf inds - includes soma
    return np.where(g.degree_property_map("total").a == 1)[0]


def g_branch_inds(g: gt.Graph) -> np.ndarray[int]:
    """
    Returns a numpy array of leaf node indicies
    """
    # graph leaf inds - includes soma
    return np.where(g.degree_property_map("total").a > 2)[0]

def g_lb_inds(g: gt.Graph, return_types:bool = False) -> np.ndarray[int]:
    """
    Returns indicies of all leaf and branch nodes

    Note: only usefull if you want them all together and don't care which is which
    """
    l_inds = g_leaf_inds(g)
    b_inds = g_branch_inds(g)

    inds = np.unique(np.concatenate([l_inds,b_inds]))
    return inds


def g_root_ind(g: gt.Graph) -> int:
    """
    Return integer of root node index
    """
    return np.where(g.degree_property_map("in").a == 0)[0][0]

def _edist_mat(g:gt.Graph,inds:list, flatten:bool = False) -> np.ndarray[float]:
    """
    Return Euclidean distance matrix between nodes, specified by inds
    """
    coords = np.array([g.vp['coordinates'][i] for i in inds])
    e_dist = pdist(coords)
    if not flatten:
        e_dist = squareform(e_dist)
    return e_dist

def _gdist_mat(g:gt.Graph,inds:list,flatten:bool = False) -> np.ndarray[float]:
    """
    return path length distance matrix
    """
    directed = g.is_directed()
    # if we need to make the graph undirected - remeber we need to swap this back
    if directed:
        g.set_directed(False)

    # add edge wieghts based on distance
    eprop_w = g.new_ep('double')
    # get length of each edge
    eprop_w.a = [np.linalg.norm(g.vp['coordinates'][i[0]].a - g.vp['coordinates'][i[1]].a) for i in g.iter_edges()]

    # generate path length distance matrix
    g_dist = np.zeros((len(inds),len(inds)))
    for i in range(len(inds)):
        g_dist[i] = gt.shortest_distance(g = g, 
                                        source = inds[i], 
                                        target = inds, 
                                        weights = eprop_w)
        
    # make symetric - I am gratutiously assuming it isn't only because of floating point errors
    g_dist = np.tril(g_dist) + np.triu(g_dist.T,1)

    # swap back if we changed the graph to directed
    if directed:
        g.set_directed(True)

    if flatten:
        g_dist = squareform(g_dist)

    return g_dist

def dist_mat(g:gt.Graph,inds:np.array,method:str = 'Euclidean', flatten:bool = False) -> np.ndarray[float]:
    """
    Generate pairwise distance matrix for graph
    """
    if g_has_property('coordinates',g,'v'):
        if method == 'Euclidean':
            dist = _edist_mat(g,inds,flatten)
        elif method == 'Path Length':
            dist = _gdist_mat(g,inds,flatten)
        else:
            raise AttributeError('Method not recognised')
        
    return dist
    

def HDBSCAN_g(g:gt.Graph, nodes:str = 'both', metric:str = 'Path Length', min_cluster = 20, output:str = 'labels', verbose:bool = True):
    """
    Compute HDBSCAN clustering on a neuron
    """
    # get node inds
    if nodes == 'both':
        inds = g_lb_inds(g)
    elif nodes == 'leaves':
        inds = g_leaf_inds(g)
    elif nodes == 'branches':
        inds = g_branch_inds(g)

    # get coordinates
    coords = g_vert_coords(g,inds)

    # path length distance matrix
    dist = dist_mat(g,inds,method = metric)

    # run hdbscan
    hdb = hdbscan.HDBSCAN(min_cluster_size = min_cluster,
                metric = 'precomputed')
    hdb.fit(dist)

    if verbose:
        print('Number clusters:' + str(len(np.unique(hdb.labels_[hdb.labels_ != -1]))))
        print('Number noise points: ' + str(len(hdb.labels_[hdb.labels_ == -1])))

    if output == 'labels':
        return hdb.labels_
    elif output == 'label_inds':
        return hdb.labels_, inds
    elif output == 'all':
        return hdb, inds
    
def random_nodes(n:int = 1, g:gt.Graph = None, subset:np.ndarray[int] = None, exclude:np.ndarray[int] | int = None) -> int|np.ndarray[int]:
    """
    Return n random nodes from a graph.

    Parameters
    ----------

    n:


    g:


    subset:


    exclude:


    Returns
    -------

    random_sample:
    
    """

    if (g is None) and (subset is None) and (exclude is None):
        raise AttributeError ('No data provided to get a sample from!')
    elif (g is None) and (subset is None) and (exclude is not None):
        raise AttributeError ('No data provided to get a sample from!')

    if g is not None:
        assert isinstance(g,gt.Graph)


    # figure out our nodes - either a subset, or all ndoes in g
    
    # if a subset is provided, this takes priority
    if subset is not None: 
        sample = subset
    # if just the graph was provided, get all node
    elif (g is not None) and (subset is None):
        sample = g.get_vertices()

    # figure out if we want to exclude stuff
    if exclude is not None:
        # subtract exclude from sample
        sample = np.setdiff1d(sample,exclude)
    # generate random sample!
    sample = np.random.choice(sample, size = n)

    return sample

def path_vertex_set(g:gt.Graph,source: int, target: int | np.ndarray[int] | List, weight:None | str | gt.EdgePropertyMap = None) -> List[np.ndarray[int]]:
    """
    Given a source node, return vertex path to target. If multiple targets, return list of paths 
    """


    # if weight string is provided, try to calculate the weights
    if isinstance(weight,str) :
        # check if graph edges have this property
        if not g_has_property(weight,g,t = 'e'):
            print('Provided weight not a graph property, ignoring!')
            weight = None

        # copy graph
    g2 = g.copy()
    # if copy is directed, change that
    if g2.is_directed():
        g2.set_directed(False)

    # if only one target node            
    if isinstance(target,  np.integer):
        if weight is None:
            path = gt.shortest_path(g2, source, target)[0]
        else:    
            path = gt.shortest_path(g2, source, target, weights = g2.ep[weight])[0] 

        # convert to indicies
        path = [g.vertex_index[i] for i in path]

    elif (isinstance(target,List)) | (isinstance(target,np.ndarray)):
        if weight is None:
            path = [gt.shortest_path(g2, source, i)[0] for i in target]
        else:
            path = [gt.shortest_path(g2, source, i, weights = g2.ep[weight])[0] for i in target]
        path = [[g.vertex_index[i] for i in j] for j in path]

    return path    

    


