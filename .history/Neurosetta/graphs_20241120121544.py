import graph_tool.all as gt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform, pdist
import hdbscan
from typing import List
import vedo as vd


from .core import Tree_graph, Node_table, infer_node_types
from .sets import Sfamily_intersect, Sfamily_XOR


def g_has_property(
    N: gt.Graph | Tree_graph, g_property: str, t: str | bool = None
) -> bool:
    """
    Check if a property is within a graph. Will either check for a property generally, or can check specifically graph, vertex, or edge.

    If the g_property argument is None (default), all graph, vertex, and edge properties are checked against. If however the looked for property is a specific type, then the g_property argument can be set to either "v", "e", or "g".

    In this case only the graph ('g'), vertex ('v'), or edge ('e') property is checked.
    """
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # if t is specifically vertex or edge
    if t is not None:
        # check if vertex property
        if t == "v":
            return ("v", g_property) in g.properties
        elif t == "e":
            return ("e", g_property) in g.properties
        elif t == "g":
            return ("g", g_property) in g.properties
    else:
        return (("v", g_property) in g.properties) | (("e", g_property) in g.properties) | (("g", g_property) in g.properties)


# function to get node coordinates from a graph


def g_vert_coords(
    N: Tree_graph | gt.Graph, subset: List | bool = None
) -> np.ndarray[float]:
    """
    return spatial coordinates of vertices in an np.array
    """

    # return an np.array of coordinates from a graph

    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # check coordinates are a property
    if not g_has_property(g, "coordinates", t="v"):
        raise AttributeError("Coordinates property missing from graph")

    # if we have no subset, return all
    if subset is None:
        coords = g.vp["coordinates"].get_2d_array().T
    else:
        # if subset is just a single int, convert to a list
        if isinstance(subset, (int, np.integer)):
            subset = [subset]
        coords = g.vp["coordinates"].get_2d_array().T
        coords = coords[subset]
        
    return coords


def get_g_distances(
    N: Tree_graph | gt.Graph, bind: bool = False, name: str = "Path_length"
) -> None | gt.PropertyMap:
    """
    create edge property map of edge lengths for a graph with coordinates vertex property
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # check coordinates are a property
    if not g_has_property(g, "coordinates", t="v"):
        raise AttributeError("Coordinates property missing from graph")

    # generate distance/weight for graph
    # add edge weights based on distance
    eprop_w = g.new_ep("double")

    edges = g.get_edges()
    coords = g_vert_coords(g)
    eprop_w.a = np.linalg.norm(coords[edges[:,0]] - coords[edges[:,1]], axis = 1)
    
    # bind this as an edge property to the graph
    if bind:
        g.ep[name] = eprop_w
    else:
        return eprop_w


# functions for getting indicies/ counts of leaves/branches /root
def g_leaf_inds(N: Tree_graph | gt.Graph) -> np.ndarray[int]:
    """
    Returns a numpy array of leaf node indices
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # graph leaf inds - includes soma
    return np.where(g.degree_property_map("out").a == 0)[0]


def g_branch_inds(N: Tree_graph | gt.Graph) -> np.ndarray[int]:
    """
    Returns a numpy array of leaf node indices
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # graph leaf inds - includes soma
    return np.where(g.degree_property_map("out").a > 1)[0]


def g_lb_inds(N: Tree_graph | gt.Graph, return_types: bool = False, root:bool = False) -> np.ndarray[int]:
    """
    Returns indices of all leaf and branch nodes

    Note: only useful if you want them all together and don't care which is which
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    l_inds = g_leaf_inds(g)
    b_inds = g_branch_inds(g)

    inds = np.unique(np.concatenate([l_inds, b_inds]))
    # if root is false we are removing it
    if ~root:
        inds = inds[inds != g_root_ind(N)]
    
    return inds


def g_root_ind(N: Tree_graph | gt.Graph) -> int:
    """
    Return integer of root node index
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")
    return np.where(g.degree_property_map("in").a == 0)[0][0]


def leaf_count(N: Tree_graph | gt.Graph) -> int:
    return len(g_leaf_inds(N))


def branch_count(N: Tree_graph | gt.Graph) -> int:
    return len(g_branch_inds(N))


def segment_counts(N: Tree_graph | gt.Graph) -> int:
    return len(g_lb_inds(N))


def _edist_mat(g: gt.Graph, inds: list, flatten: bool = False) -> np.ndarray[float]:
    """
    Return Euclidean distance matrix between nodes, specified by inds
    """
    coords = np.array([g.vp["coordinates"][i] for i in inds])
    e_dist = pdist(coords)
    if not flatten:
        e_dist = squareform(e_dist)
    return e_dist


def _gdist_mat(g: gt.Graph, inds: list, flatten: bool = False) -> np.ndarray[float]:
    """
    return path length distance matrix
    """
    directed = g.is_directed()
    # if we need to make the graph undirected - remember we need to swap this back
    if directed:
        g.set_directed(False)

    # if graph doesn't already have path length property
    if not g_has_property(g, "Path_length"):
        # add edge weights based on distance
        eprop_w = g.new_ep("double")
        # get length of each edge
        eprop_w.a = [
            np.linalg.norm(g.vp["coordinates"][i[0]].a - g.vp["coordinates"][i[1]].a)
            for i in g.iter_edges()
        ]

        # generate path length distance matrix
        g_dist = np.zeros((len(inds), len(inds)))
        for i in range(len(inds)):
            g_dist[i] = gt.shortest_distance(
                g=g, source=inds[i], target=inds, weights=eprop_w
            )

    # if we do have the property
    else:
        # generate path length distance matrix
        g_dist = np.zeros((len(inds), len(inds)))
        for i in range(len(inds)):
            g_dist[i] = gt.shortest_distance(
                g=g, source=inds[i], target=inds, weights=g.ep["Path_length"]
            )
    # make symmetric - I am gratuitously assuming it isn't only because of floating point errors
    g_dist = np.tril(g_dist) + np.triu(g_dist.T, 1)

    # swap back if we changed the graph to directed
    if directed:
        g.set_directed(True)

    if flatten:
        g_dist = squareform(g_dist)

    return g_dist


def dist_mat(
    N: Tree_graph | gt.Graph,
    inds: np.array,
    method: str = "Euclidean",
    flatten: bool = False,
) -> np.ndarray[float]:
    """
    Generate pairwise distance matrix for graph
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if g_has_property(g, "coordinates", "v"):
        if method == "Euclidean":
            dist = _edist_mat(g, inds, flatten)
        elif method == "Path Length":
            dist = _gdist_mat(g, inds, flatten)
        else:
            raise AttributeError("Method not recognised")

    return dist


def HDBSCAN_g(
    N: Tree_graph | gt.Graph,
    nodes: str = "both",
    metric: str = "Path Length",
    min_cluster=20,
    output: str = "labels",
    verbose: bool = True,
):
    """
    Compute HDBSCAN clustering on a neuron
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # get node inds
    if nodes == "both":
        inds = g_lb_inds(g)
    elif nodes == "leaves":
        inds = g_leaf_inds(g)
    elif nodes == "branches":
        inds = g_branch_inds(g)

    # get coordinates
    coords = g_vert_coords(g, inds)

    # path length distance matrix
    dist = dist_mat(g, inds, method=metric)

    # run hdbscan
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="precomputed")
    hdb.fit(dist)

    if verbose:
        print("Number clusters:" + str(len(np.unique(hdb.labels_[hdb.labels_ != -1]))))
        print("Number noise points: " + str(len(hdb.labels_[hdb.labels_ == -1])))

    if output == "labels":
        return hdb.labels_
    elif output == "label_inds":
        return hdb.labels_, inds
    elif output == "all":
        return hdb, inds


def random_nodes(
    n: int = 1,
    N: Tree_graph | gt.Graph = None,
    subset: np.ndarray[int] = None,
    exclude: np.ndarray[int] | int = None,
) -> int | np.ndarray[int]:
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
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if (g is None) and (subset is None) and (exclude is None):
        raise AttributeError("No data provided to get a sample from!")
    elif (g is None) and (subset is None) and (exclude is not None):
        raise AttributeError("No data provided to get a sample from!")

    if g is not None:
        assert isinstance(g, gt.Graph)

    # figure out our nodes - either a subset, or all nodes in g

    # if a subset is provided, this takes priority
    if subset is not None:
        sample = subset
    # if just the graph was provided, get all node
    elif (g is not None) and (subset is None):
        sample = g.get_vertices()

    # figure out if we want to exclude stuff
    if exclude is not None:
        # subtract exclude from sample
        sample = np.setdiff1d(sample, exclude)
    # generate random sample!
    sample = np.random.choice(sample, size=n)

    return sample


def path_vertex_set(
    N: Tree_graph | gt.Graph,
    source: int,
    target: int | np.ndarray[int] | List,
    weight: None | str | gt.EdgePropertyMap = None,
) -> List[np.ndarray[int]]:

    # Check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # Preprocess the graph if necessary
    if isinstance(weight, str):
        if not g_has_property(g, weight, t="e"):
            print("Provided weight not a graph property, ignoring!")
            weight = None

    # Prepare the graph for shortest path computation
    g2 = g.copy()
    if g2.is_directed():
        g2.set_directed(False)

    paths = []

    # Compute shortest paths
    if isinstance(target, (int, np.integer)):
        target = [target]

    for t in target:
        if weight is None:
            path = gt.shortest_path(g2, source, t)[0]
        else:
            path = gt.shortest_path(g2, source, t, weights=g2.ep[weight])[0]
        paths.append([g.vertex_index[i] for i in path])

    return paths


def nearest_vertex(
    coords: np.ndarray | Tree_graph | gt.Graph,
    points: np.ndarray,
    return_dist: bool = False,
) -> int | tuple:
    """
    Find the index of the vertex in coords closest to the given point.

    Parameters
    ----------
    N : np.ndarray | nr.Tree_graph
        Array of vertex coordinates or neuron tree graph
    points : np.ndarray
        Coordinates of the query point.
    return_dist : bool, optional
        If True, return distance to nearest neighbor.

    Returns
    -------
    nearest_v : int
        Index of nearest vertex.
    dist : float, optional
        Distance to nearest vertex.
    """

    if (isinstance(coords, Tree_graph)) | (isinstance(coords, gt.Graph)):
        coords = g_vert_coords(coords)
    elif not isinstance(coords, np.ndarray):
        raise TypeError("coords must be a np.ndarray, Tree_graph, pr gt.Graph")

    # KDTree of all points in N
    tree1 = KDTree(coords)
    dists, nearest_v = tree1.query(points, k=1)

    if return_dist:
        return (nearest_v, dists)
    else:
        return nearest_v


def NP_segment(
    g: gt.Graph | Tree_graph | Node_table, mesh: vd.Mesh, invert: bool = False
) -> gt.Graph | Tree_graph | Node_table:
    """
    Segments a neuron to a partition of the graph where the leaves are only inside (or outside - see inverse) of the provided mesh

    NOTE Rough version!
    """

    if not isinstance(g, gt.Graph):
        raise AttributeError("Currently only supports gt.Graph as input")

    # get leaf inds
    l_inds = g_leaf_inds(g)

    # leaf coords
    l_coords = g_vert_coords(g, l_inds)

    # coords of points in the mesh
    # in_leaves = mesh.inside_points(l_coords).points()
    # if using whole brain mesh...
    in_leaves = mesh.inside_points(l_coords, invert=invert).points()

    # find the index of these points in l_coords
    rem_inds = [find_point(l_coords, i) for i in in_leaves]

    # subset of l_inds which is these points
    dend_inds = l_inds[rem_inds]

    # 1) Pick a leaf node that is NOT in the mesh (use this as anchor)

    # get a random node leaf node that isn't in our exclude set

    rand_leaf = random_nodes(subset=l_inds, exclude=dend_inds)[0]

    # 2) For each leaf IN the mesh, get path to the above node.

    # 3) Collect this family of sets

    # dend_inds are the indicies in the graph of verticies in the dendrites

    # add weight property
    get_g_distances(g, bind=True)

    # get
    paths = path_vertex_set(g, source=rand_leaf, target=dend_inds, weight="weight")

    # 4) Take their intersection
    to_keep = Sfamily_intersect(paths)

    # find the rnew root
    # 5) Subset the intersection to only branch nodes
    b_inds = g_branch_inds(g)
    b_inds = np.intersect1d(to_keep, b_inds)

    # 7) if multiple, find which is FURTHEST from anchor node = ROOT
    g2 = g.copy()
    g2.set_directed(False)

    dists = gt.shortest_distance(
        g=g2, source=rand_leaf, target=b_inds, weights=g2.ep["weight"]
    )

    root = b_inds[np.where(dists == dists.max())][0]

    # get the nodes we will keep
    # 8) Symetrical difference of fmaily of path sets, + ROOT node (the root should already be in)
    to_keep = Sfamily_XOR(paths)

    # add the root
    to_keep = np.append(to_keep, root)

    # copy the original graph
    g2 = g.copy()
    # set up a mask of the nodes we want to keep
    mask_dat = np.zeros_like(g2.get_vertices())
    mask_dat[to_keep] = 1
    dend_mask = g2.new_vp("bool")
    dend_mask.a = mask_dat

    # activate the mask and purge what isn't in it
    g2.set_vertex_filter(dend_mask)
    g2.purge_vertices()

    # convert to NR graph or Table

    # return
    return g2


def g_reachable_leaves(N: Tree_graph | gt.Graph, bind: bool = False):
    """Returns a vertex property map with the number of reachable leaf nodes from each node."""

    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    l_inds = g_leaf_inds(g)
    leaf_paths = path_vertex_set(g, source=g_root_ind(g), target=l_inds)

    vprop_rl = g.new_vp("int")
    for v in g.iter_vertices():
        vprop_rl[v] = sum(v in path for path in leaf_paths)

    if bind:
        g.vp["reachable_leaves"] = vprop_rl
    else:
        return vprop_rl


def downstream_vertices(N: Tree_graph | gt.Graph, source: int) -> np.ndarray:
    """Returns an array of unique downstream vertex indices for the given source vertex in graph g.

    Performs a depth-first search from the source vertex and collects the visited vertices.
    The vertices are returned as a unique array of vertex indices.

    Parameters
    ----------
    g : gt.Graph
        Graph to work from
    source : int
        source node id

    Returns
    -------
    np.ndarray
        Vertices downstream from source
    """

    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    return np.unique(gt.dfs_iterator(g, source, array=True),dtype = np.int64)


def edge_length(i, g, weight = 'Path_length'):
    return g.ep[weight][i]


def g_cable_length(N: Tree_graph | gt.Graph, source: int = 0) -> float:
    """
    Calculates the total cable length from the given source vertex to all
    downstream vertices in a TreeGraph or Graph.

    Parameters
    ----------
    N : TreeGraph or Graph
        The input graph
    source : int, optional
        The source vertex index, by default 0

    Returns
    -------
    float
        The total cable length downstream from the source vertex
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if not g_has_property(g, "Path_length", t="e"):
        get_g_distances(g, bind=True)
    # if we are going from the root (total cable)
    if source == 0:
        cable = sum(g.ep["Path_length"].a)
    else:
        # Get cable from sub-tree rooted at source vertex
        sub_tree = gt.dfs_iterator(g, source=source, array=True)
        if sub_tree.shape[0] == 0:
            cable = 0
        else:
            cable = np.apply_along_axis(edge_length, 1, sub_tree, g).sum()
    return cable

def path_length(N:Tree_graph | gt.Graph,source: int, target : int, weight:str = 'Path_length'):
    """
    Weighted distance between two vertices
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if not g_has_property(g,weight,'e'):
        if weight == 'Path_length':
            get_g_distances(g, bind = True)
        else:    
            raise AttributeError('Input graph has no ' + weight + ' Edge property')   
    
    dist = gt.shortest_distance(g,
                    source = source,
                    target = target,
                    weights = g.ep[weight],
                    directed = False)

    # if the length is still inf then there is no path
    if dist == np.inf:
        raise ValueError('No path exists between source and target vertex') 
    else:
        return dist
    

def root_dist(N: Tree_graph | gt.Graph, weight: str = 'Path_length', bind = True, norm = False):
    """

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        _description_
    weight : str, optional
        _description_, by default 'Path_length'
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    
    if isinstance(N, Tree_graph):
        # get edges
        edges = N.graph.get_edges()
    elif isinstance(N, gt.Graph):
        edges = N.get_edges()
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")        
    # get root
    root = g_root_ind(N)
    # initialise
    root_dist = N.graph.new_vp('double')
    # dfs
    for e in gt.dfs_iterator(N.graph):
        if e.source() == root:
            root_dist[e.target()] = N.graph.ep['Path_length'][e]
        else:    
            root_dist[e.target()] = N.graph.ep['Path_length'][e] + root_dist[e.source()]

    if norm:
        root_dist.a = root_dist.a / sum(N.graph.ep['Path_length'].a)

    if bind:
        N.graph.vp['Root_distance'] = root_dist
    else:
        return root_dist
        

def get_edge_coords(N:Tree_graph) -> tuple[np.ndarray,np.ndarray]:
    """_summary_

    Parameters
    ----------
    N : Tree_graph
        _description_

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        _description_
    """
    edges = N.graph.get_edges()
    coords = g_vert_coords(N)
    p1 = coords[edges[:,0]]
    p2 = coords[edges[:,1]]
    return p1,p2

def get_edges(N:Tree_graph, root: int | None = None, subset: str | None = None) -> np.ndarray:
    """Return array of edges within a given neuron. If subset is not None, 'Internal' or 'External' must be specified

    In such a case, either edges with a leaf node as the target are returned ('External")
    Or edges with no leaf node are returned ("Internal")

    Parameters
    ----------
    N : nr.Tree_graph
        neurosetta Tree_graph representing a neuron
    subset : str | None
        If None (default) np.ndarray of all edges is returned. If "Internal" only edges with no leaf nodes are returned.
        If 'External" only edges with a leaf node are returned 

    Returns
    -------
    np.ndarray
        array of edges within the graph, Where the first values is the source node index, and second the target node index.

    Raises
    ------
    AttributeError
        If a string is passed for subset which is not 'Internal' or 'External'
    AttributeError
        If Subset it neither None, nor a string specifying 'Internal' or 'External'
    """
    if root is None:
        if isinstance(N, Tree_graph):
            edges =  N.graph.get_edges()
        elif isinstance(N, gt.Graph):
            edges = N.get_edges()    
        if subset is None:
            return edges
        elif isinstance(subset,str):
            l_inds = g_leaf_inds(N)
            mask = np.array([1 if i in l_inds else 0 for i in edges[:,1]],dtype = bool)
            if subset is 'Internal':
                return edges[~mask]    
            elif subset is 'External':    
                return edges[mask]
            else:
                raise AttributeError('Specified subset must be Internal, or External')
        else:
            raise AttributeError('Subset must be None, to return all edges, or Internal or External')
    else:
        if isinstance(N, Tree_graph):
            edges =  gt.dfs_iterator(N.graph,root,array = True)
        elif isinstance(N, gt.Graph):
            edges = gt.dfs_iterator(N,root,array = True) 
        if subset is None:
            return edges
        # if we want a subset
        elif isinstance(subset,str):
            l_inds = g_leaf_inds(N)
            mask = np.array([1 if i in l_inds else 0 for i in edges[:,1]],dtype = bool)
            if subset is 'Internal':
                return edges[~mask]    
            elif subset is 'External':    
                return edges[mask]
            else:
                raise AttributeError('Specified subset must be Internal, or External')
        else:
            raise AttributeError('Subset must be None, to return all edges, or Internal or External')
def graph_height(N: Tree_graph,map_to:str = 'edge',bind:bool = False):
    """_summary_

    Parameters
    ----------
    N : Tree_graph
        Neuron to generate property map of tree heights from
    map_to : str, optional
        If 'vertex' will map height property to nodes within the graph, 
        If 'edge' will map property to the edges. If 'all' returns both an edge and vertex property map.   by default 'edge'
    bind : bool, optional
        Map property directly to the neuron object if True, otherwise returns a property map, by default False

    Returns
    -------
    Tree_graph | graph_tool.PropertyMap
        If bind = True, N is modified inplace. otherwise individual property maps are returned.

    Raises
    ------
    AttributeError
        In the case where map_to is not 'edge", 'vertex', or 'all'
    """
    if map_to not in ['edge','vertex','all']:
        raise AttributeError('map_to argument must be "edge", "vertex", or "all"')
    # initialise a vertex property
    h_vprop = N.graph.new_vp('int')
    # get root ind and leaf/branch inds
    root = g_root_ind(N)
    lb_inds = g_lb_inds(N)
    # this is slower than it should be, but iterate over all verts
    for v in N.graph.iter_vertices():
        # get path
        path = gt.shortest_path(N.graph,root,v)[0]
        # convert to index for next bit
        path = [N.graph.vertex_index[v] for v in path]
        # length of take intersection with leaves/branches
        h_vprop[v] = len(np.intersect1d(path,lb_inds))

    # if we want to retyrn this as an edge property    
    if map_to in  ['edge','all']:
        h_eprop = N.graph.new_ep('int',h_vprop.a[get_edges(N)[:,1]])

    if bind:
        if map_to == 'edge':
            N.graph.ep['Edge_Height'] = h_eprop
        elif map_to == 'vertex':
            N.graph.vp['Vertex_Height'] = h_vprop
        elif map_to == 'all':
            N.graph.vp['Vertex_Height'] = h_vprop   
            N.graph.ep['Edge_height'] = h_eprop
    else:                
        if map_to == 'edge':
            return h_eprop
        elif map_to == 'vertex':
            return h_vprop
        elif map_to == 'all':
            return h_vprop, h_eprop

def Euclidean_MST(coords, root = True):
    """_summary_

    Parameters
    ----------
    coords : np.ndarray
        coordinates to find the euclidean MST of. Note, if root is true we assume that the 
        first coordinate is the root!
    root : bool, optional
        If True, we assume the first coordinate is the root. otherwise returns the unrooted MST, by default True

    Returns
    -------
    gt.Graph
        returns the Minimum Spanning Tree graph of the provided point cloud
    """
    # Note, we are assuming that the first coordinate is the root (if root is true)!
    
    # generate complete graph
    g = gt.complete_graph(coords.shape[0], self_loops = False, directed = False)
    # add coordinates as vertex property
    vprop_coords = g.new_vp('vector<double>')
    vprop_coords.set_2d_array(coords.T)
    g.vp['coordinates'] = vprop_coords
    # add weights/ euclidean distance
    get_g_distances(g, bind = True)
    # Create MST edge property map (will filter after)
    if root:
        mst = gt.min_spanning_tree(g,weights = g.ep['Path_length'], root = 0)
    else:
        # calculate the MST - returned here is a mask edge property
        mst = gt.min_spanning_tree(g,weights = g.ep['Path_length'])
    # clean up!
    g.set_edge_filter(mst)
    g.purge_vertices()
    # and we are done!
    return g

def synapse_MST(N:Tree_graph, synapses:str = 'All', root: bool = True) -> Tree_graph:
    """_summary_

    Parameters
    ----------
    N : Tree_graph
        Neuron to construct Euclidean MST from
    synapses : str, optional
        Which synapses to construct MST with, can be 'Inputs", which uses just input synapses,
        'Outputs' which uses just output synapses, or 'All' which uses inputs and outputs, by default 'All'
    root : bool, optional
        if you wish to include the root node (recommended) or not, by default True

    Returns
    -------
    nr.Tree_graph
        Tree_graph of the Euclidean Spanning Tree of given subset of synapses for the given neuron.
    """
    assert isinstance(N,Tree_graph), "Input must be neurosetta.Tree_graph"

    ### Get coordinates according to include
    if synapses == 'Inputs':
        assert g_has_property(N,'inputs'), "Given neuron must have inputs property to make MST of Input Synapses"
        coords = N.graph.gp['inputs'][['graph_x','graph_y','graph_z']].values
    elif synapses == 'Outputs':
        assert g_has_property(N,'outputs'), "Given neuron must have outputs property to make MST of Output Synapses"
        coords = N.graph.gp['outputs'][['graph_x','graph_y','graph_z']].values
    elif synapses == 'All':
        assert (g_has_property(N,'inputs') and g_has_property(N,'outputs')), "Given neuron must have both outputs and inputs property to make MST of All Synapses"
        input_coords = N.graph.gp['inputs'][['graph_x','graph_y','graph_z']].values
        output_coords = N.graph.gp['outputs'][['graph_x','graph_y','graph_z']].values
        coords = np.vstack((input_coords,output_coords))
    
    # if we want to add the root
    if root:
        root_coord = g_vert_coords(N,g_root_ind(N))[0]
        coords = np.vstack((root_coord,coords))

    # make sure we have path lengths, and if not we add it
    if ~ g_has_property(N,'Path_length'):
        get_g_distances(N, bind = True)
    
    # build MST
    g = Euclidean_MST(coords,root = root)

    # pack this into a neuron object
    g = Tree_graph(name = N.name + '_MST', graph = g)

    return g
