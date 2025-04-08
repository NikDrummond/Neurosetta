import graph_tool.all as gt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform, pdist
import hdbscan
from typing import List, Any, Literal, Tuple
import vedo as vd
import itertools
import GeoJax
import vg


from numpy import floating


from .core import Tree_graph, Node_table, infer_node_types, g_has_property
from .sets import Sfamily_intersect, Sfamily_XOR



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


def g_root_ind(N: Tree_graph | gt.Graph, all_roots = False) -> int:
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
    
    if all_roots:
        return np.where(g.degree_property_map('in').a == 0)[0]
    else:
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
    inds: np.ndarray | None = None,
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
    
    if inds is None:
        inds = g.get_vertices()

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

    ### sort out g
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError('N must be neurosetta.Tree_graph or gt.Graph')
    
    ### Get edges
    if root is None:
        edges = g.get_edges()
    else:
        edges = gt.dfs_iterator(g, root, array = True)

    ### Subset if needed
    expected_subsets = ['None','Internal','External']
    if subset == None:
        return edges
    elif subset == 'Internal':
        l_inds = g_leaf_inds(N)
        return edges[~np.isin(edges[:,1],l_inds)]   
    elif subset == 'External':
        l_inds = g_leaf_inds(N)
        return edges[np.isin(edges[:,1],l_inds)]
    else:
        raise ValueError(f'Given Subset {subset} is not valid, expected one of {expected_subsets}')

def count_edges(N: Tree_graph, root: int | None = None, subset: str | None = None) -> int:
    """Return count of the number of edges in N

    Parameters
    ----------
    N : Tree_graph
        Tree_graph representation of Neuron
    root : int | None, optional
        If provided, will return number of edges downstream from given root, by default None
    subset : str | None, optional
        If Internal, will provide count for only internal edges, if External, will provide count for only external edges, by default None

    Returns
    -------
    int
        _description_
    """
    return get_edges(N, root, subset).shape[0]

def graph_height(N: Tree_graph, map_to:str = 'edge',bind:bool = False):
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

    # if we want to return this as an edge property    
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
    """generate the Euclidean minimum spanning tree from a set of coordinates

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
    # make a copy of g and make it directed - there may be a better way to do this?
    edges = gt.dfs_iterator(g,0,array = True)
    g2 = gt.Graph(edges, hashed = True, hash_type = 'int')
    # get coordinates
    coords = np.array([g.vp["coordinates"][i] for i in g2.vp["ids"].a])
    vprop_coords = g2.new_vp("vector<double>")
    vprop_coords.set_2d_array(coords.T)
    g2.vp['coordinates'] = vprop_coords
    # re-add weights/ euclidean distance
    get_g_distances(g2, bind = True)
    
    return g2

def Random_ST(coords, root = True):
    """Generate a random spanning tree from a set of coordinates

    Parameters
    ----------
    coords : np.ndarray
        coordinates to find the Random spanning tree of. Note, if root is true we assume that the 
        first coordinate is the root!
    root : bool, optional
        If True, we assume the first coordinate is the root. otherwise returns the unrooted RST, by default True

    Returns
    -------
    gt.Graph
        returns a random spanning tree graph of the provided point cloud
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
        rst = gt.random_spanning_tree(g,weights = g.ep['Path_length'], root = 0)
    else:
        # calculate the MST - returned here is a mask edge property
        rst = gt.random_spanning_tree(g,weights = g.ep['Path_length'])
    # clean up!
    g.set_edge_filter(rst)
    g.purge_vertices()
    # make a copy of g and make it directed - there may be a better way to do this?
    edges = gt.dfs_iterator(g,0,array = True)
    g2 = gt.Graph(edges, hashed = True, hash_type = 'int')
    # get coordinates
    coords = np.array([g.vp["coordinates"][i] for i in g2.vp["ids"].a])
    vprop_coords = g2.new_vp("vector<double>")
    vprop_coords.set_2d_array(coords.T)
    g2.vp['coordinates'] = vprop_coords
    # re-add weights/ euclidean distance
    get_g_distances(g2, bind = True)
    
    return g2

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

def bf_MST(coords, root = True, bf = 0.2):

    # we use this class during the BFS call later, keep it internal to the function
    class root_dist_visitor(gt.BFSVisitor):

        def __init__(self, v_dist,g):
            self.v_dist = v_dist
            self.g = g

        def tree_edge(self, e):
            self.v_dist[e.target()] = self.v_dist[e.source()] + self.g.ep['Path_length'][e]
            
    # generate complete graph
    g = gt.complete_graph(coords.shape[0], self_loops = False,directed = False)
    # add coordinates as vertex property
    vprop_coords = g.new_vp('vector<double>')
    vprop_coords.set_2d_array(coords.T)
    g.vp['coordinates'] = vprop_coords
    # add pairwise euclidean distances
    get_g_distances(g, bind = True)

    # initialise vertex property map
    root_dist_vp = g.new_vp('double')
    # lets do this long hand before doing the vistor thing
    gt.bfs_search(g,0,root_dist_visitor(root_dist_vp,g))
    # convert out vertex property map to and edge property map
    root_dist_ep = g.new_ep('double',root_dist_vp.a[g.get_edges()[:,1]])
    # edge weights
    e_weights = g.ep['Path_length'].a
    # root distances
    r_weights = root_dist_ep.a

    # create new weights
    weights = e_weights + (bf * r_weights)

    eprop_weights = g.new_ep('double', weights)
    # mst
    mst = gt.min_spanning_tree(g,weights = eprop_weights, root = 0)
    # apply to graph
    g.set_edge_filter(mst)
    # g.purge_vertices(in_place = False)
    g.purge_edges()
    # g.reindex_edges()

    edges = gt.dfs_iterator(g,0,array = True)
    g2 = gt.Graph(edges, hashed = True, hash_type = 'int')
    # get coordinates
    coords = np.array([g.vp["coordinates"][i] for i in g2.vp["ids"].a])
    vprop_coords = g2.new_vp("vector<double>")
    vprop_coords.set_2d_array(coords.T)
    g2.vp['coordinates'] = vprop_coords

    return g2

### Node / Neuron Asymmetry

def _node_asymmetry(N: Tree_graph | gt.Graph,v: int, L: List | np.ndarray | None = None, weight: bool = True) -> Any | floating[Any] | Literal[1]:
    """
    Calculate weighted or unweighted asymmetry for a given vertex

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        Graph representation of neuron
    v : int
        specific vertex index
    L : List | np.ndarray | None, optional
        List or array of leaf nodes, by default None
    weight : bool, optional
        Weight asymmetries by proportion of tree graph downstream, by default True

    Returns
    -------
    Any | floating[Any] | Literal[1]
        Weighted or unweighted asymmetry for vertex v
    """
    # check input
    if isinstance(N,Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N

    # check we have the reachable leaves property    
    if not g_has_property(g, 'reachable_leaves'):
        g_reachable_leaves(g,bind = True)
        
    # get number of leaves if weight is True and L not provided
    if weight == True:
        if L is None:
            L = len(g_leaf_inds(g))
    # get out neighbours
    neig = g.get_out_neighbours(v)
    # if we are at a transitory node or leaf, set to 1
    if len(neig) < 2:
        sym = 1
    # if we have two children    
    elif len(neig) == 2:    
        r = g.vp['reachable_leaves'][neig[0]]
        s = g.vp['reachable_leaves'][neig[1]]
        if weight:
            w = (r + s) / L
            sym = w * ((abs(r - s)) / (r + s - 1))
        else:
            sym = (abs(r - s)) / (r + s - 1) 
    # if we have more than two children, take the mean
    else:             
        vals = []
        for pair in itertools.combinations(neig,2):
            r = g.vp['reachable_leaves'][pair[0]]
            s = g.vp['reachable_leaves'][pair[1]]
            if weight:
                w = (r + s) / L
                sym = w * ((abs(r - s)) / (r + s - 1))
            else:
                sym = (abs(r - s)) / (r + s - 1) 
            vals.append(sym)    
        sym = np.mean(vals)  

    return sym


def node_asymmetry(N: Tree_graph | gt.Graph, L: List | np.ndarray | None = None, weight = True, bind = True) -> gt.VertexPropertyMap | None:
    """_summary_

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        Graph representation of neuron
    L : List | np.ndarray | None, optional
        List or array of leaf nodes, by default None
    weight : bool, optional
        Weight asymmetries by proportion of tree graph downstream, by default True
    bind : bool, optional
        If true, bind vertex property map to neuron, otherwise return property map, by default True

    Returns
    -------
    gt.VertexPropertyMap | None
        If bind is False, returns vertex property map for given neuron. If bind is True, and weighted is True, 'Weighted_asymmetry' vertex property map is added to N,
        if weight is False, 'Asymmetry' vertex property map is bound to N
    """
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    # initialise vp of ones
    asymmetries = g.new_vp('double', np.ones_like(g.get_vertices()))
    # iterate through internal nodes
    for v in g_branch_inds(g):
        asymmetries[v] = _node_asymmetry(g, v, L, weight = weight)

    if bind:
        if weight:
            g.vp['Weighted_asymmetry'] = asymmetries
        else:
            g.vp['Asymmetry'] = asymmetries
    else:
        return asymmetries
    

def expected_asymmetry(N: Tree_graph | gt.Graph | gt.VertexPropertyMap,method: str = 'mean') -> float :
    """
    Get mean or Median graph asymmetry for N

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph | gt.VertexPropertyMap
        Graph representation of neuron or vertex property map of vertex asymmetry values. If a graph or Tree_graph 
        object is given, it must have the 'Asymmetry' or 'Weighted_asymmetry' metric bound to it as an internal vertex property map
    method : str, optional
        which to calculate, the mean or median asymmetry, by default 'mean'

    Returns
    -------
    float
        The expectancy (mean or median) of given asymmetry values

    Raises
    ------
    AttributeError
        Raised if N is a tree_graph, or gt.Graph, but doesn't have a suitable vertex property ('Asymmetry', or 'Weighted_asymmetry')
    TypeError
        Raised if N is not a Tree_graph, gt.Graph, or gt.VertexPropertyMap, meaning we cannot get the needed data
    """

    # get data
    arr = None
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N

    if g_has_property(g,'Weighted_asymmetry'):
        arr = g.vp['Weighted_asymmetry'].a
    elif g_has_property(g, 'Asymmetry'):
        arr = g.vp['Asymmetry'].a
    else:
        raise AttributeError('Input neuron does not have an Asymmetry vertex property, please generate one using node_asymmetry function')

    if isinstance(N, gt.VertexPropertyMap):
        arr = N.a
    if arr == None:
        raise TypeError('Failed to generate data from N')
    
    # get expectancy
    if method == 'mean':
        return np.mean(arr)
    elif method == 'median':
        return np.median(arr)

### Branching Angles    

def get_child_angles(N:Tree_graph, to_degree:bool = True, bind:bool = True) -> np.ndarray | None:
    """
    Computes the angles between child branches at bifurcation points in a tree graph.

    This function identifies bifurcation points in the tree graph and calculates the angle 
    between the two bifurcating child branches. The angles can be returned 
    as a NumPy array or stored in the graph's edge property map.

    Parameters
    ----------
    N : nr.Tree_graph
        Tree_graph representation of Neuron
    to_degree : bool, optional
        If True, the angles are returned in degrees; otherwise, they are returned in radians. 
        Default is True.
    bind : bool, optional
        If True, the computed angles are stored in the graph's edge property map (`'Child_angles'`); 
        otherwise, the function returns a NumPy array containing the angles. Default is True.

    Returns
    -------
    np.ndarray | None
        If `bind` is False, returns a NumPy array of angles corresponding to each bifurcation point.
        If `bind` is True, updates the graph's edge property and returns None.

    Notes
    -----
    - The function extracts all edges and coordinates from the tree graph.
    - It identifies bifurcation points (nodes with exactly two outgoing edges).
    - The angles are computed using the child branch vectors and a perpendicular normal vector.
    - If `bind=True`, the computed angles are stored in the tree graph under the edge property `'Child_angles'`.

    Examples
    --------
    >>> angles = get_child_angles(my_tree_graph, to_degree=False, bind=False)
    >>> print(angles)  # Returns an array of angles in radians

    >>> get_child_angles(my_tree_graph, bind=True)  # Stores angles in 'Child_angles' property
    >>> print(my_tree_graph.graph.ep['Child_angles'])  # Access stored angles
    """
    # get all edges
    edges = get_edges(N)
    # get all coordinates
    coords = g_vert_coords(N)
    # get all branches
    branch_inds = g_branch_inds(N)
    # subset out root
    branch_inds = branch_inds[branch_inds != g_root_ind(N)]
    # keep only bifurications
    branch_inds = branch_inds[np.where(N.graph.get_out_degrees(branch_inds) == 2)]
    # get child edges, as in the source is branch - we index to only get the target of the edge
    source_inds = np.isin(edges[:,0],branch_inds)
    child_edges_to_keep = edges[source_inds]
    child_edges = child_edges_to_keep[:,1]

    # split into odd and even
    even_child = child_edges[::2]
    odd_child = child_edges[1::2]



    # we can now use child_edges and parent_edges to subset coords

    # v1 will be the coordinate of the source of the edge leading into the branch node
    v1 = coords[even_child]
    # v2 will the the coordinate of the target of all child edges from a branch node
    v2 = coords[odd_child]
    # i need to subtract the branch coordinates from these so we are on [0,0,0]
    branch_coords = coords[branch_inds]
    v1 -= branch_coords
    v2 -= branch_coords


    # and turn into unit vectors
    v1 = GeoJax.normalise(v1)
    v2 = GeoJax.normalise(v2)

    # ok with v1 and v2, we want to also define a perspective for calculating the signed angle later
    # we will use a vector perpendicular from v1 and v2
    normals = np.array(GeoJax.perpendicular(v1,v2))

    # get angles
    angles = GeoJax.angle(v1,v2,normals, to_degree=to_degree)

    if bind:
        # I need an array of infinities of length(edges)
        data = np.zeros(N.graph.num_edges())
        data[:] = np.inf

        # then at the correct index I need to make infinity the angle for this child pair...
        # Convert both arrays into structured dtypes for easy comparison
        arr1_struct = edges.view([('', edges.dtype)] * edges.shape[1])
        arr2_struct = child_edges_to_keep.view([('', child_edges_to_keep.dtype)] * child_edges_to_keep.shape[1])

        # Find indices in arr1 where rows match any row in arr2
        matching_indices = np.nonzero(np.isin(arr1_struct, arr2_struct))[0]
        # update data
        data[matching_indices] = np.repeat(angles, 2)

        N.graph.ep['Child_angle'] = N.graph.new_ep('double', data)
    else:
        return angles

def get_parent_child_angles(N:Tree_graph, to_degree:bool = True, bind:bool = True) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Computes the angles between parent and child branches at bifurcation points in a tree graph.

    This function identifies bifurcation points in the tree graph and calculates the angles between 
    the parent branch and each of its two child branches at each bifurcation. The angles can be returned 
    as two separate NumPy arrays (one for each child) or stored in the graph's edge property map.

    Parameters
    ----------
    N : nr.Tree_graph
        Tree_graph representation of Neuron
    to_degree : bool, optional
        If True, the angles are returned in degrees; otherwise, they are returned in radians. 
        Default is True.
    bind : bool, optional
        If True, the computed angles are stored in the graph's edge property map (`'Parent_angles'`); 
        otherwise, the function returns two NumPy arrays containing the angles for each child branch. 
        Default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] | None
        If `bind` is False, returns a tuple of two NumPy arrays:
        - `angles_even`: Angles corresponding to the first child branch at each bifurcation.
        - `angles_odd`: Angles corresponding to the second child branch at each bifurcation.
        If `bind` is True, updates the graph's edge property `'Parent_angles'` and returns None.

    Notes
    -----
    - The function extracts all edges and coordinates from the tree graph.
    - It identifies bifurcation points (nodes with exactly two outgoing edges).
    - The angles are computed using the parent and child branch vectors and a perpendicular normal vector.
    - To ensure consistent signed angles, the normal vectors are aligned.
    - If `bind=True`, the computed angles are stored in the tree graph under the edge property `'Parent_angles'`.

    Examples
    --------
    >>> angles_even, angles_odd = get_parent_child_angles(my_tree_graph, to_degree=False, bind=False)
    >>> print(angles_even)  # Returns an array of angles for the first child branch
    >>> print(angles_odd)   # Returns an array of angles for the second child branch

    >>> get_parent_child_angles(my_tree_graph, bind=True)  # Stores angles in 'Parent_angles' property
    >>> print(my_tree_graph.graph.ep['Parent_angles'])  # Access stored angles
    """
    # get all edges
    edges = get_edges(N)
    # get all coordinates
    coords = g_vert_coords(N)
    # get all branches
    branch_inds = g_branch_inds(N)
    # subset out root
    branch_inds = branch_inds[branch_inds != g_root_ind(N)]
    # keep only bifurications
    branch_inds = branch_inds[np.where(N.graph.get_out_degrees(branch_inds) == 2)]
    # get child edges, as in the source is branch - we index to only get the target of the edge
    source_inds = np.isin(edges[:,0],branch_inds)
    child_edges_to_keep = edges[source_inds]
    child_edges = child_edges_to_keep[:,1]
    # now we want the parents of the branches - again we index, this time to only get the source
    source_inds = np.isin(edges[:,1], branch_inds)
    parent_edges = edges[source_inds][:,0]
    # we want these to be present twice
    parent_edges = np.repeat(parent_edges,2)

    # we can now use child_edges and parent_edges to subset coords

    # v1 will be the coordinate of the source of the edge leading into the branch node
    v1 = coords[parent_edges]
    # v2 will the the coordinate of the target of all child edges from a branch node
    v2 = coords[child_edges]

    # i need to subtract the branch coordinates from these so we are on [0,0,0]
    branch_coords = coords[np.repeat(branch_inds,2)]
    v1 -= branch_coords
    v2 -= branch_coords

    # and turn into unit vectors
    v1 = GeoJax.normalise(v1)
    v2 = GeoJax.normalise(v2)

    # ok with v1 and v2, we want to also define a perspective for calculating the signed angle later
    # we will use a vector perpendicular from v1 and v2
    normals = np.array(GeoJax.perpendicular(v1,v2))

    # this fucks up the sign though, as the directions of the normals will be flipped
    # to fix this lets make sure each pair of normals is aligned
    normals[::2] = np.vstack([vg.aligned_with(normals[::2][i], normals[1::2][i]) for i in range(int(normals.shape[0] / 2))])

    # get angles
    angles = GeoJax.angle(v1,v2,normals, to_degree=to_degree)

    # split 
    # lets split angles by odd and even
    angles_even = angles[::2]
    angles_odd = angles[1::2]

    if bind:
        # I need an array of infinities of length(edges)
        data = np.zeros(N.graph.num_edges())
        data[:] = np.inf

        # then at the correct index I need to make infinity the angle for this child pair...
        # Convert both arrays into structured dtypes for easy comparison
        arr1_struct = edges.view([('', edges.dtype)] * edges.shape[1])
        arr2_struct = child_edges_to_keep.view([('', child_edges_to_keep.dtype)] * child_edges_to_keep.shape[1])

        # Find indices in arr1 where rows match any row in arr2
        matching_indices = np.nonzero(np.isin(arr1_struct, arr2_struct))[0]
        # update data
        data[matching_indices] = angles

        N.graph.ep['Parent_angle'] = N.graph.new_ep('double', data)

    else:
        return angles_even, angles_odd

def unpack_parent_child_angles(N:Tree_graph, split: bool = True) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """retrieve np.ndarrays of paired angles going from parent edge to child edge. 
    
    if split, As each branch node has two children, we split the angles into first and second child and return an array for each.

    This means arr1[i] and arr2[i] come from the same parent

    If split is false, we return a single array, however adjacent odd and even rows come from the same parent.

    If N does not have the 'Parent_angle' property, it will be calculated and added.

    Parameters
    ----------
    N : nr.Tree_graph
        Tree_graph representation of Neuron
    split: bool
        if we want to split the output or not, by default True
    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        Arrays of angles between parent vector and child vector, paired as described above.
    """

    # if we don't have the parent to child branch angle property add it to the neuron
    if not g_has_property(N.graph, 'Parent_angle'):
        get_parent_child_angles(N, to_degree = True, bind = True)

    # get all edges
    edges = get_edges(N)

    # get all branches
    branch_inds = g_branch_inds(N)
    # subset out root
    branch_inds = branch_inds[branch_inds != g_root_ind(N)]
    # keep only bifurications
    branch_inds = branch_inds[np.where(N.graph.get_out_degrees(branch_inds) == 2)]
    # get child edges, as in the source is branch - we index to only get the target of the edge
    source_inds = np.isin(edges[:,0],branch_inds)
    child_edges = edges[source_inds]

    angles = np.array([N.graph.ep['Parent_angle'][e] for e in child_edges])
    if split:
        return angles[::2], angles[1::2]
    else:
        return angles
    
def propagate_vp_to_ep(N:Tree_graph, vp:str, ep:str, by:str = 'target'):
    """_summary_

    Parameters
    ----------
    N : Tree_graph
        Tree graph representation of a neuron
    vp : str
        name of vertex property we wish to propegate
    ep : str
        name of new edge property which will be added to neuron
    by : str, optional
        Nature of the vertex to edge mapping. If target, vertex property will be mapped to edges with this vertex as its target.
        alternatively, if source, property will be mapped to edges with with the specific vertex value as the source. By default 'target'

    Raises
    ------
    AttributeError
        by must be either source or target
    """
    vp = N.graph.vp[vp].a
    edges = get_edges(N)
    if by == 'target':
        e_ind = edges[:,1]
    elif by == 'source':
        e_ind == edges[:,]
    else:
        raise AttributeError(f'by method {by} not applicable, expected source or target')
    N.graph.ep[ep] = N.graph.new_ep('int',vp[e_ind])

def get_subtree_cable_length(N: Tree_graph, v:int) -> float:
    """Returns the total cable length of the subtree in a neuron defined with it's root at vertex v

    Parameters
    ----------
    N : neurosetta.Tree_graph
        neuron sub-tree is from
    v : int
        Index of vertex which gives the root of the sub-tree

    Returns
    -------
    float
        Total cable length of sub-tree
    """
    return sum([N.graph.ep['Path_length'][e] for e in gt.dfs_iterator(N.graph, v)])