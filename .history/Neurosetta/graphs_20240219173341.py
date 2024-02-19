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
    Check if a property is within a graph
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
    else:
        return (("v", g_property) in g.properties) | (("e", g_property) in g.properties)


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
        coords = np.array([g.vp["coordinates"][i] for i in g.get_vertices()])
    else:
        # if subset is just a single int, convert to a list
        if isinstance(subset, (int, np.integer)):
            subset = [subset]
        coords = np.array([g.vp["coordinates"][i] for i in subset])

    return coords


def get_g_distances(
    N: Tree_graph | gt.Graph, inplace: bool = False, name: str = "Path_length"
) -> None | gt.PropertyMap:
    """
    create edge property map of edge lengths for a graph with coordinates vertex property
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # check coordinates are a property
    if not g_has_property(g, "coordinates", t="v"):
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


def g_leaf_inds(N: Tree_graph | gt.Graph) -> np.ndarray[int]:
    """
    Returns a numpy array of leaf node indices
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
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
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # graph leaf inds - includes soma
    return np.where(g.degree_property_map("out").a > 1)[0]


def g_lb_inds(N: Tree_graph | gt.Graph, return_types: bool = False) -> np.ndarray[int]:
    """
    Returns indices of all leaf and branch nodes

    Note: only useful if you want them all together and don't care which is which
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    l_inds = g_leaf_inds(g)
    b_inds = g_branch_inds(g)

    inds = np.unique(np.concatenate([l_inds, b_inds]))
    return inds


def g_root_ind(N: Tree_graph | gt.Graph) -> int:
    """
    Return integer of root node index
    """
    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")
    return np.where(g.degree_property_map("in").a == 0)[0][0]


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
    # if we need to make the graph undirected - remeber we need to swap this back
    if directed:
        g.set_directed(False)

    # add edge wieghts based on distance
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

    # make symetric - I am gratutiously assuming it isn't only because of floating point errors
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
    elif isinstance(N,gt.Graph):
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
    elif isinstance(N,gt.Graph):
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
    elif isinstance(N,gt.Graph):
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
    point: np.ndarray,
    return_dist: bool = False,
) -> int | tuple:
    """
    Find the index of the vertex in coords closest to the given point.

    Parameters
    ----------
    N : np.ndarray | nr.Tree_graph
        Array of vertex coordinates or neuron tree graph
    point : np.ndarray
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

    try:
        binary_array = np.isclose(coords, point)

        # if there is an exact match
        if len(np.where(binary_array.sum(axis=1) == 3)[0]):
            nearest_v = np.where(binary_array.sum(axis=1) == 3)[0][0]
        dist = 0
    # if there is no exact match, fall back to a KDTree
    else:
        all_coords = np.vstack((coords, point))
        tree = KDTree(all_coords)
        nearest_v = tree.query(all_coords[-1], k=[2])
        dist = nearest_v[0][0]
        nearest_v = nearest_v[1][0]

    if return_dist:
        return (nearest_v, dist)
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
    get_g_distances(g, inplace=True)

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
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    return np.unique(gt.dfs_iterator(g, source, array=True))


def edge_length(i, g):
    return g.ep["Path_length"][i]


def g_cable_length(N: Tree_graph | gt.Graph, source: int = 0) -> float:

    # check input type
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if not g_has_property(g, "Path_length", t="e"):
        get_g_distances(g, inplace=True)
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
