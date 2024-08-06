import graph_tool.all as gt
import numpy as np
import scipy.stats as stats
import fastcluster as fc
from .core import Tree_graph, infer_node_types
from .graphs import *


def reroot_tree(N: Tree_graph | gt.Graph, root: int, inplace=False, prune = True):
    """_summary_

    Parameters
    ----------
    g : gt.Graph
        _description_
    root : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(N, Tree_graph):
        # get new edge list - hashing keeps ids
        g = N.graph.copy()
    elif isinstance(N,gt.Graph):
        g = N.copy()
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    g.set_directed(False)
    edges = gt.dfs_iterator(g, root, array=True)

    # generate new graph
    g2 = gt.Graph(edges, hashed=True, hash_type="int")

    # get coordinates
    coords = np.array([g.vp["coordinates"][i] for i in g2.vp["ids"].a])
    vprop_coords = g2.new_vp("vector<double>")
    vprop_coords.set_2d_array(coords.T)
    g2.vp["coordinates"] = vprop_coords
    # radius information - need to map. 'ids' vp is now the index in the original graph
    vprop_rad = g2.new_vp("double")
    vprop_rad.a = g.vp["radius"].a[g2.vp["ids"].a]

    g2.vp["radius"] = vprop_rad
    # regenerate node types
    infer_node_types(g2)
    
    # if we want to prune
    if prune:
        g2 = prune_soma(g2)

    if isinstance(N, Tree_graph):
        if inplace:
            N.graph = g2
        else:
            return Tree_graph(name=N.name, graph=g2)
    else:
        return g2


def prune_soma(N:Tree_graph | gt.Graph):

    if isinstance(N, Tree_graph):
        g =N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError('N must be Tree_graph or gt.Graph object')        
    # get out degree property map
    out_deg = g.degree_property_map("out")
    if g.degree_property_map("out")[g_root_ind(g)] != 1:
        # this gives the edge we wish to keep
        cable = 0
        for i in g.iter_out_edges(g_root_ind(g)):
            new_cable = g_cable_length(g, i[1])
            if new_cable > cable:
                cable = new_cable
                edge = i

        remove = np.array([])
        for i in g.iter_out_edges(g_root_ind(g)):
            if i != edge:
                # collect nodes to remove
                downstream = downstream_vertices(g, i[1])
                remove = np.hstack((remove, downstream))

        # copy
        g2 = g.copy()
        remove = remove.astype(int)

        mask = np.ones_like(g2.get_vertices())
        mask[remove] = 0

        mask_prop = g2.new_vp("bool")  #
        mask_prop.a = mask

        g2.set_vertex_filter(mask_prop)
        g2.purge_vertices()

        return g2
    else:
        # print("nothing to fix")
        return g

def g_edge_error(
    N: gt.Graph,
    binom_cut: float = 0.001,
    prop_cut: float = 0.01,
    method: str = "cable",
    bind=False,
) -> gt.EdgePropertyMap:
    """_summary_

    Parameters
    ----------
    g : gt.Graph
        _description_
    binom_cut : float, optional
        _description_, by default 0.001
    prop_cut : float, optional
        _description_, by default 0.01
    method : str, optional
        _description_, by default 'cable'
    bind : bool, optional
        _description_, by default False

    Returns
    -------
    gt.EdgePropertyMap
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if isinstance(N, Tree_graph):
        g =N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError('N must be Tree_graph or gt.Graph object')    

    # initialise edge property map
    eprop_err = g.new_ep("int")
    # get out degree property map
    # out_degree property map
    out_deg = g.degree_property_map("out")

    if method == "leaves":
        # count leaves
        norm: int = len(g_leaf_inds(g))
    elif method == "cable":
        norm = g_cable_length(g)
    elif method == "partial_cable":
        norm = None
    else:
        raise ValueError("method must be leaves or cable")

    if not g_has_property(g,"reachable_leaves", t="v"):
        g_reachable_leaves(g, bind=True)

    for i in g.iter_edges():
        # reachable leaves from parent and child
        l_child = g.vp["reachable_leaves"][i[1]]
        l_parent = g.vp["reachable_leaves"][i[0]]
        # out degree of parent
        p_out = out_deg[i[0]]

        # if this is isn't a transitive node
        if p_out != 1:
            # calculate p from binomial
            x = np.linspace(1, l_parent, l_parent)
            binom = stats.binom(n=l_parent, p=1 / p_out)
            binom_x = binom.cdf(l_child)

            # if binom_x is below threshold
            if binom_x < binom_cut:

                # check about how much we will remove
                # what proportion of the graph does this edge lead to?
                if method == "leaves":
                    proportion = l_child / norm
                elif method == "cable":
                    # get cable length of sub tree
                    sub_cable = g_cable_length(g, i[1])

                    # we need to add the length of the source edge
                    sub_cable += edge_length(i, g)
                    proportion = sub_cable / norm
                elif method == "partial_cable":
                    # get cable length of sub tree
                    sub_cable = g_cable_length(g, i[1])
                    # we need to add the length of the source edge
                    sub_cable += edge_length(i, g)
                    # calculate norm - amount of cable from branch point
                    norm = g_cable_length(g, i[0])
                    proportion = sub_cable / norm
                # if we are also removing less of the graph than prop_cut:
                if proportion < prop_cut:
                    eprop_err[i] = 1

    if bind:
        g.ep["possible_error"] = eprop_err
    else:
        return eprop_err
    
def simplify_neuron(N: Tree_graph) -> Tree_graph:

        # make sure we have 'Path_length' property
    if not g_has_property(N,'Path_length',"e"):
        get_g_distances(N, bind = True)
        

    # starts are all leaves and branches
    seg_stops = g_lb_inds(N)
    # ends are all branches and the root
    seg_starts = np.hstack((g_root_ind(N),g_branch_inds(N)))

    # initialise edges array
    # initialise what will become the edges
    edges = np.zeros((segment_counts(N),2)).astype(int)

    index = 0
    for e in gt.dfs_iterator(N.graph,g_root_ind(N),array=True):
        curr_start = e[0]
        curr_end = e[1]

        # if the source vertex is a start point:
        if curr_start in seg_starts:
            edges[index,0] = curr_start

        # if the target vertex is an segment end
        if curr_end in seg_stops:
            edges[index,1] = curr_end
            index += 1        


    g = gt.Graph(edges, hashed = True, hash_type = 'int')


    coords = np.array([N.graph.vp['coordinates'][i] for i in g.vp['ids'].a])
    radius = np.array([N.graph.vp['radius'][i] for i in g.vp['ids'].a])
    g.vp['coordinates'] = g.new_vp("vector<double>", coords)
    g.vp['radius'] = g.new_vp('double',radius)

    # create Path length edge property map - this preserves the path length of the edge from the original graph
    eprop_p = g.new_ep('double')

    for i in g.iter_edges():
        source = g.vp['ids'][i[0]]
        target = g.vp['ids'][i[1]]
        eprop_p[i] = path_length(N,source = source, target = target)

    # add this edge property to the graph
    g.ep['Path_length'] = eprop_p

    # add Euc_dist property - this is the euclidean distance between nodes in the simplified graph
    get_g_distances(g, bind = True,name = 'Euc_dist')

    # add node types
    infer_node_types(g)

    # add simplified graph property
    simp = g.new_gp('bool')
    simp[g] = True
    g.gp['simplified'] = simp

    return Tree_graph(N.name,g)


def apply_mask(N:Tree_graph | gt.Graph, mask: str | np.ndarray | gt.VertexPropertyMap, inplace: bool = False) -> Tree_graph | gt.Graph:
    """Apply a boolian mask to verticies in a Tree graph

    Parameters
    ----------
    N : Tree_graph | gt.Graph
        Object representing neuron graph
    mask : str | array | gt.VertexPropertyMap
        boolian array representing the mask to be applied to verticies. True/1 will be kept
        Can be a string, refering to a vertex property map already bound to the graph, an np.array, or an unbound vertex property map.
    inplace : bool, optional
        If true, will apply the mask to the graph in place.

    Returns
    -------
    Tree_graph | gt.Graph
        Masked graph

    Raises
    ------
    TypeError
        If input N is neither a Tree_graph of graph_tool graph        
    ValueError
        If a string is provided which is not an internal property of the graph
    ValueError
        If provided mask is not usable.
    """
    # make g the graph
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N,gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")  
    
    if not inplace:
        g = g.copy()

    # sort out what the mask is, can be string, array, or vertex proeprty map
    if isinstance(mask,str):
        if not g_has_property(g, g_property = mask, t = 'v'):
            raise ValueError(f"Graph does not have property {mask}")
        # set the filter
        g.set_vertex_filter(g.vp[mask])
    elif isinstance(mask, np.ndarray):
        # converty to a vertex property map     
        mask = g.new_vp("bool", mask)
        g.set_vertex_filter(mask)
    elif isinstance(mask, gt.VertexPropertyMap):
        g.set_vertex_filter(mask)
    else:
        raise ValueError(f"Mask must be string, array, or vertex property map")     

    # purge     
    g.purge_vertices()
    # remake - Note this will purge any non-core properties
    g = reroot_tree(g, g_root_ind(N))
    # reroot adds 'core' properties
    # add path length for edges (if needed)
    if not g_has_property(g,'Path_length'):
        get_g_distances(g, bind = True)

    # if we were given A TREE GRAPH, RETURN A TREE GRAPH
    if isinstance(N,Tree_graph):
        g = Tree_graph(name = N.name, graph = g)

    if inplace:
        if isinstance(N, Tree_graph):
            N.graph = g
            return
    else:
        return g  


# Function to return cluster labels from subset of nodes
def vertex_hierarchical_clusters(N: Tree_graph | gt.Graph,subset: np.ndarray | None = None ,distance: str = 'Path Length',method: str = 'ward',k: int = 3) -> (np.ndarray,np.ndarray):
    """_summary_

    Parameters
    ----------
    N : Tree_graph | gr.Graph
        neurosetta.Tree_graph or graph_tool.Graph representation of a neuron
    subset : np.ndarray | None
        Subset of nodes to calculate cluster allocation to. If None (default) calculate for all nodes 
    distance : str
        Distance to use for hierarchical clustering, can be 'Path Length' or 'Euclidean'.
        'Path Length' uses the distance along the graph between nodes
        'Euclidean' used the Euclidean distance between nodes
    method : str
        Passed to hierarchical clustering, can be 'ward','average' or any other option presented in
        scipy.clustering.hierarchical.linkage
    k : int
        Number of clusters to return.

    Returns
    -------
    np.ndarray
        Resulting linkage matrix
    np.ndarray
        cluster identites, up to k, for each node in subset, or all nodes if no subset given.

    Raises
    ------
    TypeError
        if N is neither a neurosetta Tree_graph or graph_tool Graph
    """
    
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if subset is None:
        subset = g.get_vertices()    
    # generate pairwise distance matrix based on path length
    dist_mat = dist_mat(g, subset, method = distance, flatten=True)
    # generate linkage matrix
    Z = fc.linkage(dist_mat, method = method)
    # cluster inds
    c_ids = fcluster(Z,k, criterion = 'maxclust')
    return Z, c_ids        
