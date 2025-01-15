import graph_tool.all as gt
import numpy as np
import scipy.stats as stats
from scipy.cluster.hierarchy import fcluster
import fastcluster as fc
from .core import Tree_graph, infer_node_types
from tqdm import tqdm
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
    mat = dist_mat(g, subset, method = distance, flatten=True)
    # generate linkage matrix
    Z = fc.linkage(mat, method = method)
    # cluster inds
    c_ids = fcluster(Z,k, criterion = 'maxclust')
    return Z, c_ids        

def linkage_cluster_permutation(clusters:np.ndarray,Z:np.ndarray,inplace = True,root_cluster:int | None = None,perms:int = 1000,a:float = 0.001):
    """_summary_

    Parameters
    ----------
    clusters : np.ndarray
        _description_
    Z : np.ndarray
        _description_
    inplace : bool, optional
        _description_, by default True
    root_cluster : int | None, optional
        _description_, by default None
    perms : int, optional
        _description_, by default 1000
    a : float, optional
        _description_, by default 0.001

    Returns
    -------
    _type_
        _description_
    """
    cluster_ids = np.unique(clusters)

    if root_cluster is not None:
        np.delete(cluster_ids,np.where(cluster_ids == root_cluster))
    if not inplace:
        data = clusters.copy()

    for cluster_id in cluster_ids:
        cluster = np.where(clusters == cluster_id)[0]
        link_dist = np.array([Z[np.where( (Z[:,0] == c) | (Z[:,1] == c))][0][2] for c in cluster])
        # perform permutation
        perm_sample = np.zeros(perms)
        for i in range(perms):
            perm_sample[i] = np.mean(np.random.choice(link_dist,2))
        # calculate exact p    
        ps = np.array([len(perm_sample[ perm_sample >= t]) / len(perm_sample) for t in link_dist])    
        if inplace:
            clusters[cluster[np.where(ps <= a)]] = -1
        else:
            data[cluster[np.where(ps <= a)]] = -1

    if inplace:            
        return clusters
    else:
        return data     

### Traversal based community detection

### Current functions

def feedback_tree(g):

    edges = g.get_edges()
    # stack inverted edges
    edges = np.vstack([edges,edges[:,[1,0]]])

    weights = g.ep['Path_length'].a
    # stack weights (weight is the same in both directions)
    weights = np.hstack([weights,weights])

    # make graph
    g = gt.Graph(edges)
    g.ep['Path_length'] = g.new_ep('double',weights)

    return g

def traversal_probs(weights):
    # Calculate the sum of the inverses of the edge weights
    inverse_sum = sum(1 / w for w in weights)
    # Calculate the probability for each edge
    probabilities = [(1 / w) / inverse_sum for w in weights]
    
    return probabilities

def g_traversal_weights(g):

    eprop_weights = g.new_ep('double')

    # iterate over vertices
    for v in g.iter_vertices():
        # get out edges
        edges = g.get_out_edges(v)
        # if there is only one edge, prob is 1
        if len(edges) == 1:
            eprop_weights[edges[0]] = 1
        else:
            # get out weights
            lengths = [g.ep['Path_length'][e] for e in edges]
            # exp_lengths = np.exp(np.array(lengths))
            # calulate out weights
            out_weights = traversal_probs(lengths)
            # iterate over out edges and update eprop_weights
            for e in range(len(edges)):
                eprop_weights[edges[e]] = out_weights[e]

    return eprop_weights

def path_traversal_probability(g,i,j,l_inds,eprop_weights):
    path_es = gt.shortest_path(g,l_inds[i],l_inds[j])[1]
    traversal_prob = np.prod(np.array([eprop_weights[e] for e in path_es]))
    return traversal_prob

def traversal_matrix(g,eprop_weights,inds):
    # initialise matrix
    mat = np.zeros((len(inds),len(inds)))
    # get indicies of the upper triangle excluding diagonal
    rows, cols = np.triu_indices(len(inds),k=1)

    # iterate to get traversal probs
    for i,j in tqdm(zip(rows, cols)):
        mat[i,j] = path_traversal_probability(g,i,j,inds,eprop_weights)

    mat[cols,rows] = mat[rows,cols]
    return mat

### downstream_weights

def downstream_weights(g,feedback_g_weights, edges, method = 'edge'):
    # convert to property map in original graph
    eprop_weights = g.new_ep('double',[feedback_g_weights[e] for e in g.iter_edges()])
    # create vertex property map of node probabilities
    vprop_probs = g.new_vp('double')

    for v in g.iter_vertices():
        vprop_probs[v] = np.prod([eprop_weights[e] for e in gt.dfs_iterator(g,v)])

    if method != 'node':
        # propagate these node probabilities to the edges
        eprop_probs = g.new_ep('double',vprop_probs.a[edges[:,1]])

    # what to return
    if method == 'edge':
        return eprop_probs
    elif method == 'node':
        return vprop_probs
    else:
        return eprop_probs, vprop_probs

def upstream_weights(g,feedback_g_weights, edges, l_inds, method = 'edge'):

    vprop_probs = g.new_vp('double')

    # if v is a leaf, set value to 1 (saves about 1 second and is quicker than sub-setting and only iterating over branches)
    adjust = 0
    for v in g.iter_vertices():
        if v in l_inds:
            vprop_probs[v] = 1
            adjust += 1
        else:
            # this gives us a list paths from some root to leaves
            paths = [gt.shortest_path(g,v,l)[1] for l in l_inds]
            data = [[feedback_g_weights[[e.target(),e.source()]] for e in p] for p in paths]
            data = np.hstack(data)
            adjust += len(data)
            vprop_probs[v] = np.prod(data)

    if method != 'node':
        # propagate to edges
        eprop_probs = g.new_ep('double',vprop_probs.a[edges[:,1]])

    # what to return
    if method == 'edge':
        return eprop_probs, adjust
    elif method == 'node':
        return vprop_probs, adjust
    else:
        return eprop_probs, vprop_probs, adjust
    
def community_edge_mask(N_simp,p_thresh = 0.001, len_thresh = 0.1):
    ### set up as before
    g = N_simp.graph.copy()
    edges = g.get_edges()
    g2 = feedback_tree(N_simp.graph)
    l_inds = g_leaf_inds(g)
    feedback_g_weights = g_traversal_weights(g2)

    ## use new function
    eprop_probs_upstream, adjust = upstream_weights(g, feedback_g_weights, edges, l_inds)

    p = p_thresh / adjust
    coords = g_vert_coords(N_simp)

    edges = g.get_edges()
    # keep = edges[np.where(eprop_probs_upstream.a>=p)]
    # out = edges[np.where(eprop_probs_upstream.a < p)]

    eprop_mask = g.new_ep('bool')

    # edges we will consider for removal (backbone)
    rem_edges = edges[np.where(eprop_probs_upstream.a < p)]
    # get longest edge weight in g
    norm_edge = g.ep['Path_length'].a.max()
    # get path lengths in backbone
    backbone_lengths = np.array([g.ep['Path_length'][e] for e  in rem_edges])
    # normalise backbone weights
    norm_backbone_lengths = backbone_lengths / norm_edge

    ### need some other metric. Size of the sub trees which will be merged?

    # which are greater than 0.1 of the longest edge
    rem_edges = rem_edges[np.where(norm_backbone_lengths > len_thresh)]
    # update mask
    for e in rem_edges:
        eprop_mask[e] = 1

    # make vertex property of edge targets
    vprop_mask = g.new_vp('bool')
    for v in rem_edges[:,1]:
        vprop_mask[v] = 1

    return eprop_mask, vprop_mask

def _k_largest_component_roots(N,k = 2):

    # get all root inds
    all_roots = g_root_ind(N, True)
    if len(all_roots) == 1:
        raise ValueError('N only has one root, doing nothing')
    
    # get the size of all sub trees
    sub_sizes = np.zeros(len(all_roots))
    for i in range(len(all_roots)):
        sub_sizes[i] = sum([N.graph.ep['Path_length'][e] for e in gt.dfs_iterator(N.graph,all_roots[i], array = True)])

    # Get indices of the k highest values
    indices = np.argpartition(-sub_sizes, k)[:k]

    # Sort these indices by their values to get them in descending order
    sorted_indices = indices[np.argsort(-sub_sizes[indices])]

    return all_roots[sorted_indices]

def _g_component_masks(g,roots, bind = True):
    masks = []
    inv_mask = np.ones_like(g.get_vertices()) 
    # make a mask for each root
    for i in range(len(roots)):
        # initialise mask of vertices
        mask = np.zeros_like(g.get_vertices())
        # do dfs from root returning array
        verts = gt.bfs_iterator(g,roots[i], array = True)
        # set of verts in edges

        # update mask
        mask[np.unique(verts)] = 1
        inv_mask[np.where(mask == 1)] = 0
        if bind:
            # name and bind to graph
            g.vp['k_' + str(i+1)] = g.new_vp('bool', mask)
        else:
            masks.append(g.new_vp('bool', mask))

    # if bind, add inverse mask
    if bind:
        g.vp['k_null'] = g.new_vp('bool', inv_mask)
    # else return list of all masks
    else:
        masks.append()

def N_component_masks(N,k, mask, bind = True):
    # set edge filter
    N.graph.set_edge_filter(mask, inverted = True)
    roots = _k_largest_component_roots(N, k)

    # highly recomended to bind, but you don't have to
    if bind:
        _g_component_masks(N.graph,roots, bind = bind)
    else:
        return _g_component_masks(N.graph,roots, bind = bind)