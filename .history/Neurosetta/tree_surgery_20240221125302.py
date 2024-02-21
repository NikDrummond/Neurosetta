import graph_tool.all as gt
import numpy as np
import scipy.stats as stats
from .core import Tree_graph, infer_node_types
from .graphs import *


def reroot_tree(N: Tree_graph | gt.Graph, root: int, inplace=False, prune_soma = True):
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

    if isinstance(N, Tree_graph):
        if inplace:
            N.graph = g2
        else:
            return    
            N = Tree_graph(name=N.name, graph=g2)
            return N
    else:
        return g2


def prune_soma(N:Tree_graph | gt.Graph, inplace=False):

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