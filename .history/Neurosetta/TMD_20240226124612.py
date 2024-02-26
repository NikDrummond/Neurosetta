from .core import Tree_graph
from .graphs import g_leaf_inds, g_root_ind

import graph_tool.all as gt
import numpy as np


def TMD(N: Tree_graph | gt.Graph, func: str | gt.VertexPropertyMap, bind=True):
    """_summary_

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        _description_
    func : str | gt.VertexPropertyMap
        _description_

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
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # indicies of leaves and root
    l_inds = g_leaf_inds(g)
    root = g_root_ind(g)

    # initialise segments and visited nodes
    segments = np.zeros((len(l_inds), 2)).astype(int)
    visited = np.zeros_like(g.get_vertices()).astype(bool)

    # indicies of leaves sorted by descending dist func
    if isinstance(func, str):
        sorted_l = l_inds[np.argsort(g.vp[func].a[l_inds])[::-1]]
    elif isinstance(func, gt.VertexPropertyMap):
        sorted_l = l_inds[np.argsort(func.a[l_inds])[::-1]]

    # iterate
    for i in range(len(sorted_l)):
        l = sorted_l[i]
        # get parent
        current_p = g.get_in_neighbours(l)[0]
        # while we are collecting nodes:
        while True:
            # if parent not in viewed:
            if visited[current_p] == False:
                # add to viewed
                visited[current_p] = True

                # if we are at the root:
                if current_p == root:
                    # note this segment
                    segments[i] = [l, current_p]
                    # move to next leaf
                    break
                else:
                    # move to next node
                    current_p = g.get_in_neighbours(current_p)[0]
            # else:
            else:
                # note node and l - node to segments
                segments[i] = [l, current_p]
                # move to next leaf
                break
    # get length of segments according to function used
    if isinstance(func, str):
        lens = np.array([abs(g.vp[func][i[0]] - g.vp[func][i[1]]) for i in segments])
    elif isinstance(func, gt.VertexPropertyMap):
        lens = np.array([abs(func[i[0]] - func[i[1]]) for i in segments])

    TMD = dict(
        {
            "function": func,
            "birth_ind": segments[:, 1],
            "death_ind": segments[:, 0],
            "survival_len": lens,
        }
    )

    if bind:
        g.gp["TMD"] = g.new_gp("object", TMD)
    else:
        return TMD

def TMD_barcode(g):

    # add check to make sure TMD graph property exists
    if g_has_property(g,'TMD','g'):
        segments = np.vstack((g.gp['TMD']['birth_ind'], g.gp['TMD']['death_ind'])).T
        # get leaf inds
        l_inds = g_leaf_inds(g)
        x = np.zeros_like(segments)
        y = np.vstack((np.arange(len(l_inds)) + 1,np.arange(len(l_inds)) + 1))


        # get length of each segments
        seg_len = g.gp['TMD']['survival_len']
        sort_inds = np.argsort(seg_len)

        for i in range(len(sort_inds)):
            curr_seg = segments[sort_inds[i]]
            x[i,0] = g.vp['radial_dist'][curr_seg[0]]
            x[i,1] = g.vp['radial_dist'][curr_seg[1]]      
        x = x.T     

        return x,y
    
    else:
        raise AttributeError('Graph has no TMD dictionary - generate this using the TMD function')