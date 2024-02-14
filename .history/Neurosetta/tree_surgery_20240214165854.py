import graph_tool.all as gt
import numpy as np


def reroot_tree(g:gt.Graph,root:int):
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
    # get new edge list - hashing keeps ids
    g = N.graph.copy()
    g.set_directed(False)
    edges = gt.dfs_iterator(g, root, array = True)

    # generate new graph
    g2 = gt.Graph(edges, hashed = True, hash_type = 'int')

    # get coordinates
    coords = np.array([g.vp['coordinates'][i] for i in g2.vp['ids'].a])
    vprop_coords = g2.new_vp('vector<double>')
    vprop_coords.set_2d_array(coords.T)
    g2.vp["coordinates"] = vprop_coords
    # radius information - need to map. 'ids' vp is now the index in the original graph
    vprop_rad = g2.new_vp('double')
    vprop_rad.a = g.vp['radius'].a[g2.vp['ids'].a]

    g2.vp['radius'] = vprop_rad
    # regenerate node types
    infer_node_types(g2)

    return g2

def prune_soma(g):    
    if g.degree_property_map('out')[nr.g_root_ind(g)] != 1:
        # this gives the edge we wish to keep
        cable = 0
        for i in g.iter_out_edges(nr.g_root_ind(g)):
            new_cable = g_cable_length(g,i[1])
            if new_cable > cable:
                cable = new_cable
                edge = i

        remove = np.array([])
        for i in g.iter_out_edges(g_root_ind(g)):
            if i != edge:
                # collect nodes to remove
                downstream = nr.downstream_vertices(g, i[1])
                remove = np.hstack((remove,downstream))

        # copy
        g2 = g.copy()
        remove = remove.astype(int)


        mask = np.ones_like(g2.get_vertices())
        mask[remove] = 0

        mask_prop = g2.new_vp('bool')#
        mask_prop.a = mask        

        g2.set_vertex_filter(mask_prop)
        g2.purge_vertices()

        return g2
    else:
        return g