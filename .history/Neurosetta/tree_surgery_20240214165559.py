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
    edges = gt.dfs_iterator(g, root_ind, array = True)

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