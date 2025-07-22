from .core import *
from .graphs import g_vert_coords, g_root_ind
from typing import List
import vedo as vd

# set backend
vd.settings.default_backend = "vtk"


def plot3d(N, radius: bool = False, soma: bool = True, **kwargs) -> vd.Plotter:
    """
    simple 3d plot function using vedo
    """

    # check the input type
    if isinstance(N, Tree_graph):
        # if we are plotting radius
        if radius == True:
            tubes = _vd_tree_cyl(N, **kwargs)
            vd.show(tubes).close()

        else:
            # collection of lines for plotting
            lns = _vd_tree_lines(N, **kwargs)
            if soma:
                pnt = vd.Point(g_vert_coords(N, g_root_ind(N))[0])
                vd.show([lns, pnt]).close
            else:
                vd.show(lns).close()

    elif isinstance(N, Node_table):
        if radius == True:
            tubes = _vd_nodes_cyl(N, **kwargs)
            vd.show(tubes).close()

        else:
            lns = _vd_nodes_lines(N, **kwargs)
            vd.show(lns).close()

    elif isinstance(N, Neuron_mesh):
        # generate vedo mesh
        mesh = _vd_mesh(N, **kwargs)
        # plot
        vd.show(mesh).close()


def _vd_tree_st_end(N: Tree_graph) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # get all edges
    edges = N.graph.get_edges()
    # get coordinate array
    coords = g_vert_coords(N)
    # initialise start and end points
    edges = N.graph.get_edges()
    coords = g_vert_coords(N)

    start_pts = coords[edges[:, 0]]
    end_pts = coords[edges[:, 1]]

    return start_pts, end_pts


def _vd_nodes_st_end(N: Node_table) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # node ids
    node_ids = N.nodes["node_id"].values
    # this by node ids - np.array
    coords = N.nodes[["x", "y", "z"]].values
    # get indicies of parent ids

    # get parent node ids
    parents = N.nodes.parent_id.values
    # collect ind. or root
    root_ind = np.where(parents == -1)[0][0]
    # remove root label
    parents = parents[parents != -1]
    # inds of parents in coords
    parent_inds = np.array([np.where(node_ids == i)[0][0] for i in parents])
    # get coordinates of parents
    start_pts = coords[parent_inds]
    # set boolian ind to get children
    bool_ind = np.ones_like(node_ids, bool)
    # set root to false
    bool_ind[root_ind] = False
    # remove root from child coords
    end_pts = coords[bool_ind]

    return start_pts, end_pts, bool_ind


def _vd_tree_lines(N: Tree_graph, **kwargs) -> vd.Lines:

    start_pts, end_pts = _vd_tree_st_end(N)

    # collection of lines for plotting
    lns = vd.Lines(start_pts=start_pts, end_pts=end_pts, **kwargs)
    return lns


def _vd_tree_cyl(
    N: Tree_graph, radius_prop: str = "radius", **kwargs
) -> List[vd.Cylinder]:
    start_pts, end_pts = _vd_tree_st_end(N)
    # create a list of cylinders
    if isinstance(radius_prop, str):
        radii = N.graph.vp[radius_prop].a
    elif isinstance(radius_prop, gt.VertexPropertyMap):
        radii = radius_prop.a
    # radii = radii[bool_ind]
    # create list of cylinders
    tubes = [
        vd.Cylinder(pos=[start_pts[i], end_pts[i]], r=radii[i], **kwargs)
        for i in range(start_pts.shape[0])
    ]

    return tubes


def _vd_subtree_lns(
    N: Tree_graph, c1: str = "g", c2: str = "r", **kwargs
) -> tuple[vd.Lines, vd.Lines]:
    """Create plotting object for subtree plotting, assuming neuron has

    Parameters
    ----------
    N : nr.Tree_graph
        Neuron tree graph
    c1 : str, optional
        Colour for section of neuron in sub tree, by default 'g'
    c2 : str, optional
        Colour for section of neuron not in sub tree, by default 'r'

    Returns
    -------
    tuple[vd.Lines, vd.Lines]
        Plotting objects of neuron in and not in sub tree.
    """
    ### only do something if we have the required property maps
    if g_has_property(N, g_property="subtree_mask", t="v") & g_has_property(
        N, g_property="subtree_mask", t="e"
    ):
        # if we do not have inverted masks, create them:
        if not g_has_property(N, g_property="inv_subtree_mask", t="v") & g_has_property(
        N, g_property="inv_subtree_mask", t="e"
    ):
            # create inverted masks
            msk = np.array(N.graph.vp['subtree_mask'].a).astype(bool)
            msk_inv = ~msk
            N.graph.vp['inv_subtree_mask'] = N .graph.new_vp('bool', msk_inv)

            msk = np.array(N.graph.ep['subtree_mask'].a).astype(bool)
            msk_inv = ~msk
            N.graph.ep['inv_subtree_mask'] = N.graph.new_ep('bool',msk_inv)
        
        # get all coordinates
        coords = g_vert_coords(N)



        # get lines in the subtree
        g2 = gt.GraphView(N.graph)
        g2.set_filters(eprop = g2.ep['subtree_mask'], vprop = g2.vp['subtree_mask'])

        edges = g2.get_edges()
        # get starts and stops
        start_pnts = coords[edges[:, 0]]
        stop_pnts = coords[edges[:, 1]]
        # create excluded lines
        lns_in = vd.Lines(start_pts=start_pnts, end_pts=stop_pnts, c=c1, lw = 3)
        
        # get lines not in the subtree
        g3 = gt.GraphView(N.graph)
        g3.set_filters(eprop = g3.ep['inv_subtree_mask'], vprop = g3.vp['inv_subtree_mask'])
        edges = g3.get_edges()
        # get starts and stops
        start_pnts = coords[edges[:, 0]]
        stop_pnts = coords[edges[:, 1]]
        # create excluded lines
        lns_out = vd.Lines(start_pts=start_pnts, end_pts=stop_pnts, c="r", lw = 3)


        # clear filters again
        N.graph.clear_filters()
        # return
        return lns_in, lns_out

    # if we do not have properties!
    else:
        raise AttributeError(
            "Neuron must have bound subtree masks for vertices and edges"
        )


def _vd_nodes_lines(N: Node_table, **kwargs) -> vd.Lines:
    start_pts, end_pts, bool_ind = _vd_nodes_st_end(N)
    # collection of lines for plotting
    lns = vd.Lines(start_pts=start_pts, end_pts=end_pts, **kwargs)
    return lns


def _vd_nodes_cyl(N: Node_table, **kwargs) -> List[vd.Cylinder]:
    start_pts, end_pts, bool_ind = _vd_nodes_st_end(N)
    # create a list of cylinders
    radii = N.nodes.radius.values
    radii = radii[bool_ind]
    # create list of cylinders
    tubes = [
        vd.Cylinder(pos=[end_pts[i], start_pts[i]], r=radii[i], **kwargs)
        for i in range(start_pts.shape[0])
    ]

    return tubes


def _vd_mesh(N: Neuron_mesh, **kwargs) -> vd.Mesh:
    # generate vedo mesh
    mesh = vd.Mesh([N.vertices, N.faces], **kwargs)

    return mesh
