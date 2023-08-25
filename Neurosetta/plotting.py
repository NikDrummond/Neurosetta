from .core import *

from typing import List
import vedo as vd

# set backend
vd.settings.default_backend = "ipyvtklink"


def plot3d(N, radius: bool = False, **kwargs) -> vd.Plotter:
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


def _vd_tree_st_end(N: Tree_graph) -> tuple[np.array[float], np.array[float]]:
    # get all edges
    edges = N.graph.get_edges()
    # get dimensions of our spatial embedding
    dims = len(N.graph.vp["coordinates"][0])
    # initialise start and end points
    start_pts = np.zeros((edges.shape[0], dims))
    end_pts = np.zeros_like(start_pts)
    # fill up line coords
    for i in range(len(edges)):
        # current edge
        curr_edge = edges[i]
        # add to start/end points
        start_pts[i] = N.graph.vp["coordinates"][curr_edge[0]]
        end_pts[i] = N.graph.vp["coordinates"][curr_edge[1]]
    return start_pts, end_pts


def _vd_nodes_st_end(N: Node_table) -> tuple[np.array[float], np.array[float]]:
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


def _vd_tree_cyl(N: Tree_graph, **kwargs) -> List[vd.Cylinder]:
    start_pts, end_pts = _vd_tree_st_end(N)

    # create a list of cylinders
    radii = N.graph.vp["radius"].a
    # radii = radii[bool_ind]
    # create list of cylinders
    tubes = [
        vd.Cylinder(pos=[start_pts[i], end_pts[i]], r=radii[i], **kwargs)
        for i in range(start_pts.shape[0])
    ]

    return tubes


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
