from .core import *

import vedo as vd

# set backend
vd.settings.default_backend= 'ipyvtklink'

def plot3d(N, radius = False):
    """
    simple 3d plot function using vedo
    """

    # check the input type
    if isinstance(N, Tree_graph):
        
        # get all edges
        edges = N.graph.get_edges()
        # get dimensions of our spatial embedding
        dims = len(N.graph.vp['coordinates'][0])
        # initialise start and end points
        start_pts = np.zeros((edges.shape[0],dims))
        end_pts = np.zeros_like(start_pts)
        # fill up line coords
        for i in range(len(edges)):
            # current edge
            curr_edge = edges[i]
            # add to start/end points
            start_pts[i] = N.graph.vp['coordinates'][curr_edge[0]]
            end_pts[i] = N.graph.vp['coordinates'][curr_edge[1]]

        # if we are plotting radius
        if radius == True:

            # create a list of cylinders
            radii = N.graph.vp['radius'].a
            #radii = radii[bool_ind]
            # create list of cylinders
            tubes = [vd.Cylinder(pos = [start_pts[i],end_pts[i]],r = radii[i]) for i in range(start_pts.shape[0])]

            vd.show(tubes).close()

        else:
            # collection of lines for plotting
            lns = vd.Lines(start_pts = start_pts,
                            end_pts = end_pts)
            vd.show(lns).close()
            
        
    elif isinstance(N,Node_table):
        # node ids
        node_ids = N.nodes['node_id'].values
        # this by node ids - np.array
        coords = N.nodes[['x','y','z']].values
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
        parent_coords = coords[parent_inds]
        # set boolian ind to get children
        bool_ind = np.ones_like(node_ids,bool)
        # set root to false
        bool_ind[root_ind] = False
        # remove root from child coords
        child_coords = coords[bool_ind]
        
        
        if radius == True:
            # create a list of cylinders
            radii = N.nodes.radius.values
            radii = radii[bool_ind]
            # create list of cylinders
            tubes = [vd.Cylinder(pos = [child_coords[i],parent_coords[i]],r = radii[i]) for i in range(parent_coords.shape[0])]

            vd.show(tubes).close()
            
        else:
            # collection of lines for plotting
            lns = vd.Lines(start_pts = parent_coords,
                            end_pts = child_coords)
            vd.show(lns).close()

    elif isinstance(N, Neuron_mesh):
        # generate vedo mesh
        mesh = vd.Mesh([N.vertices,N.faces])
        # plot
        vd.show(mesh).close()
