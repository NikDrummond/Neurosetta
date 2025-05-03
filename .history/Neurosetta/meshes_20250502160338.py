import vedo as vd
from collections import defaultdict
from graph_tool.all import triangulation
import numpy as np
from .graphs import get_g_distances

def point_inside(mesh: vd.mesh.Mesh, points, invert: bool = False, **kwargs) -> np.ndarray:
    """Given a vd.Mesh object and an (n x d) set of points - 
    n being points, d being dimensions, max 3, 
    return an n X 1 boolian array of points inside the mesh.

    Will try to convert an np.array to a vd.Points object within the function. 

    Parameters
    ----------
    mesh : vd.mesh.Mesh
        Mesh object to check if points are inside
    points : np.ndarray | vd.point.Points
        either an (n x d) array of points or a vd.Points object representing point coordinates
    invert : bool, optional
        Changes inside / outside behaviour, by default False, so will return true if a point is inside. Setting to True 
        will return true if a point is outside.
        
    **kwargs : TYPE, optional
        Additional keyword arguments to pass to the vedo.Mesh.inside_points() function.

    Returns
    -------
    np.ndarray
        n X 1 boolian array of points inside (or outside) the mesh.
    """
    # type checks
    # check input mesh is vedo mesh
    assert isinstance(mesh, vd.Mesh), "Mesh input not vd.Mesh"
    # convert coords to vd.points object
    if isinstance(points, np.ndarray):
        # if it is one dimension convert to vd.Point
        if np.ndim(points) == 1:
            points = vd.Point(points)
        else:
            points = vd.Points(points)
    # make sure we now have a vd Points object
    assert isinstance(points,vd.Points), "Points input not recognised - Use either an np.array or vd.Points object"     

    # vd inside mesh funciton use
    inds = mesh.inside_points(points, invert = invert, return_ids = True, **kwargs)
    # Create a boolean array of length n
    s = np.zeros(points.npoints, dtype=bool)

        # Set the corresponding elements of s to True
    s[inds] = True

    return s

def _find_triangles(edges):
    # Build undirected adjacency list with sorted node indices
    adj = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue  # skip self-loops
        adj[min(u, v)].add(max(u, v))

    triangles = set()

    for u in adj:
        for v in adj[u]:
            # Ensure v > u to avoid repeats
            if v not in adj:
                continue
            # Intersect neighbors of u and v to find common neighbors (w)
            common = adj[u].intersection(adj[v])
            for w in common:
                tri = (u, v, w)
                triangles.add(tuple(sorted(tri)))

    return np.array(list(triangles))

def triangulation(pnts, method = 'delaunay', threshold = True, t = None):
    g, pos = triangulation(pnts, type = method)
    g.vp['coordinates'] = g.new_vp('vector<double>')
    g.vp['coordinates'].set_2d_array(pnts.T)
    
    if threshold:
        get_g_distances(g, bind = True)

        if t is None:
            # make threshold 
            d = g.ep['Path_length'].a
            t = d.mean() + (3 * d.std())
        
        mask = np.ones(g.num_edges(), dtype = bool)
        mask[np.where(d >= t)] = 0

    # set edge filter
    mask = g.new_ep('bool', mask)
    g.set_edge_filter(mask)

    edges = g.get_edges()
    return = _find_triangles(edges)
