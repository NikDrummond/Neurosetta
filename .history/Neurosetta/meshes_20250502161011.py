import vedo as vd
from collections import defaultdict
from graph_tool.all import triangulation
import numpy as np
from .graphs import get_g_distances
import jax.numpy as jnp
from jax.ops import segment_sum
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import trimesh
from scipy.spatial import Delaunay


def point_inside(mesh: vd.mesh.Mesh, points, invert: bool = False, **kwargs) -> np.ndarray:
    """Given a vd.Mesh object and an (n x d) set of points - 
    n being points, d being dimensions, max 3, 
    return an n X 1 boolean array of points inside the mesh.

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
        n X 1 boolean array of points inside (or outside) the mesh.
    """
    assert isinstance(mesh, vd.Mesh), "Mesh input not vd.Mesh"
    if isinstance(points, np.ndarray):
        if np.ndim(points) == 1:
            points = vd.Point(points)
        else:
            points = vd.Points(points)
    assert isinstance(points, vd.Points), "Points input not recognised - Use either an np.array or vd.Points object"

    inds = mesh.inside_points(points, invert=invert, return_ids=True, **kwargs)
    s = np.zeros(points.npoints, dtype=bool)
    s[inds] = True
    return s


def _find_triangles(edges):
    adj = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        adj[min(u, v)].add(max(u, v))

    triangles = set()
    for u in adj:
        for v in adj[u]:
            if v not in adj:
                continue
            common = adj[u].intersection(adj[v])
            for w in common:
                tri = (u, v, w)
                triangles.add(tuple(sorted(tri)))

    return np.array(list(triangles))


def triangulation(pnts, method='delaunay', threshold=True, t=None):
    g, pos = triangulation(pnts, type=method)
    g.vp['coordinates'] = g.new_vp('vector<double>')
    g.vp['coordinates'].set_2d_array(pnts.T)

    if threshold:
        get_g_distances(g, bind=True)
        if t is None:
            d = g.ep['Path_length'].a
            t = d.mean() + (3 * d.std())
        mask = np.ones(g.num_edges(), dtype=bool)
        mask[np.where(d >= t)] = 0

    mask = g.new_ep('bool', mask)
    g.set_edge_filter(mask)

    edges = g.get_edges()
    return _find_triangles(edges)


def voxel_grid_downsample(points: jnp.ndarray, voxel_size: float) -> jnp.ndarray:
    """
    Downsample a point cloud using a 3D voxel grid.

    Parameters
    ----------
    points : jnp.ndarray
        A (N, 3) array of 3D points.
    voxel_size : float
        Side length of each voxel cell.

    Returns
    -------
    jnp.ndarray
        Downsampled points (M, 3) with one point per voxel.
    """
    voxel_indices = jnp.floor(points / voxel_size).astype(jnp.int32)
    factor = jnp.array([1_000_000, 1_000, 1], dtype=jnp.int32)
    voxel_hash = jnp.dot(voxel_indices, factor)
    unique_hashes, inv = jnp.unique(voxel_hash, return_inverse=True)
    N = unique_hashes.shape[0]
    sums = segment_sum(points, inv, num_segments=N)
    counts = segment_sum(jnp.ones(points.shape[0]), inv, num_segments=N).reshape(-1, 1)
    centroids = sums / counts
    return centroids


def remove_outliers(points, k=10, quantile=95):
    """
    Remove statistical outliers based on k-nearest neighbors.

    Parameters
    ----------
    points : np.ndarray
        Input 3D point cloud.
    k : int, optional
        Number of neighbors to consider. Default is 10.
    quantile : float, optional
        Distance threshold percentile. Default is 95.

    Returns
    -------
    np.ndarray
        Filtered point cloud.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_dists = dists.mean(axis=1)
    threshold = np.percentile(mean_dists, quantile)
    return points[mean_dists < threshold]


def estimate_alpha(pnts, percentile=1):
    """
    Estimate an appropriate alpha value based on distance distribution.

    Parameters
    ----------
    pnts : np.ndarray
        Input points.
    percentile : float
        Distance percentile to use as alpha estimate.

    Returns
    -------
    float
        Estimated alpha.
    """
    dists = pdist(pnts)
    alpha_guess = np.percentile(dists, percentile)
    return alpha_guess


def compute_alpha_shape_3d(points: np.ndarray, alpha: float) -> trimesh.Trimesh:
    """
    Compute 3D alpha shape as a trimesh.Trimesh object.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of 3D points.
    alpha : float
        Alpha radius threshold.

    Returns
    -------
    trimesh.Trimesh
        The alpha shape mesh.
    """
    tetra = Delaunay(points)
    simplices = tetra.simplices

    def tet_circumradius(p):
        a, b, c, d = p
        A = np.stack([b - a, c - a, d - a])
        volume = np.abs(np.linalg.det(A)) / 6.0
        if volume == 0:
            return np.inf
        edge_lengths = np.linalg.norm(A, axis=1)
        R = (np.prod(edge_lengths)) / (6 * volume)
        return R

    mask = np.array([tet_circumradius(points[tet]) < alpha for tet in simplices])
    alpha_tetra = simplices[mask]

    faces = trimesh.grouping.group_rows(
        np.sort(
            np.vstack([
                [tet[i] for i in [0, 1, 2]] +
                [tet[i] for i in [0, 1, 3]] +
                [tet[i] for i in [0, 2, 3]] +
                [tet[i] for i in [1, 2, 3]]
                for tet in alpha_tetra
            ]), axis=1), require_count=1
    )[0]

    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    return mesh
