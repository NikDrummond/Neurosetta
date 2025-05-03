import vedo as vd
from collections import defaultdict
from graph_tool.all import triangulation as gt_triangulation
import numpy as np
from .graphs import get_g_distances
import jax.numpy as jnp
from jax.ops import segment_sum
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import trimesh
from scipy.spatial import Delaunay


def point_inside(mesh: vd.mesh.Mesh, points, invert: bool = False, **kwargs) -> np.ndarray:
    """
    Given a vd.Mesh object and an (n x d) set of points (d <= 3), return a boolean array
    indicating whether each point lies inside the mesh.

    Parameters
    ----------
    mesh : vd.Mesh
        Mesh object to check.
    points : np.ndarray or vd.Points
        Points to test.
    invert : bool, optional
        If True, return True for outside points instead.
    **kwargs : dict
        Extra arguments passed to vedo's inside_points.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) for each input point.
    """
    assert isinstance(mesh, vd.Mesh), "Mesh input not vd.Mesh"
    if isinstance(points, np.ndarray):
        if np.ndim(points) == 1:
            points = vd.Point(points)
        else:
            points = vd.Points(points)
    assert isinstance(points, vd.Points), "Points input must be np.ndarray or vd.Points"

    inds = mesh.inside_points(points, invert=invert, return_ids=True, **kwargs)
    s = np.zeros(points.npoints, dtype=bool)
    s[inds] = True
    return s


def _find_triangles(edges):
    """
    Identify all triangles from an edge list.

    Parameters
    ----------
    edges : np.ndarray
        (E, 2) array of undirected edges.

    Returns
    -------
    np.ndarray
        (T, 3) array of triangle vertex indices.
    """
    adj = defaultdict(set)
    for u, v in edges:
        if u != v:
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


def _find_tetrahedra(edges):
    """
    Efficiently extract tetrahedra (4-cliques) from an edge list.

    Parameters
    ----------
    edges : np.ndarray
        (E, 2) array of undirected edges.

    Returns
    -------
    np.ndarray
        (T, 4) array of tetrahedron vertex indices.
    """
    adj = defaultdict(set)
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    tetrahedra = set()
    for u in adj:
        nu = np.array(sorted(adj[u]))
        for i in range(len(nu)):
            v = nu[i]
            if v <= u:
                continue
            nv = np.array(sorted(adj[v]))
            common_uv = np.intersect1d(nu, nv, assume_unique=True)
            for j in range(i + 1, len(nu)):
                w = nu[j]
                if w <= v or w not in adj[v]:
                    continue
                nw = np.array(sorted(adj[w]))
                common_uvw = np.intersect1d(common_uv, nw, assume_unique=True)
                for x in common_uvw:
                    if x > w:
                        tet = tuple(sorted((u, v, w, x)))
                        tetrahedra.add(tet)

    return np.array(list(tetrahedra))


def triangulation(pnts, method='delaunay', threshold=True, t=None, return_tetra=False):
    """
    Perform graph-based Delaunay triangulation and extract triangle or tetrahedron indices.

    Parameters
    ----------
    pnts : np.ndarray
        (N, D) array of points (D=2 or 3).
    method : str
        Triangulation method. Default 'delaunay'.
    threshold : bool
        If True, apply edge length threshold.
    t : float or None
        Edge length threshold. If None, compute adaptively.
    return_tetra : bool
        If True, return tetrahedra instead of triangles.

    Returns
    -------
    np.ndarray
        (T, 3) or (T, 4) array of simplex indices.
    """
    g, pos = gt_triangulation(pnts, type=method)
    g.vp['coordinates'] = g.new_vp('vector<double>')
    g.vp['coordinates'].set_2d_array(pnts.T)

    if threshold:
        get_g_distances(g, bind=True)
        d = g.ep['Path_length'].a
        if t is None:
            t = d.mean() + (3 * d.std())
        # edge mask
        mask = np.ones(g.num_edges(), dtype=bool)
        mask[np.where(d >= t)] = 0
        e_mask = g.new_ep('bool', mask)
        # vertex mask
        mask = np.ones(g.num_vertices(), dtype = bool)
        mask[np.unique(g.get_edges()[np.where(d >= t)])] = 0
        v_mask = g.new_vp('bool', mask)
        # set filters
        g.set_filters(e_mask, v_mask)
        # purge
        g.purge_edges()
        g.purge_vertices(in_place = True)


    edges = g.get_edges()
    return _find_tetrahedra(edges) if return_tetra else _find_triangles(edges)


def voxel_grid_downsample(points: jnp.ndarray, voxel_size: float) -> jnp.ndarray:
    """
    Downsample a point cloud using a 3D voxel grid.

    Parameters
    ----------
    points : jnp.ndarray
        (N, 3) point cloud.
    voxel_size : float
        Side length of each voxel.

    Returns
    -------
    jnp.ndarray
        (M, 3) downsampled point cloud.
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
    Remove statistical outliers based on k-NN distances.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) point cloud.
    k : int
        Number of neighbors.
    quantile : float
        Distance threshold percentile.

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
    Estimate an appropriate alpha value from point distances.

    Parameters
    ----------
    pnts : np.ndarray
        (N, 3) point cloud.
    percentile : float
        Percentile for distance threshold.

    Returns
    -------
    float
        Alpha parameter.
    """
    dists = pdist(pnts)
    return np.percentile(dists, percentile)


def compute_alpha_shape_3d(points: np.ndarray, alpha: float, use_graph_tool=True) -> trimesh.Trimesh:
    """
    Compute 3D alpha shape as a trimesh.Trimesh object.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of 3D points.
    alpha : float
        Alpha radius threshold.
    use_graph_tool : bool
        If True, uses graph-tool-based triangulation from the module.

    Returns
    -------
    trimesh.Trimesh
        Alpha shape mesh.
    """
    if use_graph_tool:
        tets = triangulation(points, method='delaunay', return_tetra=True)
    else:
        from scipy.spatial import Delaunay
        tets = Delaunay(points).simplices

    def tet_circumradius(p):
        a, b, c, d = p
        A = np.stack([b - a, c - a, d - a])
        volume = np.abs(np.linalg.det(A)) / 6.0
        if volume == 0:
            return np.inf
        edge_lengths = np.linalg.norm(A, axis=1)
        return (np.prod(edge_lengths)) / (6 * volume)

    mask = np.array([tet_circumradius(points[tet]) < alpha for tet in tets])
    alpha_tetra = tets[mask]

    faces = np.concatenate([
        [tet[[0, 1, 2]], tet[[0, 1, 3]], tet[[0, 2, 3]], tet[[1, 2, 3]]]
        for tet in alpha_tetra
    ], axis=0)

    faces = trimesh.grouping.group_rows(np.sort(faces, axis=1), require_count=1)[0]
    return trimesh.Trimesh(vertices=points, faces=faces, process=True)

