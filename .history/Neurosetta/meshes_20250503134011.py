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
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

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

def reconstruct_surface(points: np.ndarray, depth=9, normal_radius=0.1, backend='auto', density_quantile=0.1, **kwargs) -> object:
    """
    Perform Poisson surface reconstruction using Open3D or VTK.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) input point cloud.
    depth : int
        Octree depth (higher = finer surface, Open3D only).
    normal_radius : float
        Neighborhood radius for normal estimation (Open3D).
    backend : str
        'open3d', 'vtk', or 'auto'.
    density_quantile : float
        Quantile threshold for trimming low-confidence vertices (Open3D only).
    **kwargs : dict
        Additional backend-specific options:
            - For Open3D:
                - max_nn (int): Max neighbors for normal estimation (default: 30)
                - scale (float): Bounding box scaling factor (default: 1.01)
            - For VTK:
                - iterations (int): Smoothing iterations (default: 20)
                - pass_band (float): Pass band for smoothing (default: 0.001)
                - feature_angle (float): Feature angle in degrees (default: 120.0)

    Returns
    -------
    mesh : vedo.Mesh or o3d.geometry.TriangleMesh
        Reconstructed surface mesh.
    """
    if backend == 'auto':
        if HAS_OPEN3D:
            backend = 'open3d'
        elif HAS_VTK:
            backend = 'vtk'
        else:
            raise ImportError("Neither Open3D nor VTK are available.")

    if backend == 'open3d':
        if not HAS_OPEN3D:
            raise ImportError("Open3D not installed.")

        max_nn = kwargs.get("max_nn", 30)
        scale = kwargs.get("scale", 1.01)

        # Step 1: Create and preprocess point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # Step 2: Run Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

        # Step 3: Density-based filtering
        densities = np.asarray(densities)
        threshold = np.quantile(densities, density_quantile)
        vertices_to_keep = densities > threshold
        mesh.remove_vertices_by_mask(~vertices_to_keep)

        # Step 4: Crop using axis-aligned bounding box
        bbox = pcd.get_axis_aligned_bounding_box().scale(scale, pcd.get_center())
        mesh = mesh.crop(bbox)

        # Step 5: Cleanup
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()

        # convert to trimesh
        mesh = vd.Mesh(trimesh.Trimesh(vertices = np.asarray(mesh.vertices), faces = np.asarray(mesh.triangles)))

        return mesh

    elif backend == 'vtk':
        if not HAS_VTK:
            raise ImportError("VTK not installed or does not support Poisson.")

        vtk_pts = vtk.vtkPoints()
        for p in points:
            vtk_pts.InsertNextPoint(p)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)

        normal_est = vtk.vtkPCANormalEstimation()
        normal_est.SetInputData(polydata)
        normal_est.SetSampleSize(20)
        normal_est.SetNormalOrientationToGraphTraversal()
        normal_est.Update()
        polydata = normal_est.GetOutput()

        # Surface reconstruction pipeline (not Poisson)
        surf_filter = vtk.vtkSurfaceReconstructionFilter()
        surf_filter.SetInputData(polydata)
        surf_filter.Update()

        contour_filter = vtk.vtkContourFilter()
        contour_filter.SetInputConnection(surf_filter.GetOutputPort())
        contour_filter.SetValue(0, 0.0)
        contour_filter.Update()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour_filter.GetOutputPort())
        smoother.SetNumberOfIterations(kwargs.get("iterations", 20))
        smoother.SetPassBand(kwargs.get("pass_band", 0.001))
        smoother.SetFeatureAngle(kwargs.get("feature_angle", 120.0))
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        return vd.Mesh(smoother.GetOutput()).c("lightgray").lw(0.1)

    else:
        raise ValueError(f"Unknown backend: {backend}")
