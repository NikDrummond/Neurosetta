import vedo as vd
from collections import defaultdict
from graph_tool.all import triangulation as gt_triangulation
import numpy as np
from .graphs import get_g_distances
import jax.numpy as jnp
from jax.ops import segment_sum
from jax import vmap, jit
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import trimesh
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from scipy.ndimage import binary_dilation, label
from functools import partial

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


def is_point_inside_mesh(
    mesh: vd.mesh.Mesh, query_points: np.ndarray, invert: bool = False, **kwargs
) -> np.ndarray:
    """
    Determine whether each point lies inside a given mesh.

    Parameters
    ----------
    mesh : vd.Mesh
        A Vedo mesh.
    query_points : np.ndarray
        Points to be tested.
    invert : bool, optional
        Invert inside-outside test.

    Returns
    -------
    np.ndarray
        Boolean array indicating whether each point is inside.
    """
    assert isinstance(mesh, vd.Mesh), "Mesh input must be a Vedo Mesh object."
    if isinstance(query_points, np.ndarray):
        query_points = (
            vd.Points(query_points) if query_points.ndim > 1 else vd.Point(query_points)
        )
    assert isinstance(
        query_points, vd.Points
    ), "Query points must be a numpy array or vd.Points."
    indices_inside = mesh.inside_points(
        query_points, invert=invert, return_ids=True, **kwargs
    )
    result = np.zeros(query_points.npoints, dtype=bool)
    result[indices_inside] = True
    return result


def extract_triangles_from_edges(edges: np.ndarray) -> np.ndarray:
    """
    Extract triangles from a set of edges by finding three mutually connected vertices.

    Parameters
    ----------
    edges : np.ndarray
        Array of edge pairs.

    Returns
    -------
    np.ndarray
        Array of triangle vertex indices.
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
                triangles.add(tuple(sorted((u, v, w))))
    return np.array(list(triangles))


def extract_tetrahedra_from_edges(edges: np.ndarray) -> np.ndarray:
    """
    Identify tetrahedra by searching for four mutually connected vertices in edge set.

    Parameters
    ----------
    edges : np.ndarray
        Array of edge pairs.

    Returns
    -------
    np.ndarray
        Array of tetrahedron vertex indices.
    """
    adj = defaultdict(set)
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    tets = set()
    for u in adj:
        neighbors_u = sorted(adj[u])
        for i, v in enumerate(neighbors_u):
            if v <= u:
                continue
            shared_uv = set(neighbors_u).intersection(adj[v])
            for j in range(i + 1, len(neighbors_u)):
                w = neighbors_u[j]
                if w <= v or w not in adj[v]:
                    continue
                shared_uvw = shared_uv.intersection(adj[w])
                for x in shared_uvw:
                    if x > w:
                        tets.add(tuple(sorted((u, v, w, x))))
    return np.array(list(tets))


def perform_graph_triangulation(
    points: np.ndarray,
    method: str = "delaunay",
    threshold: bool = True,
    distance_threshold: float = None,
    return_tetrahedra: bool = False,
) -> np.ndarray:
    """
    Perform triangulation using graph-tool with optional edge filtering.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    method : str
        Triangulation type.
    threshold : bool
        Apply edge length thresholding.
    distance_threshold : float or None
        Edge length threshold (inferred if None).
    return_tetrahedra : bool
        Whether to return tetrahedra or triangles.

    Returns
    -------
    np.ndarray
        Extracted triangles or tetrahedra.
    """
    g, _ = gt_triangulation(points, type=method)
    g.vp["coordinates"] = g.new_vp("vector<double>")
    g.vp["coordinates"].set_2d_array(points.T)

    if threshold:
        get_g_distances(g, bind=True)
        d = g.ep["Path_length"].a
        if distance_threshold is None:
            distance_threshold = d.mean() + 3 * d.std()
        edge_mask = g.new_ep("bool", d < distance_threshold)
        vertex_mask = g.new_vp("bool", np.ones(g.num_vertices(), dtype=bool))
        vertex_mask.a[np.unique(g.get_edges()[d >= distance_threshold])] = 0
        g.set_filters(edge_mask, vertex_mask)
        g.purge_edges()
        g.purge_vertices(in_place=True)

    edges = g.get_edges()
    return (
        extract_tetrahedra_from_edges(edges)
        if return_tetrahedra
        else extract_triangles_from_edges(edges)
    )


def downsample_points_voxel_grid(points: jnp.ndarray, voxel_size: float) -> jnp.ndarray:
    """
    Downsample points using voxel grid averaging.

    Parameters
    ----------
    points : jnp.ndarray
        Input points.
    voxel_size : float
        Size of each voxel.

    Returns
    -------
    jnp.ndarray
        Downsampled points.
    """
    voxel_indices = jnp.floor(points / voxel_size).astype(jnp.int32)
    factor = jnp.array([1_000_000, 1_000, 1], dtype=jnp.int32)
    voxel_hash = jnp.dot(voxel_indices, factor)
    unique_hashes, inv = jnp.unique(voxel_hash, return_inverse=True)
    N = unique_hashes.shape[0]
    sums = segment_sum(points, inv, num_segments=N)
    counts = segment_sum(jnp.ones(points.shape[0]), inv, num_segments=N).reshape(-1, 1)
    return sums / counts


def remove_point_cloud_outliers(
    points: np.ndarray, k: int = 10, quantile: int = 95
) -> np.ndarray:
    """
    Remove outliers in a point cloud using local neighborhood distances.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    k : int
        Number of neighbors for distance computation.
    quantile : int
        Distance threshold quantile.

    Returns
    -------
    np.ndarray
        Filtered point cloud.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_dists = dists.mean(axis=1)
    return points[mean_dists < np.percentile(mean_dists, quantile)]


def estimate_alpha_threshold(
    points: np.ndarray,
    method: str = "triangle",
    min_percentile: int = 2,
    max_percentile: int = 20,
    plot: bool = False,
) -> float:
    """
    Estimate a suitable alpha threshold for surface reconstruction.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    method : str
        Method to compute alpha ('mean_std', 'percentile', 'triangle').
    min_percentile : int
        Minimum percentile for 'triangle' method.
    max_percentile : int
        Maximum percentile for 'triangle' method.
    plot : bool
        Whether to plot the sorted distances and selected alpha.

    Returns
    -------
    float
        Estimated alpha value.
    """
    distances = pdist(points)
    sorted_dists = np.sort(distances)

    if method == "mean_std":
        return np.mean(distances) + 0.5 * np.std(distances)
    elif method == "percentile":
        return np.percentile(distances, 10)
    elif method == "triangle":
        x = np.linspace(min_percentile, max_percentile, 100)
        y = np.percentile(sorted_dists, x)
        p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])

        def dist_to_line(p):
            return np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)

        dists = np.array([dist_to_line(np.array([xi, yi])) for xi, yi in zip(x, y)])
        alpha = y[np.argmax(dists)]

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 3))
            plt.plot(sorted_dists, label="Sorted distances")
            plt.axhline(
                alpha, color="red", linestyle="--", label=f"Alpha = {alpha:.4f}"
            )
            plt.legend(), plt.tight_layout(), plt.show()

        return alpha
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_compute_tetra_circumradius(
    points: jnp.ndarray, tets: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute circumradius for a batch of tetrahedra using JAX.

    Parameters
    ----------
    points : jnp.ndarray
        Point cloud.
    tets : jnp.ndarray
        Indices for tetrahedra.

    Returns
    -------
    jnp.ndarray
        Circumradius per tetrahedron.
    """

    def single_radius(idx):
        a, b, c, d = points[idx[0]], points[idx[1]], points[idx[2]], points[idx[3]]
        A = jnp.linalg.norm(b - a)
        B = jnp.linalg.norm(c - a)
        C = jnp.linalg.norm(d - a)
        D = jnp.linalg.norm(c - b)
        E = jnp.linalg.norm(d - b)
        F = jnp.linalg.norm(d - c)
        matrix = jnp.array(
            [
                [0, A**2, B**2, C**2, 1],
                [A**2, 0, D**2, E**2, 1],
                [B**2, D**2, 0, F**2, 1],
                [C**2, E**2, F**2, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        )
        det = jnp.linalg.det(matrix)
        volume = jnp.abs(jnp.linalg.det(jnp.stack([b - a, c - a, d - a]))) / 6.0
        return jnp.where(
            det <= 0, jnp.inf, (A * B * C * D * E * F) ** 0.5 / (24 * volume)
        )

    return vmap(single_radius)(tets)


def compute_alpha_shape(
    points: np.ndarray,
    alpha: float = None,
    auto_tune: bool = True,
    verbose: bool = True,
    plot: bool = False,
    strategy: str = "watertight",
) -> trimesh.Trimesh:
    """
    Compute a watertight alpha shape mesh from 3D points.

    Parameters
    ----------
    points : np.ndarray
        Input 3D point cloud.
    alpha : float or None
        Optional alpha value. Auto-selected if None.
    auto_tune : bool
        Auto-tune alpha based on mesh characteristics.
    verbose : bool
        Display status messages.
    plot : bool
        Plot histogram of circumradii.
    strategy : str
        Strategy to optimize ('watertight' or 'tetra_count').

    Returns
    -------
    trimesh.Trimesh
        Watertight surface mesh.
    """
    tets = Delaunay(points).simplices
    if len(tets) == 0:
        raise ValueError("No tetrahedra could be formed from the input points.")

    j_points = jnp.array(points)
    j_tets = jnp.array(tets)
    raw_radii = batch_compute_tetra_circumradius(j_points, j_tets)

    finite_mask = jnp.isfinite(raw_radii)
    filtered_radii = raw_radii[finite_mask]
    filtered_tets = tets[np.asarray(finite_mask)]

    if len(filtered_radii) == 0:
        raise ValueError("All circumradii were infinite or invalid.")

    if alpha is None and auto_tune:
        r_np = np.asarray(filtered_radii)
        iqr = np.percentile(r_np, 75) - np.percentile(r_np, 25)
        max_val = np.percentile(r_np, 75) + 1.5 * iqr
        sweep = r_np[r_np < max_val]
        candidates = np.linspace(np.percentile(sweep, 20), np.percentile(sweep, 80), 20)

        best_alpha = None
        best_score = -np.inf

        for a in candidates:
            mask = r_np < a
            temp_tets = filtered_tets[mask]
            if len(temp_tets) == 0:
                continue
            faces = np.concatenate(
                [
                    [tet[[0, 1, 2]], tet[[0, 1, 3]], tet[[0, 2, 3]], tet[[1, 2, 3]]]
                    for tet in temp_tets
                ],
                axis=0,
            )
            sorted_faces = np.sort(faces, axis=1)
            unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
            boundary_faces = unique_faces[counts == 1]
            score = (
                len(boundary_faces) / len(temp_tets)
                if strategy == "watertight"
                else len(temp_tets)
            )
            if score > best_score:
                best_score = score
                best_alpha = a

        alpha = best_alpha
        if verbose:
            print(f"🔹 Auto-selected alpha: {alpha:.4f}")

    mask = filtered_radii < alpha
    alpha_tetra = filtered_tets[np.asarray(mask)]
    if alpha_tetra.size == 0:
        raise ValueError("No tetrahedra passed the alpha threshold.")

    faces = np.concatenate(
        [
            [tet[[0, 1, 2]], tet[[0, 1, 3]], tet[[0, 2, 3]], tet[[1, 2, 3]]]
            for tet in alpha_tetra
        ],
        axis=0,
    )
    sorted_faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]

    if len(boundary_faces) == 0:
        raise ValueError("Alpha shape produced no surface faces.")

    mesh = trimesh.Trimesh(vertices=points, faces=boundary_faces, process=True)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("Resulting mesh has no faces or area.")

    return mesh


def reconstruct_surface_poisson(
    points: np.ndarray,
    depth: int = 9,
    normal_radius: float = 0.1,
    density_quantile: float = 0.1,
    max_nn: int = 30,
    scale: float = 1.01,
) -> trimesh.Trimesh:
    """
    Surface reconstruction using Poisson reconstruction from Open3D.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    depth : int
        Octree depth for Poisson reconstruction.
    normal_radius : float
        Radius for normal estimation.
    density_quantile : float
        Quantile to filter low-density vertices.
    max_nn : int
        Max neighbors for normal estimation.
    scale : float
        Scale for bounding box cropping.

    Returns
    -------
    trimesh.Trimesh
        Cleaned reconstructed mesh.
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is not installed.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(~keep)

    bbox = pcd.get_axis_aligned_bounding_box().scale(scale, pcd.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()

    return vd.Mesh(trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=True,
    )


def generate_voxel_grid(
    points: np.ndarray,
    voxel_size: float = 1.0,
    padding: int = 2,
    dilate_iter: int = 1,
    keep_largest: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a voxel grid from 3D points.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    voxel_size : float
        Edge length of each voxel.
    padding : int
        Number of voxel units to pad the bounding box.
    dilate_iter : int
        Number of dilation iterations.
    keep_largest : bool
        Keep only the largest connected component.

    Returns
    -------
    tuple
        (voxel grid, grid offset, voxel size)
    """
    min_corner = points.min(axis=0) - voxel_size * padding
    max_corner = points.max(axis=0) + voxel_size * padding
    dims = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
    offset = min_corner

    indices = np.floor((points - offset) / voxel_size).astype(int)
    indices = indices[(indices >= 0).all(axis=1) & (indices < dims).all(axis=1)]

    grid = np.zeros(dims, dtype=bool)
    grid[tuple(indices.T)] = True

    if dilate_iter > 0:
        grid = binary_dilation(grid, iterations=dilate_iter)

    if keep_largest:
        labeled, num = label(grid)
        if num == 0:
            raise ValueError("No connected components found.")
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        grid = labeled == sizes.argmax()

    return grid, offset, voxel_size


def surface_from_voxel_grid(
    grid: np.ndarray, offset: np.ndarray, voxel_size: float = 1.0
) -> vd.Mesh:
    """
    Generate a mesh from a binary voxel grid using marching cubes.

    Parameters
    ----------
    grid : np.ndarray
        3D binary voxel grid.
    offset : np.ndarray
        Origin of the voxel grid.
    voxel_size : float
        Size of each voxel.

    Returns
    -------
    trimesh.Trimesh
        Surface mesh.
    """
    verts, faces, _, _ = marching_cubes(
        grid.astype(np.float32), level=0.5, spacing=(voxel_size,) * 3
    )
    verts += offset
    return vd.Mesh(trimesh.Trimesh(vertices=verts, faces=faces, process=True))


def clean_mesh(mesh: trimesh.Trimesh, voxel_size: float = None) -> vd.Mesh:
    """
    Clean mesh by removing degenerate elements and optionally revoxelizing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    voxel_size : float or None
        If not watertight, revoxelize at this resolution.

    Returns
    -------
    trimesh.Trimesh
        Cleaned, watertight mesh.
    """
    mesh.fill_holes()
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("Mesh split returned no components.")
    mesh = max(components, key=lambda m: m.volume)
    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight and voxel_size:
        vox = mesh.voxelized(pitch=voxel_size)
        mesh = vox.marching_cubes

    mesh.process(validate=False)
    if mesh.volume < 0:
        mesh.invert()
    return vd.Mesh(mesh)


def reconstruct_surface_voxel(
    points: np.ndarray,
    voxel_size: float = 1.0,
    padding: int = 2,
    dilate: int = 1,
    largest_component: bool = False,
    clean: bool = True,
) -> vd.Mesh:
    """
    Reconstruct surface via voxelization and marching cubes.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    voxel_size : float
        Voxel pitch.
    padding : int
        Padding in voxel units.
    dilate : int
        Number of dilation steps.
    largest_component : bool
        Keep only largest blob.
    clean : bool
        Post-process the mesh.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.
    """
    grid, offset, voxel_size = generate_voxel_grid(
        points, voxel_size, padding, dilate, largest_component
    )
    mesh = surface_from_voxel_grid(grid, offset, voxel_size)
    mesh = clean_mesh(mesh, voxel_size) if clean else mesh
    mesh = vd.Mesh(mesh) 
    return 


def reconstruct_surface(
    points: np.ndarray, method: str = "alpha", **kwargs
) -> trimesh.Trimesh:
    """
    General-purpose wrapper for reconstructing surfaces from 3D point clouds.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3).
    method : str
        Reconstruction method to use. Options:
        - 'alpha': Alpha shape surface reconstruction.
            - alpha (float, optional): Fixed alpha value. Auto-tuned if None.
            - auto_tune (bool): Whether to optimize alpha automatically.
            - verbose (bool): Print tuning information.
            - plot (bool): Show radius histogram.
            - strategy (str): 'watertight' (default) or 'tetra_count'.
        - 'poisson': Poisson surface reconstruction using Open3D.
            - depth (int): Octree depth for reconstruction (default 9).
            - normal_radius (float): Radius for normal estimation.
            - density_quantile (float): Density threshold for filtering low-density vertices.
            - max_nn (int): Max neighbors for KD-tree.
            - scale (float): Bounding box scale factor.
        - 'voxel': Voxelization and marching cubes.
            - voxel_size (float): Size of each voxel.
            - padding (int): Voxel padding around bounds.
            - dilate (int): Number of dilation iterations.
            - largest_component (bool): Keep only largest component.
            - clean (bool): Postprocess and watertight the mesh.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.
    """
    method = method.lower()
    if method == "alpha":
        return compute_alpha_shape(points, **kwargs)
    elif method == "poisson":
        return reconstruct_surface_poisson(points, **kwargs)
    elif method == "voxel":
        return reconstruct_surface_voxel(points, **kwargs)
    else:
        raise ValueError(f"Unknown reconstruction method: '{method}'")


MAX_DDA_STEPS = 512


@jit
def dda_fixed_steps(start, end, offset, voxel_size, dims):
    start_vox = jnp.floor((start - offset) / voxel_size).astype(jnp.int32)
    end_vox = jnp.floor((end - offset) / voxel_size).astype(jnp.int32)
    direction = end - start
    step = jnp.sign(direction).astype(jnp.int32)

    def safe_div(a, b):
        return jnp.where(b != 0, a / b, jnp.inf)

    t_max = safe_div(
        ((start_vox + (step > 0)) * voxel_size + offset - start), direction
    )
    t_delta = safe_div(voxel_size, jnp.abs(direction))

    def body_fn(carry, _):
        curr_voxel, t_max, step, t_delta = carry
        axis = jnp.argmin(t_max)
        next_voxel = curr_voxel.at[axis].add(step[axis])
        new_t_max = t_max.at[axis].add(t_delta[axis])
        return (next_voxel, new_t_max, step, t_delta), curr_voxel

    init = (start_vox, t_max, step, t_delta)
    (_, _, _, _), voxels = jax.lax.scan(body_fn, init, None, length=MAX_DDA_STEPS)
    voxels = jnp.vstack([start_vox[None, :], voxels])

    total_steps = jnp.minimum(
        MAX_DDA_STEPS + 1, jnp.sum(jnp.abs(end_vox - start_vox)) + 1
    )
    mask = jnp.arange(MAX_DDA_STEPS + 1) < total_steps
    return voxels, mask


@partial(jit, static_argnums=(4,))
def trace_lines_to_voxels(starts, ends, offset, voxel_size, dims_tuple):
    """
    Given batch of lines (starts, ends), compute per-voxel intersection counts.
    dims_tuple: Python tuple of grid dimensions (nz, ny, nx), static.
    """
    nz, ny, nx = dims_tuple
    dims = jnp.array(dims_tuple, dtype=jnp.int32)
    num_segments = nz * ny * nx

    trace_fn = lambda s, e: dda_fixed_steps(s, e, offset, voxel_size, dims)
    voxels_all, masks_all = jax.vmap(trace_fn)(starts, ends)

    voxels_flat = voxels_all.reshape(-1, 3)
    masks_flat = masks_all.reshape(-1)

    indices = voxels_flat[:, 0] * ny * nx + voxels_flat[:, 1] * nx + voxels_flat[:, 2]

    weights = masks_flat.astype(jnp.int32)
    counts = segment_sum(weights, indices, num_segments=num_segments)
    return counts.reshape((nz, ny, nx))


def voxel_line_intersections(
    grid: np.ndarray,
    offset: np.ndarray,
    voxel_size: float,
    line_starts: np.ndarray,
    line_ends: np.ndarray,
) -> np.ndarray:
    """
    Count number of line segments intersecting each voxel in the grid.

    Parameters
    ----------
    grid : np.ndarray
        3D binary voxel grid.
    offset : np.ndarray
        Origin of the voxel grid.
    voxel_size : float
        Size of each voxel.
    line_starts : np.ndarray
        Array of shape (N, 3) with line start points.
    line_ends : np.ndarray
        Array of shape (N, 3) with line end points.

    Returns
    -------
    np.ndarray
        Grid of shape like `grid` with counts per voxel.
    """
    starts = jnp.array(line_starts, dtype=jnp.float32)
    ends = jnp.array(line_ends, dtype=jnp.float32)
    offset_j = jnp.array(offset, dtype=jnp.float32)
    dims_tuple = tuple(grid.shape)
    return trace_lines_to_voxels(starts, ends, offset_j, voxel_size, dims_tuple)
