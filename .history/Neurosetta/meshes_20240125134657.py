import vedo as vd
import numpy as np

def point_inside(mesh: vd.mesh.Mesh, points, invert: bool = False, **kwargs) -> np.ndarray:
    """Checks if points are inside a mesh.

    Args:
        mesh (vd.mesh.Mesh): The mesh to check.
        points: The points to check. Can be a single point, a list of points,
            or a numpy array of shape (n, 3) where n is the number of points.
        invert (bool, optional): Invert the check. Defaults to False.

    Returns:
        np.ndarray: A boolean array of length n indicating if each point is inside.
    """
    # check input mesh is vedo mesh
    assert isinstance(mesh, vd.Mesh), "Mesh input not vd.Mesh"
    # Convert points to a vedo Points object
    if isinstance(points, vd.Points):
        points_obj = points
    elif isinstance(points, np.ndarray) and points.shape[0] == 3:
        points_obj = vd.Points(points)
    elif isinstance(points, (list, tuple)) and len(points) > 0 and isinstance(points[0], (list, tuple, np.ndarray)) and len(points[0]) == 3:
        points_obj = vd.Points([vd.Point(p) for p in points])
    else:
        raise ValueError("Invalid type for points. Expected vedo Points, numpy array of shape (n, 3), list of points, or single point.")

    # Use the insidePoints method of the mesh object
    inds = mesh.inside_points(points_obj, invert=invert, return_ids=True)

    # Create a boolean array of length n
    s = np.zeros(points_obj.nPoints(), dtype=bool)

    # Set the corresponding elements of s to True
    s[inds] = True

    return s