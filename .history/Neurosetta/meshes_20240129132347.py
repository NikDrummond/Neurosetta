import vedo as vd
import numpy as np

def point_inside(mesh: vd.mesh.Mesh, points, invert: bool = False, **kwargs) -> np.ndarray:

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
    inds = mesh.inside_points(points, invert = invert, return_ids = True)
    # Create a boolean array of length n
    s = np.zeros(points.npoints, dtype=bool)

        # Set the corresponding elements of s to True
    s[inds] = True