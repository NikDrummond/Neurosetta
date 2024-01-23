import vedo as vd

def point_inside(mesh:vd.mesh.Mesh,points:np.ndarray | vd.Points,invert:bool = False, **kwargs) -> bool | np.ndarray:
    """
    Bool 3D point(s) inside mesh
    """

    # if points are an array, convert to vedo
    if isinstance(points,np.ndarray):
        # if single point:
        if points.shape == (3,):
            points = vd.Point(points)
        else:
            points = vd.Points(points)

    # if single point
    if points.vertices.shape == (3,):
        # use inside points to check if it is inside. only return the size of the returned point array  
        s = mesh.inside_points(points,invert = invert **kwargs).vertices.size
        if s != 0 :
            return True
        else:
            return True
    # if we have multiple points
    else:
        # create bool array
        s = np.zeros(points.vertices.shape[0]).astype(bool)        
        # use inside points         
        inds = mesh.inside_points(points,invert = invert, return_ids = True)
        # change returned values to True
        s[inds] = True
        return s