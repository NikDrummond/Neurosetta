import vedo as vd
import numpy as np

def point_inside(mesh:vd.mesh.Mesh,points:np.ndarray | vd.Points,invert:bool = False, **kwargs) -> bool | np.ndarray:
    

    # if points are an array, convert to vedo
    if isinstance(points,np.ndarray):
        # if single point:
        if points.shape == (3,):
            points = vd.Point(points)
        else:
            points = vd.Points(points)

    # create bool array
    s = np.zeros(points.vertices.shape[0]).astype(bool)        
    # use inside points         
    inds = mesh.inside_points(points,invert = invert, return_ids = True)
    # change returned values to True
    s[inds] = True
    if len(s) == 1:
        if s[0]:
            return True
        else:
            return False
    else:    
        return s