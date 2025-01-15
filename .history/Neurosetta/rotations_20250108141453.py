import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.spatial.transform import Rotation as R
import vg

from .core import Tree_graph
from .graphs import g_root_ind, g_vert_coords, g_root_ind

### Functions

# these will be constants for several function inputs
base_x = np.array([1,0,0], dtype = float)
base_y = np.array([0,1,0], dtype = float)
base_z = np.array([0,0,1], dtype = float)

axis_basis = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype = float)

# eigen decomposition of coordinates
def coord_eig_decomp(coords, robust = True,  center = False, PCA = True, sort = True, transpose = True):
    """_summary_

    Parameters
    ----------
    coords : _type_
        _description_
    center : bool, optional
        _description_, by default True
    PCA : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    # if we want to center
    if center:
        coords -= coords.mean(axis = 0)

    # robust covarience or not
    if robust:
        cov = MinCovDet().fit(coords).covariance_
        
    else:
        cov = EmpiricalCovariance().fit(coords).covariance_
    
    # eigen decomposition
    evals, evecs = np.linalg.eig(cov)

    # if we want variance explained (PCA)
    if PCA:
        evals /= evals.sum()

    # if we want to sort out eigenvectors and values going highest to lowest
    if sort:
        # sort indicies
        sort_inds = np.argsort(evals)[::-1]
        evals = evals[sort_inds] 
        evecs = evecs[:,sort_inds]
    
    # columns are currently eigenvectors, sometimes having rows is easier, so if we want to transpose the output
    if transpose:
        evecs = evecs.T
    
    return evals, evecs

def minimum_theta(v1,v2,plane,units = 'rad'):
    
    # make unit vectors
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    # not sure if I this to be a unit vector as well, but can't hurt
    plane /= np.linalg.norm(plane)

    # angle between v1 and v2 from perspective of plane
    theta = vg.signed_angle(v1,v2,look = plane, units = 'rad')

    # correct theta to ensure minimal rotation
    if (theta > np.pi / 2):
        theta = theta - np.pi
    elif (theta < - np.pi / 2):
        theta = np.pi - abs(theta)

    # convert to degrees if requested
    if units == 'deg':
        theta = np.degrees(theta)

    return theta

def rotate_around_axis(coords,theta,axis):
    
    # generate rotation vector
    rot_vec = axis * theta
    # initialise rotation
    rot = R.from_rotvec(rot_vec)
    # apply
    coords = rot.apply(coords)
    return coords

def align_eigen_single(coords,evecs,evec_ind,axis,perspective):
    
    # if string is given for axis, make array
    if isinstance(axis, str):
        if axis == 'x':
            axis = np.array([1,0,0], dtype = float)
        elif axis == 'y':
            axis = np.array([0,1,0], dtype = float)
        elif axis == 'z':
            axis = np.array([0,0,1], dtype = float)
        else:
            raise AttributeError('If string, axis must be x, y, or z')
    
    # if string is given for persepective, make it an array
    if isinstance(perspective, str):
        if perspective == 'x':
            perspective = np.array([1,0,0], dtype = float)
        elif perspective == 'y':
            perspective = np.array([0,1,0], dtype = float)
        elif perspective == 'z':
            perspective = np.array([0,0,1], dtype = float)
        else:
            raise AttributeError('If string, perspective must be x, y, or z')
        
    # get theta
    theta = minimum_theta(evecs[evec_ind], axis, perspective)
    # do rotation
    coords_r = rotate_around_axis(coords,theta,perspective)
    evecs_r = rotate_around_axis(evecs,theta,perspective)

    return coords_r,evecs_r

def eig_align(N, eig_order, align_order, perspective_order, center = 'root', robust = True, bind = True):


    # unpack align order is list
    if isinstance(align_order, list):
        # if we have been given a list of strings:
        if isinstance(align_order[0], str):
            align_axis = np.zeros_like(axis_basis)
            for i in range(len(align_axis)):
                if align_order[i] == 'x':
                    align_axis[i] = axis_basis[0]
                elif align_order[i] == 'y':
                    align_axis[i] = axis_basis[1]
                elif align_order[i] == 'z':
                    align_axis[i] = axis_basis[2]
        elif isinstance(align_order[0], int):
            align_axis = axis_basis[align_order]
        else:
            raise AttributeError('If list indices, align_order must be a combination of [x,y,z] or [0,1,2]')
    elif isinstance(align_order, np.ndarray):
        assert align_order.shape == (3,3), 'Passed array for axis order must be 3 by 3'
        align_axis = align_order

    # do the same for perspective_order
    if isinstance(perspective_order, list):
        # if we have been given a list of strings:
        if isinstance(perspective_order[0], str):
            perspective_axis = np.zeros_like(axis_basis)
            for i in range(len(align_axis)):
                if perspective_order[i] == 'x':
                    perspective_axis[i] = axis_basis[0]
                elif perspective_order[i] == 'y':
                    perspective_axis[i] = axis_basis[1]
                elif perspective_order[i] == 'z':
                    perspective_axis[i] = axis_basis[2]
        elif isinstance(perspective_order[0], int):
            perspective_axis = axis_basis[perspective_order]
        else:
            raise AttributeError('If list indices, align_order must be a combination of [x,y,z] or [0,1,2]')
    elif isinstance(perspective_order, np.ndarray):
        assert perspective_order.shape == (3,3), 'Passed array for axis order must be 3 by 3'
        perspective_axis = perspective_order


    ### get coordinates and center if wanted
    if isinstance(N, Tree_graph):
        coords = nr.g_vert_coords(N)
        if center == None:
            pass
        elif center == 'root':
            r_coord = nr.g_vert_coords(N,g_root_ind(N))[0]
            coords -= r_coord
        elif center == 'mean':
            coords -= coords.mean(axis = 0)
        elif center == 'median':
            coords -= np.median(coords, axis = 0)
        else:
            raise AttributeError('center argument must be None, root, mean, or median when input Neuron is given')
    elif isinstance(N, np.ndarray):
        coords = N
        # nothing to bind to so make sure it is false
        bind = False
        if center == None:
            pass
        elif center == 'mean':
            coords -= coords.mean(axis = 0)
        elif center == 'median':
            coords -= np.median(coords, axis = 0)
        else:
            raise AttributeError('center argument must be None, mean, or median when coordinate array is given')



    ### initial eigen decomposition
    _, evecs = coord_eig_decomp(coords, robust = robust)

    ### first rotation - rotate around perspective_order[0] to align eig_order[0] to align_order[0]
    coords_r,evecs_r = align_eigen_single(coords,evecs,eig_order[0],align_axis[0],perspective_axis[0])

    ### second rotation - rotate around perspective_order[1] to align eig_order[1] to align_order[1]
    coords_r,evecs_r = align_eigen_single(coords_r,evecs_r,eig_order[1],align_axis[1],perspective_axis[1])

    ### third rotation - rotate around perspective_order[2] to align eig_order[2] to align_order[2]
    coords_r,evecs_r = align_eigen_single(coords_r,evecs_r,eig_order[2],align_axis[2],perspective_axis[2])

    if bind:
        N.graph.vp['coordinates'].set_2d_array(coords_r.T)
    else:
        return coords_r

