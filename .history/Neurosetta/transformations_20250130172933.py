import numpy as np
from .core import Tree_graph, g_has_property
from .graphs import g_vert_coords, g_has_property

NoneType = type(None)

def rotation_matrix_3D(theta1, theta2, theta3, order="xyz"):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z: e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )

    return matrix


def coords_Eig(
    Coords,
    center=False,
    PCA=False,
):
    """
    Performs Eigen Decomposition on a given array of coordinates. This is done by calculating the covariance matrix of the coordinates array, then the eigenvectors and values of this coordinate matrix.
    Parameters
    ----------

    Coords:         np.array
        Coordinate array from which to calculate eigenvectors and values. This should be in the form n x d, where each row is an observation, and column a dimension.
    center:         Bool
        Whether or not to mean center the data before computing
    PCA:            Bool
        Whether or not to return eigenvalues as a fraction of the sum of all eigenvalues, as in PCA, showing variance explained.
    Returns
    -------
    evals:          np.array
        eigenvalues, ordered from largest to smallest
    evects:         list
        List of np.arrays, each of witch is the eigenvector corresponding to the descending order of eigenvalues

    """

    # check dimensions of input
    if 3 not in Coords.shape:
        raise AttributeError("Input coordinates are not 3D")
    else:
        if Coords.shape[0] != 3:
            Coords = Coords.T

    # mean center the data, if we want
    if center:
        Coords = Coords - np.mean(Coords, axis = 0)

    cov_mat = np.cov(Coords)
    evals, evects = np.linalg.eig(cov_mat)
    # sort largest to smallest
    sort_inds = np.argsort(evals)[::-1]

    if PCA:
        evals /= np.sum(evals)

    # evects = [evects[:, i] for i in sort_inds]
    evects = evects[:,sort_inds]

    return evals[sort_inds], evects


def eig_axis_eulers(evects):
    """
    Given a list of eigenvector as returned by coords_Eig, return Euler angles needed to align The first eigenvector with the y-axis, second with the x-axis, and third with the z-axis
    """
    # Yaw
    theta1 = np.rad2deg(np.arctan(evects[0][0] / evects[0][1]))
    # pitch
    theta2 = np.rad2deg(np.arctan(evects[1][2] / evects[1][0]))
    # roll
    theta3 = np.rad2deg(np.arctan(evects[2][1] / evects[2][2]))
    # Yaw
    # theta1 = np.degrees(np.arctan2(evects[0, 0], evects[1, 0]))  # Aligns first eigenvector (v1) with y
    # # Roll
    # theta2 = np.degrees(np.arctan2(evects[2, 1], evects[0, 1]))  # Aligns second eigenvector (v2) with x
    # # Pitch
    # theta3 = np.degrees(np.arctan2(evects[1, 2], evects[2, 2]))  # Aligns third eigenvector (v3) with z

    return theta1, theta2, theta3


def snap_to_axis(coords, error_tol=0.0000001, return_theta=False):
    """
    Given a set of 3D coordinates, rotates the coordinates so the Eigenvectors align with the original coordinate system axis. This is done so the first Eigenvector (corresponding to the highest eigenvalue) aligns with the y-axis, the second to the x-axis, and the third to the z-axis. Rotation is done in 'zyx' order, Rotating first around the z-axis to align the first eigenvector to the y-axis (this is Yaw), Second around the y-axis to align the second eigenvector to the x-axis (Pitch), and finally around the x-axis to align the final eigenvector to the z-axis (roll).

    Parameters
    ----------
    coords:         np.array
        dimensions by observations np.array with coordinates. Function can only accept 3D coordinates currently
    error_tol:      float
        Some error around how closely the eigenvectors can align to the image axis seems to be introduced (at this stage, it is unclear why this is the case...). The error_tol parameter sets a threshold where by the rotation will be interatively re-rotated, new eigenvectors calculated, and euler angles calculated, until the euler angles are less than this threshold.
    return_theta:   Bool
        If True, final euler angles are returned, which will be less than error_tol.
    Returns
    -------
    r_coords:       np.array
        Rotated coordinate array, the same shape as the input coordinate array
    thetas:         list
        list of euler angles of the final rotation, which will be less than error_tol. Angles are ordered in the order of rotations.



    """
    # make sure coords are the right shape:
    if 3 in coords.shape:
        if coords.shape[0] != 3:
            coords = coords.T
    else:
        raise AttributeError("Input coordinates are not 3 dimensional")
    ### Rotation - Yaw, Pitch, and Roll
    evals, evects = coords_Eig(coords)

    theta1, theta2, theta3 = eig_axis_eulers(evects)

    R = rotation_matrix_3D(theta1, theta2, theta3, order="zyx")
    r_coords = R @ coords

    # Check and correct for error
    # get "final" angles
    evals, evects = coords_Eig(r_coords)

    # original Thetas
    theta1_o, theta2_o, theta3_o = eig_axis_eulers(evects)

    theta1 = theta1_o
    theta2 = theta2_o
    theta3 = theta3_o

    while (
        (abs(theta1) > error_tol)
        and (abs(theta2) > error_tol)
        and (abs(theta3) > error_tol)
    ):

        evals, evects = coords_Eig(r_coords)
        # pitch
        theta1, theta2, theta3 = eig_axis_eulers(evects)

        R = rotation_matrix_3D(theta1, theta2, theta3, order="zyx")
        # update thetas
        theta1_o += theta1
        theta2_o += theta2
        theta3_o += theta3

        r_coords = R @ r_coords

    if return_theta == False:
        return r_coords
    else:
        return r_coords, [theta1_o, theta2_o, theta3_o]
    
def coordinate_scale(coords:np.ndarray, s:float | np.ndarray, invert = True):
    """scale the given coordinates linearly along each axis. This is just multiplying the coordinates by the diagonal matrix given by s
    In the case where s is a single value, each axis is scaled by x, otherwise s  must be an array of equal length to the last dimension 
    of the coordinates.

    The scaling matrix is simply the identity multiplied by s (so a diagonal matrix with s along the diagonal)

    If Invert is True, the scaling matrix is inverted, so the coordinates are scaled by 1/s

    Importantly, we scale each axis independently. As such, using the 'snap_to_axis' function before hand,
    which aligns coordinates to the eigenvectors is the most suitable preprocessing step
    
    Parameters
    ----------
    coords : np.ndarray
        n by d input coordinates
    s : float | np.ndarray
        either single value , or vector of values equal in length to last dimension of given coordinates
    invert: bool
        Whether or not to invert the scaling matrix. If True, coordinates are scaled by 1/s. Default is True
    
    Returns
    -------
    np.ndarray
        Input point cloud scaled by s
    """

    if (isinstance(s, float)) | (isinstance(s, int)):
        s = np.array([s] * coords.shape[-1])

    assert len(s) == coords.shape[-1], "s must either be a single value, or equal length to last dimension of given coordinates"

    # if we want to invert
    if invert:
        s = 1 / s
    # get scaling matrix
    scaling_matrix = np.diag(s)
    # matmul
    scaled_cloud = coords @ scaling_matrix
    return scaled_cloud

def align_neuron(N:Tree_graph, synapses:bool = True, scale:bool = True, s: int|float|np.ndarray|NoneType = None, invert:bool = True, bind:bool = True, unpack:bool = True, **kwargs) -> list | np.ndarray | Tree_graph:
    """Uses 'snap_to_axis' to align Neuron coordinates to the eigenvectors of the vertex point cloud. Can additionally scale using the 'coordinate_scale' function.

    Will center coordinates on the mean!

    See 'snap_to_axis' function documentation for addition arguments that can be passed.

    If synapses is True, will also convert synapse coordinates.

    If scale == True, the 'coordinate_scale' function is used, see 'coordinate_scale' function documentation.

    Parameters
    ----------
    N : nr.Tree_graph
        neurosetta.Tree_graph representation of a neuron
    synapses : bool, optional
        if possible will also convert coordinates of synapses, by default True. if False, synapses are ignored
    scale : bool, optional
        If True, coordinates are scaled by s, by default True
    s : int | float | np.ndarray | NoneType, optional
        scaling factor of neuron, by default None
    invert : bool, optional
        _description_, by default True
    bind : bool, optional
        _description_, by default True
    unpack : bool, optional
        _description_, by default True
    **kwargs
        Additional arguments passed to snap_to_axis
    Returns
    -------
    list | np.ndarray | nr.Tree_graph
        _description_
    """
    # get neuron vertex coordinates
    v_coords = g_vert_coords(N)
    # check if we have input and output properties as we will need to handle them as well
    if synapses:
        assert g_has_property(N,'inputs'), 'Input neuron must have input synapses to transform synapses'
        assert g_has_property(N,'outputs'), 'Input neuron must have output synapses to transform synapses'
        # post synaptic coordinates
        post_inputs = N.graph.gp['inputs'][['post_x','post_y','post_z']].values
        # graph coordinates
        graph_inputs = N.graph.gp['inputs'][['graph_x','graph_y','graph_z']].values
        # post synaptic coordinates
        pre_outputs = N.graph.gp['outputs'][['pre_x','pre_y','pre_z']].values
        # graph coordinates
        graph_outputs = N.graph.gp['outputs'][['graph_x','graph_y','graph_z']].values

        # build an indexing thing to lazily handle unpacking. the way this is done is a bit dumb
        ind = [0] * v_coords.shape[0]
        ind.extend([1] * post_inputs.shape[0])
        ind.extend([2] * graph_inputs.shape[0])
        ind.extend([3] * pre_outputs.shape[0])
        ind.extend([4] * graph_outputs.shape[0])
        ind = np.array(ind)

        # stack all coordinates
        all_coords = np.vstack([v_coords, post_inputs,graph_inputs,pre_outputs,graph_outputs])
    else:
        all_coords = v_coords

    # center all coordinates on mean
    all_coords = all_coords - all_coords.mean(axis = 0)

    # align to eigenaxis (remember we have to transpose)
    all_coords = snap_to_axis(all_coords).T

    # if we want to scale
    if scale:
        assert s != None, 'If you wish to scale coordinates, s must be provided. See scale function documentation for details'
        all_coords = coordinate_scale(all_coords,s,invert)


    if bind:
        if synapses:
            # bind! 
            N.graph.vp['coordinates'].set_2d_array(all_coords[np.where(ind == 0)].T)
            N.graph.gp['inputs'][['post_x','post_y','post_z']] = all_coords[np.where(ind == 1)] 
            N.graph.gp['inputs'][['graph_x','graph_y','graph_z']] = all_coords[np.where(ind == 2)]
            N.graph.gp['outputs'][['pre_x','pre_y','pre_z']] = all_coords[np.where(ind == 3)]
            N.graph.gp['outputs'][['graph_x','graph_y','graph_z']] = all_coords[np.where(ind == 4)] 
        else:
            N.graph.vp['coordinates'].set_2d_array(all_coords.T)

    else:
        if unpack:
            return [all_coords[np.where(ind == i)] for i in range(5)]
        else:
            if synapses:
                return all_coords, ind
            else:
                return all_coords



