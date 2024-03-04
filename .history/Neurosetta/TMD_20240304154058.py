from .core import Tree_graph
from .graphs import g_leaf_inds, g_root_ind, g_has_property

import graph_tool.all as gt
import numpy as np
import multiprocessing as mp
from persim import bottleneck
from tqdm import tqdm


def TMD(N: Tree_graph | gt.Graph, func: str | gt.VertexPropertyMap, bind=True):
    """_summary_

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        _description_
    func : str | gt.VertexPropertyMap
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # indicies of leaves and root
    l_inds = g_leaf_inds(g)
    root = g_root_ind(g)

    # initialise segments and visited nodes
    segments = np.zeros((len(l_inds), 2)).astype(int)
    visited = np.zeros_like(g.get_vertices()).astype(bool)

    # indicies of leaves sorted by descending dist func
    if isinstance(func, str):
        sorted_l = l_inds[np.argsort(g.vp[func].a[l_inds])[::-1]]
    elif isinstance(func, gt.VertexPropertyMap):
        sorted_l = l_inds[np.argsort(func.a[l_inds])[::-1]]

    # iterate
    for i in range(len(sorted_l)):
        l = sorted_l[i]
        # get parent
        current_p = g.get_in_neighbours(l)[0]
        # while we are collecting nodes:
        while True:
            # if parent not in viewed:
            if visited[current_p] == False:
                # add to viewed
                visited[current_p] = True

                # if we are at the root:
                if current_p == root:
                    # note this segment
                    segments[i] = [l, current_p]
                    # move to next leaf
                    break
                else:
                    # move to next node
                    current_p = g.get_in_neighbours(current_p)[0]
            # else:
            else:
                # note node and l - node to segments
                segments[i] = [l, current_p]
                # move to next leaf
                break
    # get length of segments according to function used
    if isinstance(func, str):
        lens = np.array([abs(g.vp[func][i[0]] - g.vp[func][i[1]]) for i in segments])
    elif isinstance(func, gt.VertexPropertyMap):
        lens = np.array([abs(func[i[0]] - func[i[1]]) for i in segments])

    TMD = dict(
        {
            "function": func,
            "birth_ind": segments[:, 1],
            "death_ind": segments[:, 0],
            "survival_len": lens,
        }
    )

    if bind:
        g.gp["TMD"] = g.new_gp("object", TMD)
    else:
        return TMD
def TMD_seg_edge(g):

    if g_has_property(g,'TMD','g'):
        return np.vstack((g.gp['TMD']['birth_ind'], g.gp['TMD']['death_ind'])).T
    else:
        raise AttributeError('Graph has no TMD dictionary - generate this using the TMD function')   

def TMD_barcode(g):

    # add check to make sure TMD graph property exists
    if g_has_property(g,'TMD','g'):
        segments = TMD_seg_edge(g)
        # get leaf inds
        l_inds = g_leaf_inds(g)
        x = np.zeros_like(segments)
        y = np.vstack((np.arange(len(l_inds)) + 1,np.arange(len(l_inds)) + 1))


        # get length of each segments
        seg_len = g.gp['TMD']['survival_len']
        sort_inds = np.argsort(seg_len)

        for i in range(len(sort_inds)):
            curr_seg = segments[sort_inds[i]]
            x[i,0] = g.vp[g.gp['TMD']['function']][curr_seg[0]]
            x[i,1] = g.vp[g.gp['TMD']['function']][curr_seg[1]]      
        x = x.T     

        return x,y
    
    else:
        raise AttributeError('Graph has no TMD dictionary - generate this using the TMD function')
    
def TMD_persistance_diagram(g, split = False):
    
    segments = TMD_seg_edge(g)
    func = g.gp['TMD']['function']
    points = np.asarray([[g.vp[func][i[0]],g.vp[func][i[1]]] for i in segments])

    if split:
        return points[:,0], points[:,1]   
    else:
        return points 

def TMD_persistance_im(g,xlim = None, ylim = None, norm_factor = None):

    pd = nr.TMD_persistance_diagram(g)

    if xlim is None:
        xlim = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    if ylim is None:
        ylim = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    X, Y = np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = np.max(Z)

    Zn = Z / norm_factor

    return Zn    

def bottleneck_dist(N1,N2):
    """Compute bottleneck distance between two persistance diagrams

    Parameters
    ----------
    N1 : _type_
        _description_
    N2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    N1_pd = TMD_persistance_diagram(N1.graph)
    N2_pd = TMD_persistance_diagram(N2.graph)

    return bottleneck(N1_pd,N2_pd)    



def _calculate_distances(i, N_all, result_queue):
    num_points = len(N_all)
    distances = np.zeros(num_points)
    for j in range(num_points):
        if j != i:
            distances[j] = bottleneck_dist(N_all[i], N_all[j])
    result_queue.put((i, distances))

def bottleneck_matrix(N_all, parallel = True, max_processes = None):

    if parallel:
        num_points = len(N_all)
        dist_mat = np.zeros((num_points, num_points))

        # Determine the number of processes to use
        if max_processes is None:
            max_processes = mp.cpu_count()

        # Create a multiprocessing Queue to store results
        result_queue = mp.Queue()

        # Create processes to calculate distances
        processes = []
        for i in range(num_points):
            process = mp.Process(target=_calculate_distances, args=(i, N_all, result_queue))
            processes.append(process)
            process.start()

            # Limit the number of processes
            if max_processes is not None and len(processes) >= max_processes:
                for p in processes:
                    p.join()
                processes = []

        # Retrieve remaining results from the Queue and fill the distance matrix
        while not result_queue.empty():
            i, distances = result_queue.get()
            dist_mat[i, :] = distances

        # Join remaining processes
        for process in processes:
            process.join()

    else:    
        num_points = len(N_all)
        dist_mat = np.zeros((num_points, num_points))

        # Calculate bottleneck distances
        for i in tqdm(range(num_points)):
            for j in range(i + 1, num_points):
                t = bottleneck_dist(N_all[i], N_all[j])
                dist_mat[i, j] = dist_mat[j, i] = t    

    return dist_mat

