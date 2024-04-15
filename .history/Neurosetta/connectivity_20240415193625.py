from .core import *
from .graphs import *
import pandas as pd

def snap_to_edge(start_points, end_points, point):

    # Ensure all arrays are of shape (N, 3)
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    point = np.atleast_2d(point)

    # Calculate line directions
    line_dirs = end_points - start_points
    
    # Vector from start points to the third point
    start_to_point = point - start_points
    
    # Dot product to project the vector onto the line directions
    t = np.sum(start_to_point * line_dirs, axis=1) / np.sum(line_dirs ** 2, axis=1)
    
    # Clip t values to be within [0, 1] to get the closest points on the lines
    t = np.clip(t, 0, 1)
    
    # Coordinates of closest points on the lines
    closest_points = start_points + t[:, np.newaxis] * line_dirs
    
    # Calculate distances between closest points and the third point
    distances = np.linalg.norm(closest_points - point, axis=1)
    
    # Find the index of the line with the minimum distance
    closest_index = np.argmin(distances)
    
    return closest_index, distances[closest_index], closest_points[closest_index]

def get_edge_coords(N:Tree_graph):
    edges = N.graph.get_edges()
    coords = g_vert_coords(N)
    p1 = coords[edges[:,0]]
    p2 = coords[edges[:,1]]
    return p1,p2  


def map_synapses(N:Tree_graph,df:pd.DataFrame,inputs:bool = True, outputs:bool = True,threshold: float = 4000) -> Tree_graph:
    """
    
    """

    # split given data frame into inputs and outputs
    in_df = df.loc[df.post == int(N.name),['pre','post_x','post_y','post_z','cleft_score','id']]
    out_df = df.loc[df.pre == int(N.name),['post','pre_x','pre_y','pre_z','cleft_score','id']]

    # get edge coordinates
    p1,p2 = get_edge_coords(N)
    # we need the edges as well
    edges = N.graph.get_edges()

    ### sort inputs first
    if inputs:
        syn_coords = in_df[['post_x','post_y','post_z']].values

        sources = np.zeros(len(syn_coords))
        targets = np.zeros_like(sources)
        distances = np.zeros_like(sources)
        xs = np.zeros_like(sources)
        ys = np.zeros_like(sources)
        zs = np.zeros_like(sources)

        # iterate through syn_coords:
        for i in range(len(syn_coords)):
            p3 = syn_coords[i]

            ind,dist,coordinate = snap_to_edge(p1,p2,p3)
            closest_edge = edges[ind]
            sources[i] = closest_edge[0]
            targets[i] = closest_edge[1]
            distances[i] = dist
            xs[i] = coordinate[0]
            ys[i] = coordinate[1]
            zs[i] = coordinate[2] 

        # update the in_df dataframe and bind it to N
        in_df['source'] = sources.astype(int)
        in_df['target'] = targets.astype(int)
        in_df['distance'] = distances   
        in_df['graph_x'] = xs
        in_df['graph_y'] = ys
        in_df['graph_z'] = zs

        in_df = in_df.loc[in_df.distance <= threshold]
        N.graph.gp['inputs'] = N.graph.new_gp("object", in_df)


    ## Repeat for output synapses
    if outputs:
        syn_coords = out_df[['pre_x','pre_y','pre_z']].values

        sources = np.zeros(len(syn_coords))
        targets = np.zeros_like(sources)
        distances = np.zeros_like(sources)
        xs = np.zeros_like(sources)
        ys = np.zeros_like(sources)
        zs = np.zeros_like(sources)

        # iterate through syn_coords:
        for i in range(len(syn_coords)):
            p3 = syn_coords[i]

            ind,dist,coordinate = snap_to_edge(p1,p2,p3)
            closest_edge = edges[ind]
            sources[i] = closest_edge[0]
            targets[i] = closest_edge[1]
            distances[i] = dist
            xs[i] = coordinate[0]
            ys[i] = coordinate[1]
            zs[i] = coordinate[2] 

        # update the out_df dataframe and bind it to N
        out_df['source'] = sources.astype(int)
        out_df['target'] = targets.astype(int)
        out_df['distance'] = distances   
        out_df['graph_x'] = xs
        out_df['graph_y'] = ys
        out_df['graph_z'] = zs

        out_df = out_df.loc[out_df.distance <= threshold]
        N.graph.gp['outputs'] = N.graph.new_gp("object", out_df)

