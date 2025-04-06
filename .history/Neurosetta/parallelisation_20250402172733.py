import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .core import Tree_graph, Forest_graph
from typing import List
from .graphs import g_cable_length, g_vert_coords, Random_ST

def Random_ST_cable_distribution(N: Tree_graph, perms: int = 100) -> np.ndarray:
    coords = g_vert_coords(N)

    def compute_random_length(_):
        g2 = Random_ST(coords, root=True)
        return g_cable_length(g2)

    # joblib uses loky backend by default, which avoids fork issues
    results = Parallel(n_jobs=-1)(
        delayed(compute_random_length)(i) for i in tqdm(range(perms))
    )

    return np.array(results)


def Random_ST_cable_distribution(N_all: List | Forest_graph, perms: int = 100) -> np.ndarray:
    
    if isinstance(N_all, List):
        v = np.random.randint(0, len(N_all))
        coords = g_vert_coords(N_all[v])
    elif isinstance(N_all, Forest_graph):
        v = np.random.randint(0, N_all.graph.num_vertices())
        coords = nr.g_vert_coords(N_all.graph.vp['Neurons'][v])
    
    coords = g_vert_coords(N)

    def compute_random_length(_):
        g2 = Random_ST(coords, root=True)
        return g_cable_length(g2)

    # joblib uses loky backend by default, which avoids fork issues
    results = Parallel(n_jobs=-1)(
        delayed(compute_random_length)(i) for i in tqdm(range(perms))
    )

    return np.array(results)