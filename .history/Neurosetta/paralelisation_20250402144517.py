import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .core import Tree_graph, Forest_graph
from typing import List
from .graphs import g_cable_length, g_vert_coords, Random_ST

def Random_ST_cable_distribution(N: Tree_graph, perms: int = 100) -> np.ndarray:
    coords = g_vert_coords(N)



    # joblib uses loky backend by default, which avoids fork issues
    results = Parallel(n_jobs=-1)(
        delayed(compute_random_length)(i) for i in tqdm(range(perms))
    )

    rand_lens = np.array(results)