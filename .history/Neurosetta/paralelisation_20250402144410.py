import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .core import Tree_graph, Forest_graph
from typing import List
from .graphs import g_cable_length, g_vert_coords, Random_ST

def Random_ST_cable_distribution(N_all: List | Forest_graph | Tree_graph, perms: int = 100) -