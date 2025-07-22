import pandas as pd
from typing import List
from numpy import 

from .core import Forest_graph, g_has_property


def _property_error(n: int, prop: str, ids: List):
    # Format only the first n ids in a readable way:
    formatted_ids = ", ".join(map(str, ids[:n]))

    error_message = (
        f"{n} neurons do not have {prop}, Use N_missing_property for a full list of ids. \n" 
        f"printing first {n} ids: [{formatted_ids}]"
    )

    raise ValueError(error_message)


def _all_property_error(prop):
    raise ValueError(f"No neurons in given Forest have {prop} property")

def N_missing_properties(N_all: Forest_graph, prop: str) -> List:
    bool_props = np.array(
        [
            nr.g_has_property(N_all.graph.vp["Neurons"][v], prop)
            for v in N_all.graph.iter_vertices()
        ],
        dtype=bool,
        )
    ids = [int(N_all.graph.vp['Neurons'][v].name) for v in np.where(bool_props == False)[0]]
    return ids

def check_neurons_have_property(
    N_all: nr.Forest_graph, prop: str, tol: int = 0, verbose: bool = True
) -> bool:
    bool_props = np.array(
        [
            nr.g_has_property(N_all.graph.vp["Neurons"][v], prop)
            for v in N_all.graph.iter_vertices()
        ],
        dtype=bool,
    )
    # count fails
    fails = abs( sum(bool_props) - len(bool_props))
    
    if fails <= tol:
        return True
    else:
        if verbose:
            if fails == len(bool_props):
                # _all_property_error(prop)
                ids = [int(N_all.graph.vp['Neurons'][v].name) for v in np.where(bool_props == False)[0]]
                _property_error(tol, prop, ids)
            else:
                # get ids which fail
                ids = [int(N_all.graph.vp['Neurons'][v].name) for v in np.where(bool_props == False)[0]]
                _property_error(tol, prop, ids)
        else:
            return False
        


