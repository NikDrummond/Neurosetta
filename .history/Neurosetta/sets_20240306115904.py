# simple functions for set operations on families of sets

import numpy as np
from typing import List


def Sfamily_intersect(fam: List) -> np.ndarray:
    """
    returns the intersection of a family of sets
    """
    intersect = fam[0]
    for i in range(1, len(fam)):
        intersect = np.intersect1d(intersect, fam[i])

    return intersect


def Sfamily_union(fam: List) -> np.ndarray:
    """
    returns the union of a family of sets
    """
    union = fam[0]
    for i in range(1, len(fam)):
        union = np.union1d(union, fam[i])

    return union


def Sfamily_XOR(fam: List) -> np.ndarray:
    """
    returns the symetrical difference of a family of sets - union with the intersection removed
    """
    intersection = fam[0]
    union = fam[0]
    for i in range(1, len(fam)):
        intersection = np.intersect1d(intersection, fam[i])
        union = np.union1d(union, fam[i])

    x_or = union[~np.isin(union, intersection)]

    return x_or

def jacard_similarity(A:np.ndarray,B:np.ndarray):
    """

    Parameters
    ----------
    A : _type_
        _description_
    B : _type_
        _description_
    """

