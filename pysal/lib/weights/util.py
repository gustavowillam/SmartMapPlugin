#file pysal/lib/weights/util.py

from scipy.spatial import KDTree
import scipy.spatial

__all__ = ['isKDTree']



KDTREE_TYPES = [scipy.spatial.KDTree, scipy.spatial.cKDTree]

def isKDTree(obj):
    """
    This is a utility function to determine whether or not an object is a
    KDTree, since KDTree and cKDTree have no common parent type
    """
    return any([issubclass(type(obj), KDTYPE) for KDTYPE in KDTREE_TYPES])

