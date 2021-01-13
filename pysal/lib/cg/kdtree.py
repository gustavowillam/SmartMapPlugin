#file pysal/lib/cg/kdtree.py 

"""
KDTree for PySAL: Python Spatial Analysis Library.

Adds support for Arc Distance to scipy.spatial.KDTree.
"""

import scipy.spatial
import numpy

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

__all__ = ["DISTANCE_METRICS", "FLOAT_EPS", "KDTree"]

DISTANCE_METRICS = ['Euclidean', 'Arc']
FLOAT_EPS = numpy.finfo(float).eps
RADIUS_EARTH_KM = 6371.0

def KDTree(data, leafsize=10, distance_metric='Euclidean',
           radius=RADIUS_EARTH_KM):
    """
    kd-tree built on top of kd-tree functionality in scipy. If using scipy 0.12
    or greater uses the scipy.spatial.cKDTree, otherwise uses
    scipy.spatial.KDTree. Offers both Arc distance and Euclidean distance.
    Note that Arc distance is only appropriate when points in latitude and
    longitude, and the radius set to meaningful value (see docs below).

    Parameters
    ----------
    data            : array
                      The data points to be indexed. This array is not copied,
                      and so modifying this data will result in bogus results.
                      Typically nx2.
    leafsize        : int
                      The number of points at which the algorithm switches over
                      to brute-force. Has to be positive. Optional, default is 10.
    distance_metric : string
                      Options: "Euclidean" (default) and "Arc".
    radius          : float
                      Radius of the sphere on which to compute distances.
                      Assumes data in latitude and longitude. Ignored if
                      distance_metric="Euclidean". Typical values:
                      pysal.cg.RADIUS_EARTH_KM  (default)
                      pysal.cg.RADIUS_EARTH_MILES
    """

    if distance_metric.lower() == 'euclidean':
        if int(scipy.version.version.split(".")[1]) < 12:
            return scipy.spatial.KDTree(data, leafsize)
        else:
            return scipy.spatial.cKDTree(data, leafsize)


# internal hack for the Arc_KDTree class inheritance
if int(scipy.version.version.split(".")[1]) < 12:
    temp_KDTree = scipy.spatial.KDTree
else:
    temp_KDTree = scipy.spatial.cKDTree

