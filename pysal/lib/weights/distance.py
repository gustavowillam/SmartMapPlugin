#file pysal/lib/weights/distance.py

__all__ = ["Kernel"]
__author__ = "Sergio J. Rey <srey@asu.edu>, Levi John Wolf <levi.john.wolf@gmail.com>"


from .util import isKDTree       #file pysal/lib/weights/util.py
from ..cg.kdtree import KDTree   #file pysal/lib/cg/kdtree.py 
from .weights import W           #file pysal/lib/weights/weights.py 


import numpy as np



class Kernel(W):
    """
    Spatial weights based on kernel functions.

    Parameters
    ----------

    data        : array
                  (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidth   : float
                  or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel.
    fixed       : binary
                  If true then :math:`h_i=h \\forall i`. If false then
                  bandwidth is adaptive across observations.
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    diagonal    : boolean
                  If true, set diagonal weights = 1.0, if false (default),
                  diagonals weights are set to value according to kernel
                  function.
    function    : {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = 1/2 \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    eps         : float
                  adjustment to ensure knn distance range is closed on the
                  knnth observations

    Attributes
    ----------
    weights : dict
              Dictionary keyed by id with a list of weights for each neighbor

    neighbors : dict
                of lists of neighbors keyed by observation id

    bandwidth : array
                array of bandwidths

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=Kernel(points)
    >>> kw.weights[0]
    [1.0, 0.500000049999995, 0.4409830615267465]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> kw.bandwidth
    array([[20.000002],
           [20.000002],
           [20.000002],
           [20.000002],
           [20.000002],
           [20.000002]])
    >>> kw15=Kernel(points,bandwidth=15.0)
    >>> kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> kw15.neighbors[0]
    [0, 1, 3]
    >>> kw15.bandwidth
    array([[15.],
           [15.],
           [15.],
           [15.],
           [15.],
           [15.]])

    Adaptive bandwidths user specified

    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=Kernel(points,bandwidth=bw)
    >>> kwa.weights[0]
    [1.0, 0.6, 0.552786404500042, 0.10557280900008403]
    >>> kwa.neighbors[0]
    [0, 1, 3, 4]
    >>> kwa.bandwidth
    array([[25. ],
           [15. ],
           [25. ],
           [16. ],
           [14.5],
           [25. ]])

    Endogenous adaptive bandwidths

    >>> kwea=Kernel(points,fixed=False)
    >>> kwea.weights[0]
    [1.0, 0.10557289844279438, 9.99999900663795e-08]
    >>> kwea.neighbors[0]
    [0, 1, 3]
    >>> kwea.bandwidth
    array([[11.18034101],
           [11.18034101],
           [20.000002  ],
           [11.18034101],
           [14.14213704],
           [18.02775818]])

    Endogenous adaptive bandwidths with Gaussian kernel

    >>> kweag=Kernel(points,fixed=False,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[11.18034101],
           [11.18034101],
           [20.000002  ],
           [11.18034101],
           [14.14213704],
           [18.02775818]])

    Diagonals to 1.0

    >>> kq = Kernel(points,function='gaussian')
    >>> kq.weights
    {0: [0.3989422804014327, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 0.3989422804014327, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 0.3989422804014327, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 0.3989422804014327, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 0.3989422804014327, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 0.3989422804014327]}
    >>> kqd = Kernel(points, function='gaussian', diagonal=True)
    >>> kqd.weights
    {0: [1.0, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 1.0, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 1.0, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 1.0, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 1.0, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 1.0]}

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=2,
                 function='triangular', eps=1.0000001, ids=None,
                 diagonal=False, 
                 distance_metric='euclidean', radius=None, 
                 **kwargs):
        if radius is not None:
            distance_metric='arc'
        if isKDTree(data):
            self.kdtree = data
            self.data = self.kdtree.data
            data = self.data
        else:
            self.kdtree = KDTree(data, distance_metric=distance_metric,
                                 radius=radius)
            self.data = self.kdtree.data
        self.k = k + 1
        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), 'float') * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()

        self._eval_kernel()
        neighbors, weights = self._k_to_W(ids)
        if diagonal:
            for i in neighbors:
                weights[i][neighbors[i].index(i)] = 1.0
        W.__init__(self, neighbors, weights, ids, **kwargs)

  

    def _k_to_W(self, ids=None):
        allneighbors = {}
        weights = {}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self.data))
        for i, neighbors in enumerate(self.kernel):
            if len(self.neigh[i]) == 0:
                allneighbors[ids[i]] = []
                weights[ids[i]] = []
            else:
                allneighbors[ids[i]] = list(ids[self.neigh[i]])
                weights[ids[i]] = self.kernel[i].tolist()
        return allneighbors, weights

    def _set_bw(self):
        dmat, neigh = self.kdtree.query(self.data, k=self.k)
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(dmat)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)
            # identify knn neighbors for each point
            nnq = self.kdtree.query(self.data, k=self.k)
            self.neigh = nnq[1]

    def _eval_kernel(self):
        # get points within bandwidth distance of each point
        if not hasattr(self, 'neigh'):
            kdtq = self.kdtree.query_ball_point
            neighbors = [kdtq(self.data[i], r=bwi[0]) for i,
                         bwi in enumerate(self.bandwidth)]
            self.neigh = neighbors
        # get distances for neighbors
        bw = self.bandwidth

        kdtq = self.kdtree.query
        z = []
        for i, nids in enumerate(self.neigh):
            di, ni = kdtq(self.data[i], k=len(nids))
            if not isinstance(di, np.ndarray):
                di = np.asarray([di] * len(nids))
                ni = np.asarray([ni] * len(nids))
            zi = np.array([dict(list(zip(ni, di)))[nid] for nid in nids]) / bw[i]
            z.append(zi)
        zs = z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            self.kernel = [1 - zi for zi in zs]
        elif self.function == 'uniform':
            self.kernel = [np.ones(zi.shape) * 0.5 for zi in zs]
        elif self.function == 'quadratic':
            self.kernel = [(3. / 4) * (1 - zi ** 2) for zi in zs]
        elif self.function == 'quartic':
            self.kernel = [(15. / 16) * (1 - zi ** 2) ** 2 for zi in zs]
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            self.kernel = [c * np.exp(-(zi ** 2) / 2.) for zi in zs]
        else:
            print(('Unsupported kernel function', self.function))

