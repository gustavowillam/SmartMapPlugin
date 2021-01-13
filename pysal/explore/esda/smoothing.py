from __future__ import division
"""
Apply smoothing to rate computation

[Longer Description]

Author(s):
    Myunghwa Hwang mhwang4@gmail.com
    David Folch dfolch@asu.edu
    Luc Anselin luc.anselin@asu.edu
    Serge Rey srey@asu.edu

"""

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>, David Folch <dfolch@asu.edu>, Luc Anselin <luc.anselin@asu.edu>, Serge Rey <srey@asu.edu"

import numpy as np 


__all__ = ['assuncao_rate']


def assuncao_rate(e, b):
    """The standardized rates where the mean and stadard deviation used for
    the standardization are those of Empirical Bayes rate estimates
    The standardized rates resulting from this function are used to compute
    Moran's I corrected for rate variables [Choynowski1959]_ .

    Parameters
    ----------
    e          : array(n, 1)
                 event variable measured at n spatial units
    b          : array(n, 1)
                 population at risk variable measured at n spatial units

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array (nx1)

    Examples
    --------

    Creating an array of an event variable (e.g., the number of cancer patients)
    for 8 regions.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same 8 regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Computing the rates

    >>> assuncao_rate(e, b)[:4]
    array([ 1.03843594, -0.04099089, -0.56250375, -1.73061861])

    """

    y = e * 1.0 / b
    e_sum, b_sum = sum(e), sum(b)
    ebi_b = e_sum * 1.0 / b_sum
    s2 = sum(b * ((y - ebi_b) ** 2)) / b_sum
    ebi_a = s2 - ebi_b / (float(b_sum) / len(e))
    ebi_v = ebi_a + ebi_b / b
    return (y - ebi_b) / np.sqrt(ebi_v)
