#  py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
#  University of Birmingham.
#
#  Copyright (C) 2019-2024  James Ball unless otherwise stated
#
#  This file is part of py3DXRDProc.
#
#  py3DXRDProc is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  py3DXRDProc is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.
from itertools import product

#  This file incorporates work covered by the following copyright and
#  permission notice:

# Copyright(c) 2013 - 2019 Henry Proudhon.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import numpy.typing as npt
from scipy import spatial
from numba import prange, njit, jit
from typing import Tuple

import logging

from pymicro.crystal.lattice import Symmetry

log = logging.getLogger(__name__)

# SN = orientation.ShojiNishiyama(cs_cubic, cs_hex)
# variant_order = variants(SN)
# SN mtex order is:
# parent  || daughter  parent || daughter
# (111)   || (0001)   [10-1] || [2-1-10]
# (1-11)  || (0001)   [10-1] || [-2110]
# (1-1-1) || (000-1)  [01-1] || [2-1-10]
# (11-1)  || (0001)   [1-10] || [2-1-10]

# matrices are inverted:
S_N_variant_matrices = np.array(
    [[[0.4082, -0.7071, 0.5774],
      [0.4082, 0.7071, 0.5774],
      [-0.8165, 0.0000, 0.5774]]
        ,
     [[-0.8165, 0.0000, 0.5774],
      [-0.4082, -0.7071, -0.5774],
      [0.4082, -0.7071, 0.5774]]
        ,
     [[-0.4082, -0.7071, -0.5774],
      [0.4082, -0.7071, 0.5774],
      [-0.8165, 0.0000, 0.5774]]
        ,
     [[0.4082, -0.7071, 0.5774],
      [-0.8165, 0.0000, 0.5774],
      [-0.4082, -0.7071, -0.5774]]]
)

# KS = orientation.KurdjumovSachs(cs_gamma, cs_aprime)
# variant_order = variants(KS)
# KS mtex order is:
# parent  || daughter  parent || daughter
# (111) || (011)   [10-1] || [11-1]
# (111) || (0-1-1)   [10-1] || [11-1]
# (111) || (011)   [01-1] || [-1-11]
# (111) || (0-1-1)   [01-1] || [-1-11]
# (111) || (011)   [1-10] || [-1-11]
# (111) || (0-1-1)   [1-10] || [-1-11]
# (1-11) || (011)   [10-1] || [-1-11]
# (1-11) || (0-1-1)   [10-1] || [-1-11]
# (1-11) || (011)   [110] || [11-1]
# (1-11) || (0-1-1)   [110] || [11-1]
# (1-11) || (011)   [011] || [-1-11]
# (1-11) || (0-1-1)   [011] || [-1-11]
# (1-1-1) || (0-1-1)   [01-1] || [11-1]
# (1-1-1) || (011)   [01-1] || [11-1]
# (1-1-1) || (0-1-1)   [101] || [11-1]
# (1-1-1) || (011)   [101] || [11-1]
# (1-1-1) || (0-1-1)   [110] || [-1-11]
# (1-1-1) || (011)   [110] || [-1-11]
# (11-1) || (011)   [1-10] || [11-1]
# (11-1) || (0-1-1)   [1-10] || [11-1]
# (11-1) || (011)   [011] || [11-1]
# (11-1) || (0-1-1)   [011] || [11-1]
# (11-1) || (011)   [101] || [-1-11]
# (11-1) || (0-1-1)   [101] || [-1-11]

# matrices are inverted:

K_S_variant_matrices = np.array([
    [[0.7416, 0.6498, 0.1667],
     [-0.6667, 0.7416, 0.0749],
     [-0.0749, -0.1667, 0.9832]]
    ,
    [[0.0749, 0.1667, -0.9832],
     [0.6667, -0.7416, -0.0749],
     [-0.7416, -0.6498, -0.1667]]
    ,
    [[-0.6667, 0.7416, 0.0749],
     [-0.0749, -0.1667, 0.9832],
     [0.7416, 0.6498, 0.1667]]
    ,
    [[0.6667, -0.7416, -0.0749],
     [-0.7416, -0.6498, -0.1667],
     [0.0749, 0.1667, -0.9832]]
    ,
    [[-0.0749, -0.1667, 0.9832],
     [0.7416, 0.6498, 0.1667],
     [-0.6667, 0.7416, 0.0749]]
    ,
    [[-0.7416, -0.6498, -0.1667],
     [0.0749, 0.1667, -0.9832],
     [0.6667, -0.7416, -0.0749]]
    ,
    [[-0.0749, -0.1667, 0.9832],
     [0.6667, -0.7416, -0.0749],
     [0.7416, 0.6498, 0.1667]]
    ,
    [[-0.7416, -0.6498, -0.1667],
     [-0.6667, 0.7416, 0.0749],
     [0.0749, 0.1667, -0.9832]]
    ,
    [[0.7416, 0.6498, 0.1667],
     [0.0749, 0.1667, -0.9832],
     [-0.6667, 0.7416, 0.0749]]
    ,
    [[0.0749, 0.1667, -0.9832],
     [0.7416, 0.6498, 0.1667],
     [0.6667, -0.7416, -0.0749]]
    ,
    [[-0.6667, 0.7416, 0.0749],
     [-0.7416, -0.6498, -0.1667],
     [-0.0749, -0.1667, 0.9832]]
    ,
    [[0.6667, -0.7416, -0.0749],
     [-0.0749, -0.1667, 0.9832],
     [-0.7416, -0.6498, -0.1667]]
    ,
    [[0.6667, -0.7416, -0.0749],
     [0.7416, 0.6498, 0.1667],
     [-0.0749, -0.1667, 0.9832]]
    ,
    [[-0.6667, 0.7416, 0.0749],
     [0.0749, 0.1667, -0.9832],
     [-0.7416, -0.6498, -0.1667]]
    ,
    [[0.0749, 0.1667, -0.9832],
     [-0.6667, 0.7416, 0.0749],
     [0.7416, 0.6498, 0.1667]]
    ,
    [[0.7416, 0.6498, 0.1667],
     [0.6667, -0.7416, -0.0749],
     [0.0749, 0.1667, -0.9832]]
    ,
    [[-0.7416, -0.6498, -0.1667],
     [-0.0749, -0.1667, 0.9832],
     [-0.6667, 0.7416, 0.0749]]
    ,
    [[-0.0749, -0.1667, 0.9832],
     [-0.7416, -0.6498, -0.1667],
     [0.6667, -0.7416, -0.0749]]
    ,
    [[0.7416, 0.6498, 0.1667],
     [-0.0749, -0.1667, 0.9832],
     [0.6667, -0.7416, -0.0749]]
    ,
    [[0.0749, 0.1667, -0.9832],
     [-0.7416, -0.6498, -0.1667],
     [-0.6667, 0.7416, 0.0749]]
    ,
    [[-0.6667, 0.7416, 0.0749],
     [0.7416, 0.6498, 0.1667],
     [0.0749, 0.1667, -0.9832]]
    ,
    [[0.6667, -0.7416, -0.0749],
     [0.0749, 0.1667, -0.9832],
     [0.7416, 0.6498, 0.1667]]
    ,
    [[-0.0749, -0.1667, 0.9832],
     [-0.6667, 0.7416, 0.0749],
     [-0.7416, -0.6498, -0.1667]]
    ,
    [[-0.7416, -0.6498, -0.1667],
     [0.6667, -0.7416, -0.0749],
     [-0.0749, -0.1667, 0.9832]]
])

# NW = orientation.NishiyamaWassermann(cs_gamma, cs_aprime)
# variant_order = variants(NW)
# NW mtex order is:
# parent  || daughter  parent || daughter
# (111) || (011)   [1-10] || [-100]
# (111) || (0-1-1)   [01-1] || [-100]
# (111) || (011)   [10-1] || [100]
# (1-11) || (011)   [011] || [-100]
# (1-11) || (0-1-1)   [110] || [100]
# (1-11) || (011)   [10-1] || [-100]
# (1-1-1) || (0-1-1)   [110] || [-100]
# (1-1-1) || (011)   [101] || [100]
# (1-1-1) || (0-1-1)   [01-1] || [100]
# (11-1) || (011)   [101] || [-100]
# (11-1) || (0-1-1)   [011] || [100]
# (11-1) || (011)   [1-10] || [100]

# matrices are inverted:

N_W_variant_matrices = np.array([
    [[-0.7071, 0.1196, 0.6969],
     [0.7071, 0.1196, 0.6969],
     [0.0000, 0.9856, -0.1691]]
    ,
    [[-0.0000, -0.9856, 0.1691],
     [-0.7071, -0.1196, -0.6969],
     [0.7071, -0.1196, -0.6969]]
    ,
    [[0.7071, 0.1196, 0.6969],
     [0.0000, 0.9856, -0.1691],
     [-0.7071, 0.1196, 0.6969]]
    ,
    [[0.0000, 0.9856, -0.1691],
     [-0.7071, -0.1196, -0.6969],
     [-0.7071, 0.1196, 0.6969]]
    ,
    [[0.7071, -0.1196, -0.6969],
     [0.7071, 0.1196, 0.6969],
     [-0.0000, -0.9856, 0.1691]]
    ,
    [[-0.7071, 0.1196, 0.6969],
     [-0.0000, -0.9856, 0.1691],
     [0.7071, 0.1196, 0.6969]]
    ,
    [[-0.7071, -0.1196, -0.6969],
     [-0.7071, 0.1196, 0.6969],
     [0.0000, 0.9856, -0.1691]]
    ,
    [[0.7071, 0.1196, 0.6969],
     [-0.0000, -0.9856, 0.1691],
     [0.7071, -0.1196, -0.6969]]
    ,
    [[-0.0000, -0.9856, 0.1691],
     [0.7071, 0.1196, 0.6969],
     [-0.7071, 0.1196, 0.6969]]
    ,
    [[-0.7071, 0.1196, 0.6969],
     [0.0000, 0.9856, -0.1691],
     [-0.7071, -0.1196, -0.6969]]
    ,
    [[-0.0000, -0.9856, 0.1691],
     [0.7071, -0.1196, -0.6969],
     [0.7071, 0.1196, 0.6969]]
    ,
    [[0.7071, 0.1196, 0.6969],
     [-0.7071, 0.1196, 0.6969],
     [-0.0000, -0.9856, 0.1691]]
])

# GT = orientation.GreningerTrojano(cs_gamma, cs_aprime)
# GT is irrationally defined in MTEX (grrr)
# so I got the plane and direction relationships from here: 10.1016/j.matchar.2021.111501
# then I checked each plane and direction relationship against each variant
# that way I could associate the planes and directions to the variant ID in MTEX
# variant_order = variants(GT)
# GT mtex order is:
# parent  || daughter  parent || daughter
# (111)  || (110)    (17 -12 -5) || (-7 -17 17)
# (111)  || (-1-10)  (-5 -12 17) || (-7 17 -17)
# (111)  || (110)    (-12 -5 17) || (-7 -17 17)
# (111)  || (-1-10)  (-12 17 -5) || (-7 17 -17)
# (111)  || (110)    (-5 17 -12) || (-7 -17 17)
# (111)  || (-1-10)  (17 -5 -12) || (-7 17 -17)
# (1-11) || (110)    (5 -12 -17) || (-7 -17 17)
# (1-11) || (-1-10)  (-17 -12 5) || (-7 17 -17)
# (1-11) || (110)    (-17 -5 12) || (-7 -17 17)
# (1-11) || (-1-10)  (5 17 12)   || (-7 17 -17)
# (1-11) || (110)    (12 17 5)   || (-7 -17 17)
# (1-11) || (-1-10)  (12 -5 -17) || (-7 17 -17)
# (-111) || (110)    (-12 -17 5) || (-7 -17 17)
# (-111) || (-1-10)  (-12 5 -17) || (-7 17 -17)
# (-111) || (110)    (-5 12 -17) || (-7 -17 17)
# (-111) || (-1-10)  (17 12 5)   || (-7 17 -17)
# (-111) || (110)    (17 5 12)   || (-7 -17 17)
# (-111) || (-1-10)  (-5 -17 12) || (-7 17 -17)
# (11-1) || (110)    (-17 5 -12) || (-7 -17 17)
# (11-1) || (-1-10)  (5 -17 -12) || (-7 17 -17)
# (11-1) || (110)    (12 -17 -5) || (-7 -17 17)
# (11-1) || (-1-10)  (12 5 17)   || (-7 17 -17)
# (11-1) || (110)    (5 12 17)   || (-7 -17 17)
# (11-1) || (-1-10)  (-17 12 -5) || (-7 17 -17)
G_T_variant_matrices = np.array([
    [
        [0.9861, -0.1625, 0.0342],
        [0.1363, 0.6743, -0.7258],
        [0.0948, 0.7204, 0.6871]
    ],
    [
        [-0.0948, -0.7204, -0.6871],
        [-0.1363, -0.6743, 0.7258],
        [-0.9861, 0.1625, -0.0342]
    ],
    [
        [0.1363, 0.6743, -0.7258],
        [0.0948, 0.7204, 0.6871],
        [0.9861, -0.1625, 0.0342]
    ],
    [
        [-0.1363, -0.6743, 0.7258],
        [-0.9861, 0.1625, -0.0342],
        [-0.0948, -0.7204, -0.6871]
    ],
    [
        [0.0948, 0.7204, 0.6871],
        [0.9861, -0.1625, 0.0342],
        [0.1363, 0.6743, -0.7258]
    ],
    [
        [-0.9861, 0.1625, -0.0342],
        [-0.0948, -0.7204, -0.6871],
        [-0.1363, -0.6743, 0.7258]
    ],
    [
        [0.0948, 0.7204, 0.6871],
        [-0.1363, -0.6743, 0.7258],
        [0.9861, -0.1625, 0.0342]
    ],
    [
        [-0.9861, 0.1625, -0.0342],
        [0.1363, 0.6743, -0.7258],
        [-0.0948, -0.7204, -0.6871]
    ],
    [
        [0.9861, -0.1625, 0.0342],
        [-0.0948, -0.7204, -0.6871],
        [0.1363, 0.6743, -0.7258]
    ],
    [
        [-0.0948, -0.7204, -0.6871],
        [0.9861, -0.1625, 0.0342],
        [-0.1363, -0.6743, 0.7258]
    ],
    [
        [0.1363, 0.6743, -0.7258],
        [-0.9861, 0.1625, -0.0342],
        [0.0948, 0.7204, 0.6871]
    ],
    [
        [-0.1363, -0.6743, 0.7258],
        [0.0948, 0.7204, 0.6871],
        [-0.9861, 0.1625, -0.0342]
    ],
    [
        [-0.1363, -0.6743, 0.7258],
        [0.9861, -0.1625, 0.0342],
        [0.0948, 0.7204, 0.6871]
    ],
    [
        [0.1363, 0.6743, -0.7258],
        [-0.0948, -0.7204, -0.6871],
        [-0.9861, 0.1625, -0.0342]
    ],
    [
        [-0.0948, -0.7204, -0.6871],
        [0.1363, 0.6743, -0.7258],
        [0.9861, -0.1625, 0.0342]
    ],
    [
        [0.9861, -0.1625, 0.0342],
        [-0.1363, -0.6743, 0.7258],
        [-0.0948, -0.7204, -0.6871]
    ],
    [
        [-0.9861, 0.1625, -0.0342],
        [0.0948, 0.7204, 0.6871],
        [0.1363, 0.6743, -0.7258]
    ],
    [
        [0.0948, 0.7204, 0.6871],
        [-0.9861, 0.1625, -0.0342],
        [-0.1363, -0.6743, 0.7258]
    ],
    [
        [0.9861, -0.1625, 0.0342],
        [0.0948, 0.7204, 0.6871],
        [-0.1363, -0.6743, 0.7258]
    ],
    [
        [-0.0948, -0.7204, -0.6871],
        [-0.9861, 0.1625, -0.0342],
        [0.1363, 0.6743, -0.7258]
    ],
    [
        [0.1363, 0.6743, -0.7258],
        [0.9861, -0.1625, 0.0342],
        [-0.0948, -0.7204, -0.6871]
    ],
    [
        [-0.1363, -0.6743, 0.7258],
        [-0.0948, -0.7204, -0.6871],
        [0.9861, -0.1625, 0.0342]
    ],
    [
        [0.0948, 0.7204, 0.6871],
        [0.1363, 0.6743, -0.7258],
        [-0.9861, 0.1625, -0.0342]
    ],
    [
        [-0.9861, 0.1625, -0.0342],
        [-0.1363, -0.6743, 0.7258],
        [0.0948, 0.7204, 0.6871]
    ]
])

# Pitsch = orientation.Pitsch(cs_gamma, cs_aprime)
# variant_order = variants(Pitsch)
# Pitsch mtex order is:
# parent  || daughter  parent || daughter
# (101) || (-111)   [010] || [101]
# (101) || (1-1-1)   [010] || [-10-1]
# (011) || (-111)   [100] || [101]
# (011) || (1-1-1)   [100] || [-10-1]
# (110) || (-111)   [001] || [101]
# (110) || (1-1-1)   [001] || [-10-1]
# (1-10) || (-111)   [001] || [101]
# (1-10) || (1-1-1)   [001] || [-10-1]
# (01-1) || (1-1-1)   [100] || [101]
# (01-1) || (-111)   [100] || [-10-1]
# (10-1) || (1-1-1)   [010] || [101]
# (10-1) || (-111)   [010] || [-10-1]
Pitsch_variant_matrices = np.array([
    [[-0.6969, -0.1691, 0.6969],
     [0.7071, -0.0000, 0.7071],
     [-0.1196, 0.9856, 0.1196]]
    ,
    [[0.1196, -0.9856, -0.1196],
     [-0.7071, -0.0000, -0.7071],
     [0.6969, 0.1691, -0.6969]]
    ,
    [[0.7071, -0.0000, 0.7071],
     [-0.1196, 0.9856, 0.1196],
     [-0.6969, -0.1691, 0.6969]]
    ,
    [[-0.7071, -0.0000, -0.7071],
     [0.6969, 0.1691, -0.6969],
     [0.1196, -0.9856, -0.1196]]
    ,
    [[-0.1196, 0.9856, 0.1196],
     [-0.6969, -0.1691, 0.6969],
     [0.7071, -0.0000, 0.7071]]
    ,
    [[0.6969, 0.1691, -0.6969],
     [0.1196, -0.9856, -0.1196],
     [-0.7071, 0.0000, -0.7071]]
    ,
    [[-0.6969, -0.1691, 0.6969],
     [0.1196, -0.9856, -0.1196],
     [0.7071, -0.0000, 0.7071]]
    ,
    [[0.1196, -0.9856, -0.1196],
     [-0.6969, -0.1691, 0.6969],
     [-0.7071, 0.0000, -0.7071]]
    ,
    [[0.7071, -0.0000, 0.7071],
     [0.6969, 0.1691, -0.6969],
     [-0.1196, 0.9856, 0.1196]]
    ,
    [[-0.7071, -0.0000, -0.7071],
     [-0.1196, 0.9856, 0.1196],
     [0.6969, 0.1691, -0.6969]]
    ,
    [[0.1196, -0.9856, -0.1196],
     [0.7071, -0.0000, 0.7071],
     [-0.6969, -0.1691, 0.6969]]
    ,
    [[-0.6969, -0.1691, 0.6969],
     [-0.7071, -0.0000, -0.7071],
     [0.1196, -0.9856, -0.1196]]
])

cubic_symm_ops = Symmetry.cubic.symmetry_operators()
hex_symm_ops = Symmetry.hexagonal.symmetry_operators()


def upper_triangular_to_symmetric(upper_tri: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts a 3x3 upper triangular matrix to a symmetric one just with indexing:
    `["eps11", "eps22", "eps33", "eps23", "eps13", "eps12"]`
    to
    [[11 12 13],
    [21 22 23],
    [31 32 33]]

    :param upper_tri: The 1x6 array
    :raises TypeError: If `upper_tri` isn't a Numpy array
    :raises ValueError: If `upper_tri` should be an array of length 6
    :raises TypeError: If `upper_tri` isn't an array of float64 elements

    :return: The 3x3 symmetric matrix
    """
    if not isinstance(upper_tri, np.ndarray):
        raise TypeError("upper_tri should be a numpy array!")
    if not np.shape(upper_tri) == (6,):
        raise ValueError("upper_tri should be an array of length (6,)")
    if not upper_tri.dtype == np.dtype("float64"):
        raise TypeError("upper_tri should be an array of floats!")

    symmetric = np.zeros((3, 3))
    symmetric[0, 0] = upper_tri[0]
    symmetric[1, 1] = upper_tri[1]
    symmetric[2, 2] = upper_tri[2]
    symmetric[0, 1] = upper_tri[5]
    symmetric[1, 0] = upper_tri[5]
    symmetric[0, 2] = upper_tri[4]
    symmetric[2, 0] = upper_tri[4]
    symmetric[1, 2] = upper_tri[3]
    symmetric[2, 1] = upper_tri[3]

    return symmetric


def symmetric_to_upper_triangular(symmetric: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts a symmetric array to an upper triangular:
    `[[11 12 13], [21 22 23], [31 32 33]]` to `["eps11", "eps22", "eps33", "eps23", "eps13", "eps12"]`

    :param symmetric: The 3x3 symmetric matrix
    :raises TypeError: If `symmetric` isn't a Numpy array
    :raises ValueError: If `symmetric` isn't an array of shape (3,3)
    :raises TypeError: If `symmetric` isn't an array of float64 elements
    :raises ValueError: If `symmetric` isn't symmetric
    :return: The 3x3 array
    """

    if not isinstance(symmetric, np.ndarray):
        raise TypeError("symmetric should be a numpy array!")
    if not np.shape(symmetric) == (3, 3):
        raise ValueError("symmetric should be an array of length (3,3)")
    if not symmetric.dtype == np.dtype("float64"):
        raise TypeError("symmetric should be an array of floats!")
    if not np.allclose(symmetric, symmetric.T):
        raise ValueError("symmetric should be a symmetric matrix!")

    flat = np.zeros(6)
    flat[0] = symmetric[0, 0]
    flat[1] = symmetric[1, 1]
    flat[2] = symmetric[2, 2]
    flat[3] = symmetric[1, 2]
    flat[4] = symmetric[0, 2]
    flat[5] = symmetric[0, 1]

    return flat


def custom_array_to_string(input_array: npt.NDArray[np.float64]) -> str:
    """Convert a flat numpy array to a string. Used for fixed-precision GFF writing

    :param input_array: Input numpy array
    :raises TypeError: If `input_array` is not a numpy array
    :return: The array converted to a flattened string
    """

    if not isinstance(input_array, np.ndarray):
        raise TypeError("Must supply an array")

    return " ".join(
        np.array2string(input_array, precision=8, separator=" ", floatmode="fixed").replace("[", "").replace("]",
                                                                                                             "").replace(
            "\n", "").rstrip().lstrip().split())


def MVCOBMatrix(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede after Joel Bernier
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """GenerateS array of 6 x 6 basis transformation matrices for the
    Mandel-Voigt tensor representation in 3-D.
    Components of symmetric 4th-rank tensors transform in a
    manner analogous to symmetric 2nd-rank tensors in full
    matrix notation.

    Copyright (C) 2008 Jette Oddershede, September 16 2008 after Joel Bernier

    :param R: (3, 3) ndarray representing a change of basis matrix
    :raises TypeError: If `R` isn't a Numpy array
    :raises ValueError: If `R` isn't an array of shape (3,3)
    :raises TypeError: If `R` isn't an array of float64 elements
    :return: (6, 6) ndarray of transformation matrices
    """

    if not isinstance(R, np.ndarray):
        raise TypeError("R should be a numpy array!")
    if not np.shape(R) == (3, 3):
        raise ValueError("R should be an array of length (3,3)")
    if not R.dtype == np.dtype("float64"):
        raise TypeError("R should be an array of floats!")

    T = np.zeros((6, 6), dtype='float64')

    T[0, 0] = R[0, 0] ** 2
    T[0, 1] = R[0, 1] ** 2
    T[0, 2] = R[0, 2] ** 2
    T[0, 3] = np.sqrt(2.) * R[0, 1] * R[0, 2]
    T[0, 4] = np.sqrt(2.) * R[0, 0] * R[0, 2]
    T[0, 5] = np.sqrt(2.) * R[0, 0] * R[0, 1]
    T[1, 0] = R[1, 0] ** 2
    T[1, 1] = R[1, 1] ** 2
    T[1, 2] = R[1, 2] ** 2
    T[1, 3] = np.sqrt(2.) * R[1, 1] * R[1, 2]
    T[1, 4] = np.sqrt(2.) * R[1, 0] * R[1, 2]
    T[1, 5] = np.sqrt(2.) * R[1, 0] * R[1, 1]
    T[2, 0] = R[2, 0] ** 2
    T[2, 1] = R[2, 1] ** 2
    T[2, 2] = R[2, 2] ** 2
    T[2, 3] = np.sqrt(2.) * R[2, 1] * R[2, 2]
    T[2, 4] = np.sqrt(2.) * R[2, 0] * R[2, 2]
    T[2, 5] = np.sqrt(2.) * R[2, 0] * R[2, 1]
    T[3, 0] = np.sqrt(2.) * R[1, 0] * R[2, 0]
    T[3, 1] = np.sqrt(2.) * R[1, 1] * R[2, 1]
    T[3, 2] = np.sqrt(2.) * R[1, 2] * R[2, 2]
    T[3, 3] = R[1, 2] * R[2, 1] + R[1, 1] * R[2, 2]
    T[3, 4] = R[1, 2] * R[2, 0] + R[1, 0] * R[2, 2]
    T[3, 5] = R[1, 1] * R[2, 0] + R[1, 0] * R[2, 1]
    T[4, 0] = np.sqrt(2.) * R[0, 0] * R[2, 0]
    T[4, 1] = np.sqrt(2.) * R[0, 1] * R[2, 1]
    T[4, 2] = np.sqrt(2.) * R[0, 2] * R[2, 2]
    T[4, 3] = R[0, 2] * R[2, 1] + R[0, 1] * R[2, 2]
    T[4, 4] = R[0, 2] * R[2, 0] + R[0, 0] * R[2, 2]
    T[4, 5] = R[0, 1] * R[2, 0] + R[0, 0] * R[2, 1]
    T[5, 0] = np.sqrt(2.) * R[0, 0] * R[1, 0]
    T[5, 1] = np.sqrt(2.) * R[0, 1] * R[1, 1]
    T[5, 2] = np.sqrt(2.) * R[0, 2] * R[1, 2]
    T[5, 3] = R[0, 2] * R[1, 1] + R[0, 1] * R[1, 2]
    T[5, 4] = R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0]
    T[5, 5] = R[0, 1] * R[1, 0] + R[0, 0] * R[1, 1]
    return T


def rotate_tensor(origin_frame_tensor, orientation_matrix):
    # Copyright (C) 2008  Jette Oddershede after Joel Bernier
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Conversion of symmetric tensor in origin frame to destination frame
    via the use of an orientation matrix U, defined such that V_dest = U @ V_origin

    Copyright (C) 2008 Jette Oddershede September 16 2008 after Joel Bernier

    :param origin_frame_tensor: 3x3 symmetric tensor in origin system
    :param orientation_matrix: 3x3 unitary orientation matrix
    :return: 3x3 symmetric tensor in destination system
    """

    origin_frame_tensor_MV = symmToMVvec(origin_frame_tensor)
    orientation_matrix_MV = MVCOBMatrix(orientation_matrix)
    destination_frame_tensor_MV = np.dot(orientation_matrix_MV, origin_frame_tensor_MV)
    destination_frame_tensor = MVvecToSymm(destination_frame_tensor_MV)
    return destination_frame_tensor


def rotate_tensor_to_lab_frame(grain: npt.NDArray[np.float64], U: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede after Joel Bernier
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Conversion of symmetric tensor in cartesian grain system to sample system
    via the use of the grain orientation matrix U

    Copyright (C) 2008 Jette Oddershede September 16 2008 after Joel Bernier

    :param grain: 3x3 symmetric tensor in grain system
    :param U: 3x3 unitary orientation matrix
    :return: 3x3 symmetric tensor in sample system
    """

    return rotate_tensor(grain, U)


def symmToMVvec(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede after Joel Bernier
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Convert from symmetric matrix to Mandel-Voigt vector
    representation (JVB)

    Copyright (C) 2008 Jette Oddershede, September 16 2008 after Joel Bernier

    :param A: 3x3 symmetric matrix
    :raises TypeError: If `A` isn't a Numpy array
    :raises ValueError: If `A` isn't an array of shape (3,3)
    :raises TypeError: If `A` isn't an array of float64 elements
    :raises ValueError: If `A` isn't symmetric
    :return: 1x6 Mandel-Voigt vector
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("A should be a numpy array!")
    if not np.shape(A) == (3, 3):
        raise ValueError("A should be an array of length (3,3)")
    if not A.dtype == np.dtype("float64"):
        raise TypeError("A should be an array of floats!")
    if not np.allclose(A, A.T):
        raise ValueError("A should be a symmetric matrix!")

    mvvec = np.zeros(6, dtype='float64')
    mvvec[0] = A[0, 0]
    mvvec[1] = A[1, 1]
    mvvec[2] = A[2, 2]
    mvvec[3] = np.sqrt(2.) * A[1, 2]
    mvvec[4] = np.sqrt(2.) * A[0, 2]
    mvvec[5] = np.sqrt(2.) * A[0, 1]
    return mvvec


def MVvecToSymm(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede after Joel Bernier
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Convert from Mandel-Voigt vector to symmetric matrix
    representation (JVB)

    Copyright (C) 2008 Jette Oddershede, September 16 2008 after Joel Bernier

    :param A: 1x6 Mandel-Voigt vector
    :raises TypeError: If `A` isn't a Numpy array
    :raises ValueError: If `A` isn't an array of shape (6,)
    :raises TypeError: If `A` isn't an array of float64 elements
    :return: 3x3 symmetrix matrix
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("A should be a numpy array!")
    if not np.shape(A) == (6,):
        raise ValueError("A should be an array of length (6,)")
    if not A.dtype == np.dtype("float64"):
        raise TypeError("A should be an array of floats!")

    symm = np.zeros((3, 3), dtype='float64')
    symm[0, 0] = A[0]
    symm[1, 1] = A[1]
    symm[2, 2] = A[2]
    symm[1, 2] = A[3] * (1. / np.sqrt(2.))
    symm[0, 2] = A[4] * (1. / np.sqrt(2.))
    symm[0, 1] = A[5] * (1. / np.sqrt(2.))
    symm[2, 1] = A[3] * (1. / np.sqrt(2.))
    symm[2, 0] = A[4] * (1. / np.sqrt(2.))
    symm[1, 0] = A[5] * (1. / np.sqrt(2.))
    return symm


def strain2stress(epsilon: npt.NDArray[np.float64], C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Conversion from strain to stress tensor using the 6x6 stiffness tensor C
    which can be formed by :func:`~.formStiffnessMV`

    Copyright (C) 2008 Jette Oddershede, September 16 2008

    :param epsilon: 3x3 symmetric strain tensor
    :param C: 6x6 stiffness tensor
    :return: 3x3 symmetric stress tensor
    """

    if not isinstance(epsilon, np.ndarray):
        raise TypeError("epsilon should be a numpy array!")
    if not np.shape(epsilon) == (3, 3):
        raise ValueError("epsilon should be an array of length (3,3)")
    if not epsilon.dtype == np.dtype("float64"):
        raise TypeError("epsilon should be an array of floats!")
    if not np.allclose(epsilon, epsilon.T):
        raise ValueError("epsilon should be a symmetric matrix!")

    if not isinstance(C, np.ndarray):
        raise TypeError("C should be a numpy array!")
    if not np.shape(C) == (6, 6):
        raise ValueError("C should be an array of length (6, 6)")
    if not C.dtype == np.dtype("float64"):
        raise TypeError("C should be an array of floats!")
    if not np.allclose(C, C.T):
        raise ValueError("C should be a symmetric matrix!")

    epsilonMV = symmToMVvec(epsilon)
    sigmaMV = np.dot(C, epsilonMV)
    sigma = MVvecToSymm(sigmaMV)
    return sigma


def eps_error_to_sig_error(eps_error: npt.NDArray[np.float64], stiffnessMV: npt.NDArray[np.float64]) -> npt.NDArray[
    np.float64]:
    """The grain stress error (in Pa) as a symmetric tensor in the grain reference system, one standard deviation.
    Propagated through :attr:`~py3DXRDProc.grain.BaseMapGrain.eps` -> :attr:`~py3DXRDProc.grain.BaseMapGrain.sig` conversion

    :param eps_error: 3x3 symmetric strain error tensor (:attr:`py3DXRDProc.grain.BaseMapGrain.eps_error`)
    :param stiffnessMV: 6x6 stiffness tensor (:attr:`py3DXRDProc.grain.BaseMapGrain.phase.stiffnessMV`)
    :raises TypeError: If `eps_error` or `stiffnessMV` isn't a Numpy array of floats
    :raises ValueError: If `eps_error` isn't a (3, 3) array
    :raises ValueError: If `stiffnessMV` isn't a (6, 6) array
    :return: The error in :attr:`py3DXRDProc.grain.BaseMapGrain.sig`
    """

    # These were determined by using sympy to follow the eps tensor symbollicaly through its conversion to strain as per Jette's functions:

    # sig[0, 0] = C[0, 0]*eps[0, 0] + C[0, 1]*eps[1, 1] + C[0, 2]*eps[2, 2] + sqrt(2)*C[0, 3]*eps[1, 2] + sqrt(2)*C[0, 4]*eps[0, 2] + sqrt(2)*C[0, 5]*eps[0, 1]

    # sig[0, 1] = (1/sqrt(2))*C[5, 0]*eps[0, 0] + (1/sqrt(2))*C[5, 1]*eps[1, 1] + (1/sqrt(2))*C[5, 2]*eps[2, 2] + C[5, 3]*eps[1, 2] + C[5, 4]*eps[0, 2] + C[5, 5]*eps[0, 1]

    # sig[0, 2] = (1/sqrt(2))*C[4, 0]*eps[0, 0] + (1/sqrt(2))*C[4, 1]*eps[1, 1] + (1/sqrt(2))*C[4, 2]*eps[2, 2] + C[4, 3]*eps[1, 2] + C[4, 4]*eps[0, 2] + C[4, 5]*eps[0, 1]

    # sig[1, 1] = C[1, 0]*eps[0, 0] + C[1, 1]*eps[1, 1] + C[1, 2]*eps[2, 2] + sqrt(2)*C[1, 3]*eps[1, 2] + sqrt(2)*C[1, 4]*eps[0, 2] + sqrt(2)*C[1, 5]*eps[0, 1]

    # sig[1, 2] = (1/sqrt(2))*C[3, 0]*eps[0, 0] + (1/sqrt(2))*C[3, 1]*eps[1, 1] + (1/sqrt(2))*C[3, 2]*eps[2, 2] + C[3, 3]*eps[1, 2] + C[3, 4]*eps[0, 2] + C[3, 5]*eps[0, 1]

    # sig[2, 2] = C[2, 0]*eps[0, 0] + C[2, 1]*eps[1, 1] + C[2, 2]*eps[2, 2] + sqrt(2)*C[2, 3]*eps[1, 2] + sqrt(2)*C[2, 4]*eps[0, 2] + sqrt(2)*C[2, 5]*eps[0, 1]

    if not isinstance(eps_error, np.ndarray):
        raise TypeError("eps_error must be a Numpy array!")
    if not np.shape(eps_error) == (3, 3):
        raise ValueError("eps_error attribute should be an array of shape (3,3)")
    if not eps_error.dtype == np.dtype("float64"):
        raise TypeError("eps_error should be an array of floats!")
    if not np.allclose(eps_error, eps_error.T):
        raise ValueError("eps_error should be a symmetric matrix!")

    if not isinstance(stiffnessMV, np.ndarray):
        raise TypeError("stiffnessMV must be a Numpy array!")
    if not np.shape(stiffnessMV) == (6, 6):
        raise ValueError("stiffnessMV attribute should be an array of shape (6,6)")
    if not stiffnessMV.dtype == np.dtype("float64"):
        raise TypeError("stiffnessMV should be an array of floats!")
    if not np.allclose(stiffnessMV, stiffnessMV.T):
        raise ValueError("stiffnessMV should be a symmetric matrix!")

    C = stiffnessMV

    deps00 = eps_error[0, 0]
    deps01 = eps_error[0, 1]
    deps02 = eps_error[0, 2]
    deps10 = eps_error[1, 0]
    deps11 = eps_error[1, 1]
    deps12 = eps_error[1, 2]
    deps20 = eps_error[2, 0]
    deps21 = eps_error[2, 1]
    deps22 = eps_error[2, 2]

    sig_error = np.zeros((3, 3), dtype='float64')

    sig_error[0, 0] = np.sqrt(
        np.power(C[0, 0] * deps00, 2) +
        np.power(C[0, 1] * deps11, 2) +
        np.power(C[0, 2] * deps22, 2) +
        2 * np.power(C[0, 3] * deps12, 2) +
        2 * np.power(C[0, 4] * deps02, 2) +
        2 * np.power(C[0, 5] * deps01, 2)
    )

    sig_error[0, 1] = np.sqrt(
        (1 / 2.) * np.power(C[5, 0] * deps00, 2) +
        (1 / 2.) * np.power(C[5, 1] * deps11, 2) +
        (1 / 2.) * np.power(C[5, 2] * deps22, 2) +
        np.power(C[5, 3] * deps12, 2) +
        np.power(C[5, 4] * deps02, 2) +
        np.power(C[5, 5] * deps01, 2)
    )

    sig_error[0, 2] = np.sqrt(
        (1 / 2.) * np.power(C[4, 0] * deps00, 2) +
        (1 / 2.) * np.power(C[4, 1] * deps11, 2) +
        (1 / 2.) * np.power(C[4, 2] * deps22, 2) +
        np.power(C[4, 3] * deps12, 2) +
        np.power(C[4, 4] * deps02, 2) +
        np.power(C[4, 5] * deps01, 2)
    )

    sig_error[1, 0] = sig_error[0, 1]

    sig_error[1, 1] = np.sqrt(
        np.power(C[1, 0] * deps00, 2) +
        np.power(C[1, 1] * deps11, 2) +
        np.power(C[1, 2] * deps22, 2) +
        2 * np.power(C[1, 3] * deps12, 2) +
        2 * np.power(C[1, 4] * deps02, 2) +
        2 * np.power(C[1, 5] * deps01, 2)
    )

    sig_error[1, 2] = np.sqrt(
        (1 / 2.) * np.power(C[3, 0] * deps00, 2) +
        (1 / 2.) * np.power(C[3, 1] * deps11, 2) +
        (1 / 2.) * np.power(C[3, 2] * deps22, 2) +
        np.power(C[3, 3] * deps12, 2) +
        np.power(C[3, 4] * deps02, 2) +
        np.power(C[3, 5] * deps01, 2)
    )

    sig_error[2, 0] = sig_error[0, 2]

    sig_error[2, 1] = sig_error[1, 2]

    sig_error[2, 2] = np.sqrt(
        np.power(C[2, 0] * deps00, 2) +
        np.power(C[2, 1] * deps11, 2) +
        np.power(C[2, 2] * deps22, 2) +
        2 * np.power(C[2, 3] * deps12, 2) +
        2 * np.power(C[2, 4] * deps02, 2) +
        2 * np.power(C[2, 5] * deps01, 2)
    )

    return sig_error


def sig_error_to_sig_lab_error(U: npt.NDArray[np.float64],
                               U_error: npt.NDArray[np.float64],
                               sig: npt.NDArray[np.float64],
                               sig_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The stress strain error (in Pa) as a symmetric tensor in the :attr:`~py3DXRDProc.grain.BaseMapGrain.sample` reference system, one standard deviation.
    Propagated through :attr:`~py3DXRDProc.grain.BaseMapGrain.sig` -> :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab` tensor rotation

    :param U: The grain orientation matrix (:attr:`py3DXRDProc.grain.BaseGrain.U`)
    :param U_error: The error in grain orientation matrix (:attr:`py3DXRDProc.grain.BaseMapGrain.U_error`)
    :param sig: The grain symmetric stress tensor in Pa (:attr:`py3DXRDProc.grain.BaseMapGrain.sig`)
    :param sig_error: The error in grain symmetric stress tensor in Pa (:attr:`py3DXRDProc.grain.BaseMapGrain.sig_error`)
    :raises TypeError: If `U`, `U_error`, `sig`, or `sig_error` aren't Numpy arrays of floats
    :raises ValueError: If `U`, `U_error`, `sig`, or `sig_error` aren't (3, 3) arrays
    :return: The error in :attr:`py3DXRDProc.grain.BaseMapGrain.sig_lab`
    """

    # These were determined by using sympy to follow the sig tensor symbollicaly through its rotation as per Jette's functions:

    # sig_lab[0, 0] = U[0, 0]**2*sig[0, 0] + 2.0*U[0, 0]*U[0, 1]*sig[0, 1] + 2.0*U[0, 0]*U[0, 2]*sig[0, 2] + U[0, 1]**2*sig[1, 1] + 2.0*U[0, 1]*U[0, 2]*sig[1, 2] + U[0, 2]**2*sig[2, 2]

    # sig_lab[0, 1] =(U[0, 0]*U[1, 1] + U[0, 1]*U[1, 0])*sig[0, 1] +(U[0, 0]*U[1, 2] + U[0, 2]*U[1, 0])*sig[0, 2] + (U[0, 1]*U[1, 2] + U[0, 2]*U[1, 1])*sig[1, 2] + U[0, 0]*U[1, 0]*sig[0, 0] + U[0, 1]*U[1, 1]*sig[1, 1] + U[0, 2]*U[1, 2]*sig[2, 2]

    # sig_lab[0, 2] =(U[0, 0]*U[2, 1] + U[0, 1]*U[2, 0])*sig[0, 1] +(U[0, 0]*U[2, 2] + U[0, 2]*U[2, 0])*sig[0, 2] + (U[0, 1]*U[2, 2] + U[0, 2]*U[2, 1])*sig[1, 2] + U[0, 0]*U[2, 0]*sig[0, 0] + U[0, 1]*U[2, 1]*sig[1, 1] + U[0, 2]*U[2, 2]*sig[2, 2]

    # sig_lab[1, 1] = U[1, 0]**2*sig[0, 0] + 2.0*U[1, 0]*U[1, 1]*sig[0, 1] + 2.0*U[1, 0]*U[1, 2]*sig[0, 2] + U[1, 1]**2*sig[1, 1] + 2.0*U[1, 1]*U[1, 2]*sig[1, 2] + U[1, 2]**2*sig[2, 2]

    # sig_lab[1, 2] =(U[1, 0]*U[2, 1] + U[1, 1]*U[2, 0])*sig[0, 1] +(U[1, 0]*U[2, 2] + U[1, 2]*U[2, 0])*sig[0, 2] + (U[1, 1]*U[2, 2] + U[1, 2]*U[2, 1])*sig[1, 2] + U[1, 0]*U[2, 0]*sig[0, 0] + U[1, 1]*U[2, 1]*sig[1, 1] + U[1, 2]*U[2, 2]*sig[2, 2]

    # sig_lab[2, 2] = U[2, 0]**2*sig[0, 0] + 2.0*U[2, 0]*U[2, 1]*sig[0, 1] + 2.0*U[2, 0]*U[2, 2]*sig[0, 2] + U[2, 1]**2*sig[1, 1] + 2.0*U[2, 1]*U[2, 2]*sig[1, 2] + U[2, 2]**2*sig[2, 2]

    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a Numpy array!")
    if not np.shape(U) == (3, 3):
        raise ValueError("U attribute should be an array of shape (3,3)")
    if not U.dtype == np.dtype("float64"):
        raise TypeError("U should be an array of floats!")

    if not isinstance(U_error, np.ndarray):
        raise TypeError("U_error must be a Numpy array!")
    if not np.shape(U_error) == (3, 3):
        raise ValueError("U_error attribute should be an array of shape (3,3)")
    if not U_error.dtype == np.dtype("float64"):
        raise TypeError("U_error should be an array of floats!")

    if not isinstance(sig, np.ndarray):
        raise TypeError("sig must be a Numpy array!")
    if not np.shape(sig) == (3, 3):
        raise ValueError("sig attribute should be an array of shape (3,3)")
    if not sig.dtype == np.dtype("float64"):
        raise TypeError("sig should be an array of floats!")
    if not np.allclose(sig, sig.T):
        raise ValueError("sig should be a symmetric matrix!")

    if not isinstance(sig_error, np.ndarray):
        raise TypeError("sig_error must be a Numpy array!")
    if not np.shape(sig_error) == (3, 3):
        raise ValueError("sig_error attribute should be an array of shape (3,3)")
    if not sig_error.dtype == np.dtype("float64"):
        raise TypeError("sig_error should be an array of floats!")
    if not np.allclose(sig_error, sig_error.T):
        raise ValueError("sig_error should be a symmetric matrix!")

    sig_lab_error = np.zeros((3, 3), dtype='float64')

    U00 = U[0, 0]
    U01 = U[0, 1]
    U02 = U[0, 2]
    U10 = U[1, 0]
    U11 = U[1, 1]
    U12 = U[1, 2]
    U20 = U[2, 0]
    U21 = U[2, 1]
    U22 = U[2, 2]

    dU00 = U_error[0, 0]
    dU01 = U_error[0, 1]
    dU02 = U_error[0, 2]
    dU10 = U_error[1, 0]
    dU11 = U_error[1, 1]
    dU12 = U_error[1, 2]
    dU20 = U_error[2, 0]
    dU21 = U_error[2, 1]
    dU22 = U_error[2, 2]

    # Fractional errors:

    fU00 = dU00 / U00
    fU01 = dU01 / U01
    fU02 = dU02 / U02
    fU10 = dU10 / U10
    fU11 = dU11 / U11
    fU12 = dU12 / U12
    fU20 = dU20 / U20
    fU21 = dU21 / U21
    fU22 = dU22 / U22

    sig00 = sig[0, 0]
    sig01 = sig[0, 1]
    sig02 = sig[0, 2]
    sig10 = sig[1, 0]
    sig11 = sig[1, 1]
    sig12 = sig[1, 2]
    sig20 = sig[2, 0]
    sig21 = sig[2, 1]
    sig22 = sig[2, 2]

    dsig00 = sig_error[0, 0]
    dsig01 = sig_error[0, 1]
    dsig02 = sig_error[0, 2]
    dsig10 = sig_error[1, 0]
    dsig11 = sig_error[1, 1]
    dsig12 = sig_error[1, 2]
    dsig20 = sig_error[2, 0]
    dsig21 = sig_error[2, 1]
    dsig22 = sig_error[2, 2]

    fsig00 = dsig00 / sig00
    fsig01 = dsig01 / sig01
    fsig02 = dsig02 / sig02
    fsig10 = dsig10 / sig10
    fsig11 = dsig11 / sig11
    fsig12 = dsig12 / sig12
    fsig20 = dsig20 / sig20
    fsig21 = dsig21 / sig21
    fsig22 = dsig22 / sig22

    sig_lab_error[0, 0] = np.sqrt(
        ((((U00 ** 2) * sig00) ** 2) * ((2 * fU00) ** 2 + fsig00 ** 2)) +
        ((((U01 ** 2) * sig11) ** 2) * ((2 * fU01) ** 2 + fsig11 ** 2)) +
        ((((U02 ** 2) * sig22) ** 2) * ((2 * fU02) ** 2 + fsig22 ** 2)) +
        (4 * (((U00 * U01 * sig01) ** 2) * (fU00 ** 2 + fU01 ** 2 + fsig01 ** 2))) +
        (4 * (((U00 * U02 * sig02) ** 2) * (fU00 ** 2 + fU02 ** 2 + fsig02 ** 2))) +
        (4 * (((U01 * U02 * sig12) ** 2) * (fU01 ** 2 + fU02 ** 2 + fsig12 ** 2)))
    )

    sig_lab_error[1, 1] = np.sqrt(
        ((((U10 ** 2) * sig00) ** 2) * ((2 * fU10) ** 2 + fsig00 ** 2)) +
        ((((U11 ** 2) * sig11) ** 2) * ((2 * fU11) ** 2 + fsig11 ** 2)) +
        ((((U12 ** 2) * sig22) ** 2) * ((2 * fU12) ** 2 + fsig22 ** 2)) +
        (4 * (((U10 * U11 * sig01) ** 2) * (fU10 ** 2 + fU11 ** 2 + fsig01 ** 2))) +
        (4 * (((U10 * U12 * sig02) ** 2) * (fU10 ** 2 + fU12 ** 2 + fsig02 ** 2))) +
        (4 * (((U11 * U12 * sig12) ** 2) * (fU11 ** 2 + fU12 ** 2 + fsig12 ** 2)))
    )

    sig_lab_error[2, 2] = np.sqrt(
        ((((U20 ** 2) * sig00) ** 2) * ((2 * fU20) ** 2 + fsig00 ** 2)) +
        ((((U21 ** 2) * sig11) ** 2) * ((2 * fU21) ** 2 + fsig11 ** 2)) +
        ((((U22 ** 2) * sig22) ** 2) * ((2 * fU22) ** 2 + fsig22 ** 2)) +
        (4 * (((U20 * U21 * sig01) ** 2) * (fU20 ** 2 + fU21 ** 2 + fsig01 ** 2))) +
        (4 * (((U20 * U22 * sig02) ** 2) * (fU20 ** 2 + fU22 ** 2 + fsig02 ** 2))) +
        (4 * (((U21 * U22 * sig12) ** 2) * (fU21 ** 2 + fU22 ** 2 + fsig12 ** 2)))
    )

    sig_lab_error[0, 1] = np.sqrt(
        ((sig01 ** 2) * ((U00 * U11 + U01 * U10) ** 2) * ((((((U00 * U11) ** 2) * (fU00 ** 2 + fU11 ** 2)) + (
                ((U01 * U10) ** 2) * (fU01 ** 2 + fU10 ** 2))) / ((
                                                                          U00 * U11 + U01 * U10) ** 2)) + fsig01 ** 2)) +
        ((sig02 ** 2) * ((U00 * U12 + U02 * U10) ** 2) * ((((((U00 * U12) ** 2) * (fU00 ** 2 + fU12 ** 2)) + (
                ((U02 * U10) ** 2) * (fU02 ** 2 + fU10 ** 2))) / ((
                                                                          U00 * U12 + U02 * U10) ** 2)) + fsig02 ** 2)) +
        ((sig12 ** 2) * ((U01 * U12 + U02 * U11) ** 2) * ((((((U01 * U12) ** 2) * (fU01 ** 2 + fU12 ** 2)) + (
                ((U02 * U11) ** 2) * (fU02 ** 2 + fU11 ** 2))) / ((
                                                                          U01 * U12 + U02 * U11) ** 2)) + fsig12 ** 2)) +
        (((U00 * U10 * sig00) ** 2) * (fU00 ** 2 + fU10 ** 2 + fsig00 ** 2)) +
        (((U01 * U11 * sig11) ** 2) * (fU01 ** 2 + fU11 ** 2 + fsig11 ** 2)) +
        (((U02 * U12 * sig22) ** 2) * (fU02 ** 2 + fU12 ** 2 + fsig22 ** 2))
    )

    sig_lab_error[0, 2] = np.sqrt(
        ((sig01 ** 2) * ((U00 * U21 + U01 * U20) ** 2) * ((((((U00 * U21) ** 2) * (fU00 ** 2 + fU21 ** 2)) + (
                ((U01 * U20) ** 2) * (fU01 ** 2 + fU20 ** 2))) / ((
                                                                          U00 * U21 + U01 * U20) ** 2)) + fsig01 ** 2)) +
        ((sig02 ** 2) * ((U00 * U22 + U02 * U20) ** 2) * ((((((U00 * U22) ** 2) * (fU00 ** 2 + fU22 ** 2)) + (
                ((U02 * U20) ** 2) * (fU02 ** 2 + fU20 ** 2))) / ((
                                                                          U00 * U22 + U02 * U20) ** 2)) + fsig02 ** 2)) +
        ((sig12 ** 2) * ((U01 * U22 + U02 * U21) ** 2) * ((((((U01 * U22) ** 2) * (fU01 ** 2 + fU22 ** 2)) + (
                ((U02 * U21) ** 2) * (fU02 ** 2 + fU21 ** 2))) / ((
                                                                          U01 * U22 + U02 * U21) ** 2)) + fsig12 ** 2)) +
        (((U00 * U20 * sig00) ** 2) * (fU00 ** 2 + fU20 ** 2 + fsig00 ** 2)) +
        (((U01 * U21 * sig11) ** 2) * (fU01 ** 2 + fU21 ** 2 + fsig11 ** 2)) +
        (((U02 * U22 * sig22) ** 2) * (fU02 ** 2 + fU22 ** 2 + fsig22 ** 2))
    )

    sig_lab_error[1, 0] = sig_lab_error[0, 1]

    sig_lab_error[1, 2] = np.sqrt(
        ((sig01 ** 2) * ((U10 * U21 + U11 * U20) ** 2) * ((((((U10 * U21) ** 2) * (fU10 ** 2 + fU21 ** 2)) + (
                ((U11 * U20) ** 2) * (fU11 ** 2 + fU20 ** 2))) / ((
                                                                          U10 * U21 + U11 * U20) ** 2)) + fsig01 ** 2)) +
        ((sig02 ** 2) * ((U10 * U22 + U12 * U20) ** 2) * ((((((U10 * U22) ** 2) * (fU10 ** 2 + fU22 ** 2)) + (
                ((U12 * U20) ** 2) * (fU12 ** 2 + fU20 ** 2))) / ((
                                                                          U10 * U22 + U12 * U20) ** 2)) + fsig02 ** 2)) +
        ((sig12 ** 2) * ((U11 * U22 + U12 * U21) ** 2) * ((((((U11 * U22) ** 2) * (fU11 ** 2 + fU22 ** 2)) + (
                ((U12 * U21) ** 2) * (fU12 ** 2 + fU21 ** 2))) / ((
                                                                          U11 * U22 + U12 * U21) ** 2)) + fsig12 ** 2)) +
        (((U10 * U20 * sig00) ** 2) * (fU10 ** 2 + fU20 ** 2 + fsig00 ** 2)) +
        (((U11 * U21 * sig11) ** 2) * (fU11 ** 2 + fU21 ** 2 + fsig11 ** 2)) +
        (((U12 * U22 * sig22) ** 2) * (fU12 ** 2 + fU22 ** 2 + fsig22 ** 2))
    )

    sig_lab_error[2, 0] = sig_lab_error[0, 2]

    sig_lab_error[2, 1] = sig_lab_error[1, 2]

    return sig_lab_error


def formStiffnessMV(crystal_system: str, c11: float = None, c12: float = None, c13: float = None, c14: float = None,
                    c15: float = None, c16: float = None,
                    c22: float = None, c23: float = None, c24: float = None, c25: float = None, c26: float = None,
                    c33: float = None, c34: float = None, c35: float = None, c36: float = None,
                    c44: float = None, c45: float = None, c46: float = None,
                    c55: float = None, c56: float = None,
                    c66: float = None) -> npt.NDArray[np.float64]:
    # Copyright (C) 2008  Jette Oddershede
    # Ported from FitAllB/conversion.py at https://github.com/FABLE-3DXRD/FitAllB/

    """Form the stiffness matrix to convert from strain to stress in the Mandel-Voigt notation
    for a given crystal system using the unique input stiffness constants (Voigt convention)

    Copyright (C) 2008 Jette Oddershede, September 16 2008 after MATLAB routine by Joel Bernier

    :param crystal_system: Crystal system, one of `["isotropic", "cubic", "hexagonal", "trigonal_high", "trigonal_low", "tetragonal_high", "tetragonal_low", "orthorhombic", "monoclinic", "triclinic"]`
    :param c11: C11 stiffness constant (Pa)
    :param c12: C12 stiffness constant (Pa)
    :param c13: C13 stiffness constant (Pa)
    :param c14: C14 stiffness constant (Pa)
    :param c15: C15 stiffness constant (Pa)
    :param c16: C16 stiffness constant (Pa)
    :param c22: C22 stiffness constant (Pa)
    :param c23: C23 stiffness constant (Pa)
    :param c24: C24 stiffness constant (Pa)
    :param c25: C25 stiffness constant (Pa)
    :param c26: C26 stiffness constant (Pa)
    :param c33: C33 stiffness constant (Pa)
    :param c34: C34 stiffness constant (Pa)
    :param c35: C35 stiffness constant (Pa)
    :param c36: C36 stiffness constant (Pa)
    :param c44: C44 stiffness constant (Pa)
    :param c45: C45 stiffness constant (Pa)
    :param c46: C46 stiffness constant (Pa)
    :param c55: C55 stiffness constant (Pa)
    :param c56: C56 stiffness constant (Pa)
    :param c66: C66 stiffness constant (Pa)
    :raises ValueError: If an unsupported `crystal_system` is supplied
    :return: 6x6 stiffness matrix
    """

    if crystal_system == 'isotropic':
        unique_list = 'c11,c12'
        unique = [c11, c12]
        full = [c11, c12, c12, 0, 0, 0, c11, c12, 0, 0, 0, c11, 0, 0, 0, (c11 - c12) / 2, 0, 0, (c11 - c12) / 2, 0,
                (c11 - c12) / 2]
    elif crystal_system == 'cubic':
        unique_list = 'c11,c12,c44'
        unique = [c11, c12, c44]
        full = [c11, c12, c12, 0, 0, 0, c11, c12, 0, 0, 0, c11, 0, 0, 0, c44, 0, 0, c44, 0, c44]
    elif crystal_system == 'hexagonal':
        unique_list = 'c11,c12,c13,c33,c44'
        unique = [c11, c12, c13, c33, c44]
        full = [c11, c12, c13, 0, 0, 0, c11, c13, 0, 0, 0, c33, 0, 0, 0, c44, 0, 0, c44, 0, (c11 - c12) / 2]
    elif crystal_system == 'trigonal_high':
        unique_list = 'c11,c12,c13,c14,c33,c44'
        unique = [c11, c12, c13, c14, c33, c44]
        full = [c11, c12, c13, c14, 0, 0, c11, c13, -c14, 0, 0, c33, 0, 0, 0, c44, 0, 0, c44, c14 / 2, (c11 - c12) / 2]
    elif crystal_system == 'trigonal_low':
        unique_list = 'c11,c12,c13,c14,c25,c33,c44'
        unique = [c11, c12, c13, c14, c25, c33, c44]
        full = [c11, c12, c13, c14, -c25, 0, c11, c13, -c14, c25, 0, c33, 0, 0, 0, c44, 0, c25 / 2, c44, c14 / 2,
                (c11 - c12) / 2]
    elif crystal_system == 'tetragonal_high':
        unique_list = 'c11,c12,c13,c33,c44,c66'
        unique = [c11, c12, c13, c33, c44, c66]
        full = [c11, c12, c13, 0, 0, 0, c11, c13, 0, 0, 0, c33, 0, 0, 0, c44, 0, 0, c44, 0, c66]
    elif crystal_system == 'tetragonal_low':
        unique_list = 'c11,c12,c13,c16, c33,c44,c66'
        unique = [c11, c12, c13, c16, c33, c44, c66]
        full = [c11, c12, c13, 0, 0, c16, c11, c13, 0, 0, -c16, c33, 0, 0, 0, c44, 0, 0, c44, 0, c66]
    elif crystal_system == 'orthorhombic':
        unique_list = 'c11,c12,c13,c22,c23,c33,c44,c55,c66'
        unique = [c11, c12, c13, c22, c23, c33, c44, c55, c66]
        full = [c11, c12, c13, 0, 0, 0, c22, c23, 0, 0, 0, c33, 0, 0, 0, c44, 0, 0, c55, 0, c66]
    elif crystal_system == 'monoclinic':
        unique_list = 'c11,c12,c13,c15,c22,c23,c25,c33,c35,c44,c46,c55,c66'
        unique = [c11, c12, c13, c15, c22, c23, c25, c33, c35, c44, c46, c55, c66]
        full = [c11, c12, c13, 0, c15, 0, c22, c23, 0, c25, 0, c33, 0, c35, 0, c44, 0, c46, c55, 0, c66]
    elif crystal_system == 'triclinic':
        unique_list = 'c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66'
        unique = [c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56,
                  c66]
        full = unique
    else:
        raise ValueError(f'crystal system {crystal_system} not supported')

    assert None not in unique, 'For crytal_system %s, the following must be given:\n %s' % (crystal_system, unique_list)

    full = np.array(full)
    stiffness = np.zeros((6, 6))
    stiffness[0][0:3] = full[0:3]
    stiffness[0][3:6] = full[3:6] * np.sqrt(2.)
    stiffness[1][1:3] = full[6:8]
    stiffness[1][3:6] = full[8:11] * np.sqrt(2.)
    stiffness[2][2] = full[11]
    stiffness[2][3:6] = full[12:15] * np.sqrt(2.)
    stiffness[3][3:6] = full[15:18] * 2.
    stiffness[4][4:6] = full[18:20] * 2.
    stiffness[5][5] = full[20] * 2.
    for i in range(1, 6):
        for j in range(0, i):
            stiffness[i][j] = stiffness[j][i]

    return stiffness


def ps_hdf5_to_flt(h5in: str, key: str, pks_path: str, outname: str, spline: str):
    # Copyright (C) 2005-2019  Jon Wright
    # Modified from ImageD11/sandbox/ma4750/to_fly.py at https://github.com/jonwright/ImageD11/

    """Convert new peaksearch output format (HDF5) to old flt-based format

    Copyright (C) 2005-2019  Jon Wright

    :param h5in: Path to HDF5 input file
    :param key: Key of specific scan
    :param pks_path: Path to peaks within scan dataset
    :param outname: Path to flt output file
    :param spline: Path to spline file for optional distortion correction
    """

    import h5py, tqdm, numpy as np
    from ImageD11 import columnfile, blobcorrector

    cor = blobcorrector.correctorclass(spline)

    with h5py.File(h5in, 'r') as hin:
        log.debug(list(hin), pks_path)
        g = hin[key][pks_path]
        cd = {name: g[name][:] for name in list(g)}
        cd['sc'] = np.zeros_like(cd['s_raw'])
        cd['fc'] = np.zeros_like(cd['f_raw'])

    inc = columnfile.colfile_from_dict(cd)

    for i in tqdm.tqdm(range(inc.nrows)):
        inc.sc[i], inc.fc[i] = cor.correct(inc.s_raw[i], inc.f_raw[i])

    columnfile.colfile_to_hdf(inc, outname)


@njit(fastmath=True)
def rot_array_to_matrix(rot_array):
    """Converts an array of 3 ZXZ Euler angles to a rotation matrix.
    Follows Proper Euler Angle ZXZ convention with angles (alpha, beta, gamma) representing
    R = Z(alpha) @ X(beta) @ Z(gamma)"""
    # convert the rotation array into a matrix
    zxz_angle_radians = np.radians(rot_array)

    alpha = zxz_angle_radians[0]
    beta = zxz_angle_radians[1]
    gamma = zxz_angle_radians[2]

    c1 = np.cos(alpha)
    c2 = np.cos(beta)
    c3 = np.cos(gamma)

    s1 = np.sin(alpha)
    s2 = np.sin(beta)
    s3 = np.sin(gamma)

    rot_matrix = np.zeros((3, 3), np.float64)

    rot_matrix[0, 0] = c1 * c3 - c2 * s1 * s3
    rot_matrix[0, 1] = -1.0 * c1 * s3 - c2 * c3 * s1
    rot_matrix[0, 2] = s1 * s2
    rot_matrix[1, 0] = c3 * s1 + c1 * c2 * s3
    rot_matrix[1, 1] = c1 * c2 * c3 - s1 * s3
    rot_matrix[1, 2] = -1.0 * c1 * s2
    rot_matrix[2, 0] = s2 * s3
    rot_matrix[2, 1] = c3 * s2
    rot_matrix[2, 2] = c2

    return rot_matrix


@njit(fastmath=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.
    https://stackoverflow.com/a/13849249/12374817"""
    return vector / np.linalg.norm(vector)


@njit(fastmath=True)
def angle_between_vectors_rad(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
        https://stackoverflow.com/a/13849249/12374817
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dp = np.dot(v1_u, v2_u)
    if dp < -1.0:
        dp = -1.0
    elif dp > 1.0:
        dp = 1.0
    return np.arccos(dp)


@njit(fastmath=True)
def misorientation_from_delta(delta):
    # Copyright(c) 2013 - 2019 Henry Proudhon
    # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
    """Determine misorientation angle in degrees from 3x3 delta matrix.

    Copyright(c) 2013 - 2019 Henry Proudhon

    :param delta: Input 3x3 Numpy array rotation matrix
    :return: The misorientation angle in degrees
    """
    cw = (np.trace(delta) - 1.0) / 2.0

    # guard against arccos returning NaN:
    # cw can sometimes be slightly more than 1 or slightly less than -1
    # clip to [-1,+1]
    if cw > 1.:
        cw = 1.
    if cw < -1.:
        cw = -1.

    mis = np.arccos(cw)
    misd = np.rad2deg(mis)

    return misd


def disorientation_single(grain_a, grain_b):
    # Copyright(c) 2013 - 2019 Henry Proudhon
    # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
    """Reference disorientation function. All other disorientation functions must agree with this.
    Calculate disorientation between two grains. Grains must have same symmetry to do this!
    Returns the misorientation angle in degrees.

    Copyright(c) 2013 - 2019 Henry Proudhon

    :param grain_a: First grain
    :param grain_b: Second grain

    :return: The misorientation angle in degrees
    """
    gA = grain_a.U_sample.T
    gB = grain_b.U_sample.T
    symmetries = grain_a.phase.symmetry.symmetry_operators()
    n_symms = symmetries.shape[0]

    angle_array = np.zeros(shape=n_symms)

    for k in range(n_symms):
        sym_k = symmetries[k]
        o_k = np.dot(sym_k, gB)
        delta = np.dot(o_k, gA.T)

        misd = misorientation_from_delta(delta)

        angle_array[k] = misd

    the_angle_deg = np.min(angle_array)

    return the_angle_deg


@njit(fastmath=True)
def disorientation_single_numba(matrix_tuple: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                                symmetries: npt.NDArray[np.float64]):
    # Copyright(c) 2013 - 2019 Henry Proudhon
    # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
    """
    Calculate disorientation between a single pair of grain orientations, JITted with numba for speed.
    Returns the misorientation angle in degrees.

    Copyright(c) 2013 - 2019 Henry Proudhon

    :param matrix_tuple: 2-tuple of 3x3 orientation matrices as Numpy arrays
    :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class

    :return: The misorientation angle in degrees
    """

    gA, gB = matrix_tuple
    gA = gA.T
    gB = gB.T

    n_symms = symmetries.shape[0]

    angle_array = np.zeros(shape=n_symms)

    for k in range(n_symms):
        sym_k = symmetries[k]
        o_k = np.dot(sym_k, gB)
        delta = np.dot(o_k, gA.T)

        misd = misorientation_from_delta(delta)

        angle_array[k] = misd

    the_angle_deg = np.min(angle_array)

    return the_angle_deg


@njit(fastmath=True)
def disorientation_single_check_numba(matrix_tuple: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                                      misd_tol: float,
                                      symmetries: npt.NDArray[np.float64]):
    # Copyright(c) 2013 - 2019 Henry Proudhon
    # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
    """Checks disorientation between a single pair of orientations, JITted with numba for speed.

    Copyright(c) 2013 - 2019 Henry Proudhon

    :param matrix_tuple: 2-tuple of 3x3 orientation matrices as Numpy arrays
    :param misd_tol: float misorientation tolerance in degrees
    :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class

    :return: True if less than tolerance, False otherwise
    """

    gA, gB = matrix_tuple
    gA = gA.T
    gB = gB.T

    for k in range(symmetries.shape[0]):
        sym_k = symmetries[k]
        o_k = np.dot(sym_k, gB)
        delta = np.dot(o_k, gA.T)

        misd = misorientation_from_delta(delta)

        if misd < misd_tol:
            return True
    return False


# TODO: below currently unused.
# @njit(fastmath=True, parallel=True)
# def disorientation_array(all_orientations: npt.NDArray[np.float64],
#                          pair_indices: npt.NDArray[np.float64],
#                          symmetries: npt.NDArray[np.float64]):
#     # Copyright(c) 2013 - 2019 Henry Proudhon
#     # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
#     """Calculate disorientation between an array of pair indices, JITted with numba for speed.
#     Returns the misorientation in degrees as a 1D array for each pair
#
#     Copyright(c) 2013 - 2019 Henry Proudhon
#
#     :param all_orientations: Nx(3x3) Numpy array of 3x3 orientation matrices
#     :param pair_indices: Mx2 Numpy array of indices for the all_orientations array
#     :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class
#
#     :return: The misorientation angle in degrees as a 1D M-length Numpy array
#     """
#
#     n_pairs = pair_indices.shape[0]
#
#     misd_array = np.zeros(n_pairs)
#
#     for pair_index in prange(n_pairs):
#         index_A, index_B = pair_indices[pair_index]
#         gA = all_orientations[index_A].T
#         gB = all_orientations[index_B].T
#
#         misd = disorientation_single_numba((gA, gB), symmetries)
#
#         misd_array[pair_index] = misd
#
#     return misd_array


# TODO: below currently unused.
# @njit(fastmath=True, parallel=True)
# def disorientation_check_array(all_orientations: npt.NDArray[np.float64],
#                                pair_indices: npt.NDArray[np.float64],
#                                misd_tol: float,
#                                symmetries: npt.NDArray[np.float64]):
#     # Copyright(c) 2013 - 2019 Henry Proudhon
#     # Adapted from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
#
#     """Calculate disorientation between an array of pair indices, JITted with numba for speed.
#        Returns true if less than tolerance, false otherwise as a 1D array for each pair.
#
#     Copyright(c) 2013 - 2019 Henry Proudhon
#
#     :param all_orientations: Nx(3x3) Numpy array of 3x3 orientation matrices
#     :param pair_indices: Mx2 Numpy array of indices for the all_orientations array
#     :param misd_tol: float misorientation tolerance in degrees
#     :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class
#
#     :return: 1D M-length Numpy boolean array, True/False for each pair in pair_indices
#     """
#
#     n_pairs = pair_indices.shape[0]
#
#     pairs_match_orien_check = np.zeros(n_pairs, dtype=np.bool_)
#
#     for pair_index in prange(n_pairs):
#         index_A, index_B = pair_indices[pair_index]
#         gA = all_orientations[index_A].T
#         gB = all_orientations[index_B].T
#
#         pairs_match = disorientation_single_check_numba((gA, gB), misd_tol, symmetries)
#
#         pairs_match_orien_check[pair_index] = pairs_match
#
#     return pairs_match_orien_check


def are_grains_duplicate(grain_a, grain_b, dist_tol=0.10, angle_tol=1.0):
    """Reference function for grain de-duplication. Uses flat distance tolerance and angle tolerance.
    Returns True if grain centre-of-mass separation less than dist_tol, and if grain misorientation less than angle_tol.

    :param grain_a: First grain
    :param grain_b: Second grain
    :param dist_tol: Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees

    :raises ValueError: If grain_a and grain_b are the same grain. This shouldn't happen when this function is called.

    :return: True if distance and misorientation less than tolerances, False otherwise
    """
    if grain_a is grain_b:
        raise ValueError("Both grains the same!")

    # check they have the same phase
    if grain_a.phase != grain_b.phase:
        return False

    # check their separation less than distance tolerance
    pos_a = grain_a.pos_sample
    pos_b = grain_b.pos_sample
    grain_com_separation = spatial.distance.euclidean(pos_a, pos_b)
    if grain_com_separation > dist_tol:
        return False

    # check their misorientations
    U_a = grain_a.U_sample
    U_b = grain_b.U_sample
    cs = grain_a.phase.symmetry.symmetry_operators()
    misd = disorientation_single_numba((U_a, U_b), cs)
    if misd > angle_tol:
        return False

    # all three checks pass, so return true
    return True


def are_grains_duplicate_array_numba_wrapper(grains_list, dist_tol=0.10, angle_tol=1.0):
    """Wrapper for are_grains_duplicate_array_numba.
    Lets us prepare a list of pairs of indices to grains_list to check, using nice itertools functions.

    :param grains_list: List of grains to check.
    :param dist_tol: Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees

    :return: List of indices of matching grain pairs.
    """
    if not isinstance(dist_tol, float):
        raise TypeError("dist_tol should be a float!")
    if not isinstance(angle_tol, float):
        raise TypeError("angle_tol should be a float")

    pos_array = np.array([grain.pos_sample for grain in grains_list])
    U_array = np.array([grain.U_sample for grain in grains_list])

    symmetries = grains_list[0].phase.symmetry.symmetry_operators()

    # prepare list of input pair indices
    # e.g [(1, 2), (1, 3), (2, 3)]

    index_array = np.arange(pos_array.shape[0])

    idx = np.stack(np.triu_indices(len(index_array), k=1), axis=-1)

    pair_indices = index_array[idx]

    pairs_match_bool_array = are_grains_duplicate_array_numba(pos_array, U_array,
                                                              pair_indices, symmetries,
                                                              dist_tol, angle_tol)

    matching_indices_masked = pair_indices[pairs_match_bool_array, :]
    return matching_indices_masked


@njit(fastmath=True, parallel=True)
def are_grains_duplicate_array_numba(pos_array, U_array, pair_indices, symmetries, dist_tol=0.10, angle_tol=1.0):
    """Array-ified numba-ified version of are_grains_duplicate.
    From an array of positions and orientations, returns 1D M-length array of True/False for each pair index in pair_indices.
    Checks each pair_index from pair_indices in parallel.

    :param pos_array: Nx3 Numpy array of grain positions
    :param U_array: Nx(3x3) Nump array of grain orientations
    :param pair_indices: Mx2 Numpy array of indices for pos_array and U_array
    :param dist_tol: Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees
    :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class

    :return: 1D M-length Numpy boolean array, True/False for each pair in pair_indices
    """

    # check positions first
    n_pairs = pair_indices.shape[0]

    pairs_match_bool_array = np.zeros(n_pairs, dtype=np.bool_)
    for pair_index in prange(n_pairs):
        gi, gj = pair_indices[pair_index]
        # get grain positions
        pos_i = pos_array[gi]
        pos_j = pos_array[gj]
        distance = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        # check distances
        if distance < dist_tol:
            # distance matches, so check misorientations
            U_a = U_array[gi]
            U_b = U_array[gj]

            pairs_match_bool_array[pair_index] = disorientation_single_check_numba((U_a, U_b), angle_tol, symmetries)

    return pairs_match_bool_array


def are_grains_duplicate_array_2lists_numba_wrapper(grains_list_A, grains_list_B, dist_tol=0.10, angle_tol=1.0):
    """Wrapper for are_grains_duplicate_array_2lists_numba.
    Lets us prepare a list of pairs of indices from grains_list_A and grains_list_B to check, using nice itertools functions.

    :param grains_list_A: List A of grains to check.
    :param grains_list_B: List B of grains to check.
    :param dist_tol: Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees

    :return: List of indices of matching grain pairs between grains_list_A and grains_list_B
    """
    if not isinstance(dist_tol, float):
        raise TypeError("dist_tol should be a float!")
    if not isinstance(angle_tol, float):
        raise TypeError("angle_tol should be a float")

    pos_array_A = np.array([grain.pos_sample for grain in grains_list_A])
    pos_array_B = np.array([grain.pos_sample for grain in grains_list_B])
    U_array_A = np.array([grain.U_sample for grain in grains_list_A])
    U_array_B = np.array([grain.U_sample for grain in grains_list_B])

    symmetries = grains_list_A[0].phase.symmetry.symmetry_operators()

    # prepare list of input pair indices from both input lists
    pair_indices = np.array(list(product(range(pos_array_A.shape[0]), range(pos_array_B.shape[0]))))

    # e.g [(1, 2), (1, 3), (2, 3)]

    pairs_match_bool_array = are_grains_duplicate_array_2lists_numba(pos_array_A, pos_array_B,
                                                                     U_array_A, U_array_B,
                                                                     pair_indices, symmetries,
                                                                     dist_tol, angle_tol)

    matching_indices_masked = pair_indices[pairs_match_bool_array, :]
    return matching_indices_masked


@njit(fastmath=True, parallel=True)
def are_grains_duplicate_array_2lists_numba(pos_array_A, pos_array_B,
                                            U_array_A, U_array_B,
                                            pair_indices, symmetries,
                                            dist_tol=0.10, angle_tol=1.0):
    """Array-ified numba-ified version of are_grains_duplicate but using two separate grains lists.
    From two arrays of positions and orientations, returns 1D M-length array of True/False for each pair index in pair_indices.

    :param pos_array_A: Nx3 Numpy array of grain positions from list A
    :param pos_array_B: Mx3 Numpy array of grain positions from list B
    :param U_array_A: Nx(3x3) Numpy array of grain orientations from list A
    :param U_array_B: Mx(3x3) Numpy array of grain orientations from list B
    :param pair_indices: Kx2 Numpy array of indices for pos and U arrays, in format [(index_A, index_B),...]
    :param dist_tol: Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees
    :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class

    :return: 1D K-length Numpy boolean array, True/False for each pair in pair_indices
    """
    # check positions first
    n_pairs = pair_indices.shape[0]

    pairs_match_bool_array = np.zeros(n_pairs, dtype=np.bool_)
    for pair_index in prange(n_pairs):
        gi, gj = pair_indices[pair_index]
        # get grain positions
        pos_i = pos_array_A[gi]
        pos_j = pos_array_B[gj]
        distance = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        # check distances
        if distance < dist_tol:
            # distance matches, so check misorientations
            U_a = U_array_A[gi]
            U_b = U_array_B[gj]

            pairs_match_bool_array[pair_index] = disorientation_single_check_numba((U_a, U_b), angle_tol, symmetries)

    return pairs_match_bool_array


def are_grains_duplicate_stitching(grain_a, grain_b, dist_tol_xy=0.10, dist_tol_z=0.20, angle_tol=1.0):
    """Reference function for grain de-duplication. Uses horizontal and vertical distance tolerances and angle tolerance.
    Returns True if grain XY centre-of-mass separation less than dist_tol_xy,
    grain Z centre-of-mass separation less than dist_tol_z, and if grain misorientation less than angle_tol.

    :param grain_a: First grain
    :param grain_b: Second grain
    :param dist_tol_xy: XY Distance tolerance (mm)
    :param dist_tol_z: Z Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees

    :raises ValueError: If grain_a and grain_b are the same grain. This shouldn't happen when this function is called.

    :return: True if distance and misorientation less than tolerances, False otherwise
    """

    if grain_a is grain_b:
        raise ValueError("Both grains the same!")

    # check they have the same phase
    if grain_a.phase != grain_b.phase:
        return False

    # check their vertical separation
    pos_a_z = grain_a.pos_sample[2]
    pos_b_z = grain_b.pos_sample[2]
    grain_com_separation_z = np.abs(pos_a_z - pos_b_z)
    if grain_com_separation_z > dist_tol_z:
        return False

    # check their horizontal separation
    pos_a_xy = grain_a.pos_sample[0:2]
    pos_b_xy = grain_b.pos_sample[0:2]
    grain_com_separation_xy = spatial.distance.euclidean(pos_a_xy, pos_b_xy)
    if grain_com_separation_xy > dist_tol_xy:
        return False

    # check their misorientations
    U_a = grain_a.U_sample
    U_b = grain_b.U_sample
    cs = grain_a.phase.symmetry.symmetry_operators()
    misd = disorientation_single_numba((U_a, U_b), cs)
    if misd > angle_tol:
        return False

    # all three checks pass, so return true
    return True


def are_grains_duplicate_stitching_array_numba_wrapper(grains_list, dist_tol_xy=0.10, dist_tol_z=0.20, angle_tol=1.0):
    """Wrapper for are_grains_duplicate_stitching_array_numba_wrapper.
    Lets us prepare a list of pairs of indices to grains_list to check, using nice itertools functions.

    :param grains_list: List of grains to check.
    :param dist_tol_xy: XY Distance tolerance (mm)
    :param dist_tol_z: Z Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees

    :return: List of indices of matching grain pairs.
    """
    pos_array = np.array([grain.pos_sample for grain in grains_list])
    U_array = np.array([grain.U_sample for grain in grains_list])

    symmetries = grains_list[0].phase.symmetry.symmetry_operators()

    # prepare list of input pair indices
    # e.g [(1, 2), (1, 3), (2, 3)]
    index_array = np.arange(pos_array.shape[0])

    idx = np.stack(np.triu_indices(len(index_array), k=1), axis=-1)

    pair_indices = index_array[idx]

    pairs_match_bool_array = are_grains_duplicate_stitching_array_numba(pos_array, U_array,
                                                                        pair_indices, symmetries,
                                                                        dist_tol_xy, dist_tol_z,
                                                                        angle_tol)

    matching_indices_masked = pair_indices[pairs_match_bool_array, :]
    return matching_indices_masked


@njit(fastmath=True, parallel=True)
def are_grains_duplicate_stitching_array_numba(pos_array, U_array,
                                               pair_indices, symmetries,
                                               dist_tol_xy=0.10,
                                               dist_tol_z=0.20,
                                               angle_tol=1.0):
    """Array-ified numba-ified version of are_grains_duplicate_stitching.
    From an array of positions and orientations, returns 1D M-length array of True/False for each pair index in pair_indices.
    :param pos_array: Nx3 Numpy array of grain positions
    :param U_array: Nx(3x3) Nump array of grain orientations
    :param pair_indices: Mx2 Numpy array of indices for pos_array and U_array
    :param dist_tol_xy: XY Distance tolerance (mm)
    :param dist_tol_z: Z Distance tolerance (mm)
    :param angle_tol: float misorientation tolerance in degrees
    :param symmetries: All the symmetry-equivalent rotations (symmetry operators) from pymicro Symmetry class

    :return: 1D M-length Numpy boolean array, True/False for each pair in pair_indices
    """
    n_pairs = pair_indices.shape[0]

    # check positions first
    pairs_match_bool_array = np.zeros(n_pairs, dtype=np.bool_)

    # iterate over all input pairs
    for pair_index in prange(n_pairs):
        gi, gj = pair_indices[pair_index]
        # get grain positions
        pos_i = pos_array[gi]
        pos_j = pos_array[gj]

        # check Z distance first
        distance_z = np.abs(pos_i[2] - pos_j[2])
        if distance_z < dist_tol_z:
            # distance_xy is more expensive, so check only if distance_z matches
            distance_xy = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
            if distance_xy < dist_tol_xy:
                # distance matches, so check misorientations

                U_a = U_array[gi]
                U_b = U_array[gj]

                pairs_match_bool_array[pair_index] = disorientation_single_check_numba((U_a, U_b), angle_tol,
                                                                                       symmetries)

    return pairs_match_bool_array


# TODO: test this!
def are_grains_neighbours(grain_a, grain_b, dist_const=1.5):
    """Reference function for grain neighbourhood detection. Uses radius-based distance tolerance.
    Returns True if grain centre-of-mass separation less than dist_const * (radius_a + radius_b).

    :param grain_a: First grain
    :param grain_b: Second grain
    :param dist_const: Multiplier for (radius_a + radius_b)

    :return: True if distance < dist_const * (radius_a + radius_b), False otherwise.
    """
    # check their separation less than distance tolerance
    pos_a = grain_a.pos_sample
    pos_b = grain_b.pos_sample
    rad_a = grain_a.radius
    rad_b = grain_b.radius
    grain_com_separation = spatial.distance.euclidean(pos_a, pos_b)
    dist_tol = dist_const * (rad_a + rad_b)
    if grain_com_separation > dist_tol:
        return False
    return True


def are_grains_neighbours_array_numba_wrapper(grains_list, dist_const=1.5):
    """Wrapper for are_grains_neighbours_array_numba.
    Lets us prepare a list of pairs of indices to grains_list to check, using nice itertools functions.

    :param grains_list: List of grains to check.
    :param dist_const: Multiplier for (radius_a + radius_b)

    :return: List of indices of neighbourhly grain pairs.
    """
    pos_array = np.array([grain.pos_sample for grain in grains_list])
    radius_array = np.array([grain.radius for grain in grains_list])

    # prepare list of input pair indices
    # e.g [(1, 2), (1, 3), (2, 3)]
    index_array = np.arange(pos_array.shape[0])

    idx = np.stack(np.triu_indices(len(index_array), k=1), axis=-1)

    pair_indices = index_array[idx]

    pairs_match_bool_array = are_grains_neighbours_array_numba(pos_array, radius_array,
                                                               pair_indices,
                                                               dist_const)

    matching_indices_masked = pair_indices[pairs_match_bool_array, :]
    return matching_indices_masked


@njit(fastmath=True, parallel=True)
def are_grains_neighbours_array_numba(pos_array, radius_array, pair_indices, dist_const=1.5):
    """Array-ified numba-ified version of are_grains_neighbours.
    From an array of positions and orientations, returns 1D M-length array of True/False for each pair index in pair_indices.
    :param pos_array: Nx3 Numpy array of grain positions
    :param radius_array: Nx3 Numpy array of grain positions
    :param pair_indices: Mx2 Numpy array of indices for pos_array and radius_array
    :param dist_const: Multiplier for (radius_a + radius_b)

    :return: 1D M-length Numpy boolean array, True/False for each pair in pair_indices
    """
    n_pairs = pair_indices.shape[0]

    # check positions only
    pairs_match_bool_array = np.zeros(n_pairs, dtype=np.bool_)

    # iterate over all input pairs
    for pair_index in prange(n_pairs):
        gi, gj = pair_indices[pair_index]
        # get grain positions
        pos_i = pos_array[gi]
        pos_j = pos_array[gj]
        rad_i = radius_array[gi]
        rad_j = radius_array[gj]
        grain_com_separation = np.sqrt(
            (pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        dist_tol = dist_const * (rad_i + rad_j)

        pairs_match_bool_array[pair_index] = grain_com_separation < dist_tol

    return pairs_match_bool_array


# TODO: test this
def are_grains_embedded(parent_grain, child_grain, dist_const=1.5):
    """
    Reference function for embedding. Checks if child_grain is inside parent_grain
    """
    if parent_grain is child_grain:
        raise ValueError("Parent grain and child grain are the same!")
    pos_parent = parent_grain.pos_sample
    pos_child = child_grain.pos_sample
    rad_parent = parent_grain.radius
    grain_com_separation = spatial.distance.euclidean(pos_parent, pos_child)
    dist_tol = dist_const * rad_parent
    if grain_com_separation > dist_tol:
        return False
    return True


def are_grains_embedded_array_numba_wrapper(parent_grains, child_grains, dist_const=1.5):
    pos_array_parent = np.array([grain.pos_sample for grain in parent_grains])
    pos_array_child = np.array([grain.pos_sample for grain in child_grains])
    radius_array_parent = np.array([grain.radius for grain in parent_grains])

    # prepare list of input pair indices
    # e.g [(1, 2), (1, 3), (2, 3)]
    # prepare list of input pair indices from both input lists
    pair_indices = np.array(list(product(range(pos_array_parent.shape[0]), range(pos_array_child.shape[0]))))

    pairs_match_bool_array = are_grains_embedded_array_numba(pos_array_parent, pos_array_child, radius_array_parent,
                                                             pair_indices,
                                                             dist_const)

    matching_indices_masked = pair_indices[pairs_match_bool_array, :]
    return matching_indices_masked


@njit(fastmath=True, parallel=True)
def are_grains_embedded_array_numba(pos_array_parent, pos_array_child, radius_array_parent, pair_indices,
                                    dist_const=1.5):
    """Array-ified numba-ified version of are_grains_embedded.
    From two arrays of positions (parent and child) and an array of parent radii, returns 1D M-length array of True/False for each pair index in pair_indices.
    :param pos_array_parent: Nx3 Numpy array of parent grain positions
    :param pos_array_child: Nx3 Numpy array of child grain positions
    :param radius_array_parent: Nx3 Numpy array of parent grain radii
    :param pair_indices: Mx2 Numpy array of indices for pos_array_parent/radius_array_parent and pos_array_child
    :param dist_const: Multiplier for radis_parent
    :return: 1D M-length Numpy boolean array, True/False for each pair in pair_indices
    """
    n_pairs = pair_indices.shape[0]

    # check positions only
    pairs_match_bool_array = np.zeros(n_pairs, dtype=np.bool_)

    # iterate over all input pairs
    for pair_index in prange(n_pairs):
        gi, gj = pair_indices[pair_index]
        # get grain positions
        pos_parent = pos_array_parent[gi]
        pos_child = pos_array_child[gj]
        rad_parent = radius_array_parent[gi]

        grain_com_separation = np.sqrt(
            (pos_parent[0] - pos_child[0]) ** 2 + (pos_parent[1] - pos_child[1]) ** 2 + (
                    pos_parent[2] - pos_child[2]) ** 2)
        dist_tol = dist_const * rad_parent

        pairs_match_bool_array[pair_index] = grain_com_separation < dist_tol

    return pairs_match_bool_array


@njit(parallel=True)
def get_Pitsch_OR_angle_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.float)

    for sample_index in prange(n_grains):
        observed_parent_ori = gamma_grain_u_samples[sample_index]
        observed_child_ori = aprime_grain_u_samples[sample_index]

        smallest_misorientation_deg = np.inf

        for variant_mori in Pitsch_variant_matrices:
            calculated_child_ori = observed_parent_ori @ variant_mori

            misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

            if misd < smallest_misorientation_deg:
                smallest_misorientation_deg = misd

        results_array[sample_index] = smallest_misorientation_deg

        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit(parallel=True)
def get_G_T_OR_angle_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.float64)

    for sample_index in prange(n_grains):
        observed_parent_ori = gamma_grain_u_samples[sample_index]
        observed_child_ori = aprime_grain_u_samples[sample_index]

        smallest_misorientation_deg = np.inf

        for variant_mori in G_T_variant_matrices:
            calculated_child_ori = observed_parent_ori @ variant_mori

            misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

            if misd < smallest_misorientation_deg:
                smallest_misorientation_deg = misd

        results_array[sample_index] = smallest_misorientation_deg

        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit(parallel=True)
def get_N_W_OR_angle_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.float64)

    for sample_index in prange(n_grains):
        observed_parent_ori = gamma_grain_u_samples[sample_index]
        observed_child_ori = aprime_grain_u_samples[sample_index]

        smallest_misorientation_deg = np.inf

        for variant_mori in N_W_variant_matrices:
            calculated_child_ori = observed_parent_ori @ variant_mori

            misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

            if misd < smallest_misorientation_deg:
                smallest_misorientation_deg = misd

        results_array[sample_index] = smallest_misorientation_deg

        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit(parallel=True)
def get_K_S_OR_angle_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.float64)

    for sample_index in prange(n_grains):
        observed_parent_ori = gamma_grain_u_samples[sample_index]
        observed_child_ori = aprime_grain_u_samples[sample_index]

        smallest_misorientation_deg = np.inf

        for variant_mori in K_S_variant_matrices:
            calculated_child_ori = observed_parent_ori @ variant_mori

            misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

            if misd < smallest_misorientation_deg:
                smallest_misorientation_deg = misd

        results_array[sample_index] = smallest_misorientation_deg

        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit(parallel=True)
def get_S_N_OR_angle_numba_parallel(cubic_grain_u_samples, hex_grain_u_samples, progress_proxy=None):
    n_grains = cubic_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.float64)

    for sample_index in prange(n_grains):
        observed_parent_ori = cubic_grain_u_samples[sample_index]
        observed_child_ori = hex_grain_u_samples[sample_index]

        smallest_misorientation_deg = np.inf

        for variant_mori in S_N_variant_matrices:
            calculated_child_ori = observed_parent_ori @ variant_mori

            misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), hex_symm_ops)

            if misd < smallest_misorientation_deg:
                smallest_misorientation_deg = misd

        results_array[sample_index] = smallest_misorientation_deg

        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit(parallel=True)
def check_Pitsch_OR_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, misd_tol, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.bool_)

    for sample_index in prange(n_grains):
        found_match_for_this_grain = False
        gamma_grain_U_sample = gamma_grain_u_samples[sample_index]
        aprime_grain_U_sample = aprime_grain_u_samples[sample_index]

        observed_misorientation = gamma_grain_U_sample.T @ aprime_grain_U_sample

        for desired_misorientation in Pitsch_variant_matrices:
            if not found_match_for_this_grain:
                found_match_for_this_grain = disorientation_single_check_numba(
                    (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops)

        results_array[sample_index] = found_match_for_this_grain
        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit
def check_Pitsch_OR_numba(gamma_grain_u_sample, aprime_grain_u_sample, misd_tol):
    observed_misorientation = gamma_grain_u_sample.T @ aprime_grain_u_sample

    for desired_misorientation in Pitsch_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


def check_Pitsch_OR(gamma_grain, aprime_grain, misd_tol):
    observed_misorientation = gamma_grain.U_sample.T @ aprime_grain.U_sample
    for desired_misorientation in Pitsch_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False

# replacing with matrix-based versions
# @njit(parallel=True)
# def check_Pitsch_OR_numba_parallel(gamma_grain_UB_samples, aprime_grain_UB_samples, misd_tol, progress_proxy=None):
#     # Koumatos, K and Muehlemann, A (2017) A theoretical investigation of orientation relationships
#     # and transformation strains in steels. Acta Crystallographica Section A: Foundations and
#     # Advances, A73. pp. 115-123. ISSN 2053-2733
#     directions_1_1bar_0_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_1bar_1bar_2bar_aprime_hkl = np.array([
#         [1., 1, 2],
#         [1., 1, -2],
#         [1., -1, 2],
#         [1., -1, -2],
#         [1., 2, 1],
#         [1., 2, -1],
#         [1., -2, 1],
#         [1., -2, -1],
#         [-1., 1, 2],
#         [-1., 1, -2],
#         [-1., -1, 2],
#         [-1., -1, -2],
#         [-1., 2, 1],
#         [-1., 2, -1],
#         [-1., -2, 1],
#         [-1., -2, -1],
#         [2., 1, 1],
#         [2., 1, -1],
#         [2., -1, 1],
#         [2., -1, -1],
#         [-2., 1, 1],
#         [-2., 1, -1],
#         [-2., -1, -1],
#         [-2., -1, 1],
#     ])
#
#     directions_0_0_1_gamma_hkl = np.array([
#         [1., 0, 0],
#         [-1., 0, 0],
#         [0., 1, 0],
#         [0., -1, 0],
#         [0., 0, 1],
#         [0., 0, -1],
#     ])
#
#     plane_normals_110_gamma_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     n_grains = gamma_grain_UB_samples.shape[0]
#     results_array = np.zeros(n_grains, dtype=np.bool_)
#     for sample_index in prange(n_grains):
#         gamma_grain_UB_sample = gamma_grain_UB_samples[sample_index]
#         aprime_grain_UB_sample = aprime_grain_UB_samples[sample_index]
#
#         plane_normals_1bar_1bar_2bar_aprime_sample = (
#                 aprime_grain_UB_sample @ plane_normals_1bar_1bar_2bar_aprime_hkl.T).T
#         plane_normals_110_gamma_sample = (gamma_grain_UB_sample @ plane_normals_110_gamma_hkl.T).T
#
#         plane_normals_parallel = False
#         for plane_normal_aprime_sample in plane_normals_1bar_1bar_2bar_aprime_sample:
#             if not plane_normals_parallel:
#                 for plane_normal_gam_sample in plane_normals_110_gamma_sample:
#                     if not plane_normals_parallel:
#                         angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                               plane_normal_gam_sample)
#                         angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                         if angle_between_normals_deg < misd_tol:
#                             plane_normals_parallel = True
#
#         # return early if no plane normals are parallel
#         if not plane_normals_parallel:
#             results_array[sample_index] = False
#         else:
#             directions_1_1bar_0_aprime_sample = (aprime_grain_UB_sample @ directions_1_1bar_0_aprime_hkl.T).T
#             directions_0_0_1_gamma_sample = (gamma_grain_UB_sample @ directions_0_0_1_gamma_hkl.T).T
#
#             # otherwise keep going:
#             direcs_parallel = False
#             for direc_aprime_sample in directions_1_1bar_0_aprime_sample:
#                 if not direcs_parallel:
#                     for direc_gam_sample in directions_0_0_1_gamma_sample:
#                         if not direcs_parallel:
#                             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#                             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#                             if angle_between_direcs_deg < misd_tol:
#                                 direcs_parallel = True
#
#             if direcs_parallel:
#                 results_array[sample_index] = True
#             else:
#                 results_array[sample_index] = False
#
#         if progress_proxy is not None:
#             progress_proxy.update(1)
#
#     return results_array
#
#
# @njit
# def check_Pitsch_OR_numba(gamma_grain_UB_sample, aprime_grain_UB_sample, misd_tol):
#     # Koumatos, K and Muehlemann, A (2017) A theoretical investigation of orientation relationships
#     # and transformation strains in steels. Acta Crystallographica Section A: Foundations and
#     # Advances, A73. pp. 115-123. ISSN 2053-2733
#     directions_1_1bar_0_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_1bar_1bar_2bar_aprime_hkl = np.array([
#         [1., 1, 2],
#         [1., 1, -2],
#         [1., -1, 2],
#         [1., -1, -2],
#         [1., 2, 1],
#         [1., 2, -1],
#         [1., -2, 1],
#         [1., -2, -1],
#         [-1., 1, 2],
#         [-1., 1, -2],
#         [-1., -1, 2],
#         [-1., -1, -2],
#         [-1., 2, 1],
#         [-1., 2, -1],
#         [-1., -2, 1],
#         [-1., -2, -1],
#         [2., 1, 1],
#         [2., 1, -1],
#         [2., -1, 1],
#         [2., -1, -1],
#         [-2., 1, 1],
#         [-2., 1, -1],
#         [-2., -1, -1],
#         [-2., -1, 1],
#     ])
#
#     directions_0_0_1_gamma_hkl = np.array([
#         [1., 0, 0],
#         [-1., 0, 0],
#         [0., 1, 0],
#         [0., -1, 0],
#         [0., 0, 1],
#         [0., 0, -1],
#     ])
#
#     plane_normals_110_gamma_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_1bar_1bar_2bar_aprime_sample = (aprime_grain_UB_sample @ plane_normals_1bar_1bar_2bar_aprime_hkl.T).T
#     plane_normals_110_gamma_sample = (gamma_grain_UB_sample @ plane_normals_110_gamma_hkl.T).T
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_sample in plane_normals_1bar_1bar_2bar_aprime_sample:
#         if not plane_normals_parallel:
#             for plane_normal_gam_sample in plane_normals_110_gamma_sample:
#                 if not plane_normals_parallel:
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     directions_1_1bar_0_aprime_sample = (aprime_grain_UB_sample @ directions_1_1bar_0_aprime_hkl.T).T
#     directions_0_0_1_gamma_sample = (gamma_grain_UB_sample @ directions_0_0_1_gamma_hkl.T).T
#
#     for direc_aprime_sample in directions_1_1bar_0_aprime_sample:
#         for direc_gam_sample in directions_0_0_1_gamma_sample:
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False
#
#
# def check_Pitsch_OR(gamma_grain, aprime_grain, misd_tol):
#     # Koumatos, K and Muehlemann, A (2017) A theoretical investigation of orientation relationships
#     # and transformation strains in steels. Acta Crystallographica Section A: Foundations and
#     # Advances, A73. pp. 115-123. ISSN 2053-2733
#     directions_1_1bar_0_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_1bar_1bar_2bar_aprime_hkl = np.array([
#         [1., 1, 2],
#         [1., 1, -2],
#         [1., -1, 2],
#         [1., -1, -2],
#         [1., 2, 1],
#         [1., 2, -1],
#         [1., -2, 1],
#         [1., -2, -1],
#         [-1., 1, 2],
#         [-1., 1, -2],
#         [-1., -1, 2],
#         [-1., -1, -2],
#         [-1., 2, 1],
#         [-1., 2, -1],
#         [-1., -2, 1],
#         [-1., -2, -1],
#         [2., 1, 1],
#         [2., 1, -1],
#         [2., -1, 1],
#         [2., -1, -1],
#         [-2., 1, 1],
#         [-2., 1, -1],
#         [-2., -1, -1],
#         [-2., -1, 1],
#     ])
#
#     directions_0_0_1_gamma_hkl = np.array([
#         [1., 0, 0],
#         [-1., 0, 0],
#         [0., 1, 0],
#         [0., -1, 0],
#         [0., 0, 1],
#         [0., 0, -1],
#     ])
#
#     plane_normals_110_gamma_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_hkl in plane_normals_1bar_1bar_2bar_aprime_hkl:
#         if not plane_normals_parallel:
#             for plane_normal_gam_hkl in plane_normals_110_gamma_hkl:
#                 if not plane_normals_parallel:
#                     plane_normal_apr_sample = aprime_grain.hkl_vec_as_sample_vec(plane_normal_aprime_hkl)
#                     plane_normal_gam_sample = gamma_grain.hkl_vec_as_sample_vec(plane_normal_gam_hkl)
#
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_apr_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     for direc_aprime_hkl in directions_1_1bar_0_aprime_hkl:
#         for direc_gam_hkl in directions_0_0_1_gamma_hkl:
#             direc_apr_sample = aprime_grain.hkl_vec_as_sample_vec(direc_aprime_hkl)
#             direc_gam_sample = gamma_grain.hkl_vec_as_sample_vec(direc_gam_hkl)
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_apr_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False


@njit(parallel=True)
def check_G_T_OR_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, misd_tol, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.bool_)

    for sample_index in prange(n_grains):
        found_match_for_this_grain = False
        gamma_grain_U_sample = gamma_grain_u_samples[sample_index]
        aprime_grain_U_sample = aprime_grain_u_samples[sample_index]

        observed_misorientation = gamma_grain_U_sample.T @ aprime_grain_U_sample

        for desired_misorientation in G_T_variant_matrices:
            if not found_match_for_this_grain:
                found_match_for_this_grain = disorientation_single_check_numba(
                    (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops)

        results_array[sample_index] = found_match_for_this_grain
        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit
def check_G_T_OR_numba(gamma_grain_u_sample, aprime_grain_u_sample, misd_tol):
    observed_misorientation = gamma_grain_u_sample.T @ aprime_grain_u_sample

    for desired_misorientation in G_T_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


def check_G_T_OR(gamma_grain, aprime_grain, misd_tol):
    observed_misorientation = gamma_grain.U_sample.T @ aprime_grain.U_sample
    for desired_misorientation in G_T_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


# def check_G_T_OR_vectors(gamma_grain, aprime_grain, misd_tol):
#     # He, Y., Godet, S., & Jonas, J. J. (2006).
#     # Observations of the Gibeon meteorite and the inverse Greninger–Troiano orientation relationship.
#     # Journal of Applied Crystallography, 39(1), 72–81. https://doi.org/10.1107/S0021889805038276
#     directions_7_17_17_aprime_hkl = np.array([
#         [7., 17, 17],
#         [7., 17, -17],
#         [7., -17, 17],
#         [7., -17, -17],
#         [-7., 17, 17],
#         [-7., 17, -17],
#         [-7., -17, 17],
#         [-7., -17, -17],
#         [-17., 7, 17],
#         [-17., 7, -17],
#         [-17., -7, 17],
#         [-17., -7, -17],
#         [-17., 17, 7],
#         [-17., 17, -7],
#         [-17., -17, 7],
#         [-17., -17, -7],
#         [17., 7, 17],
#         [17., 7, -17],
#         [17., -7, 17],
#         [17., -7, -17],
#         [17., 17, 7],
#         [17., 17, -7],
#         [17., -17, 7],
#         [17., -17, -7],
#
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [-1, 1, 0],
#         [-1, 0, 1],
#         [0, -1, 1],
#         [1, -1, 0],
#         [1, 0, -1],
#         [0, 1, -1],
#         [-1, -1, 0],
#         [-1, 0, -1],
#         [0, -1, -1]
#     ])
#
#     directions_5_12_17_gamma_hkl = np.array([
#         [5., 12, 17],
#         [5., 12, -17],
#         [5., -12, 17],
#         [5., -12, -17],
#         [5., 17, 12],
#         [5., 17, -12],
#         [5., -17, 12],
#         [5., -17, -12],
#         [-5., 12, 17],
#         [-5., 12, -17],
#         [-5., -12, 17],
#         [-5., -12, -17],
#         [-5., 17, 12],
#         [-5., 17, -12],
#         [-5., -17, 12],
#         [-5., -17, -12],
#         [12., 5, 17],
#         [12., 5, -17],
#         [12., -5, 17],
#         [12., -5, -17],
#         [12., 17, 5],
#         [12., 17, -5],
#         [12., -17, 5],
#         [12., -17, -5],
#         [-12., 5, 17],
#         [-12., 5, -17],
#         [-12., -5, 17],
#         [-12., -5, -17],
#         [-12., 17, 5],
#         [-12., 17, -5],
#         [-12., -17, 5],
#         [-12., -17, -5],
#         [17., 5, 12],
#         [17., 5, -12],
#         [17., -5, 12],
#         [17., -5, -12],
#         [17., 12, 5],
#         [17., 12, -5],
#         [17., -12, 5],
#         [17., -12, -5],
#         [-17., 5, 12],
#         [-17., 5, -12],
#         [-17., -5, 12],
#         [-17., -5, -12],
#         [-17., 12, 5],
#         [-17., 12, -5],
#         [-17., -12, 5],
#         [-17., -12, -5],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1, 1, 1],
#         [-1, 1, 1],
#         [1, -1, 1],
#         [1, 1, -1],
#         [-1, -1, 1],
#         [-1, 1, -1],
#         [1, -1, -1],
#         [-1, -1, -1]
#     ])
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_hkl in plane_normals_0_1_1_aprime_hkl:
#         if not plane_normals_parallel:
#             for plane_normal_gam_hkl in plane_normals_111_gamma_hkl:
#                 if not plane_normals_parallel:
#                     plane_normal_apr_sample = aprime_grain.hkl_vec_as_sample_vec(plane_normal_aprime_hkl)
#                     plane_normal_gam_sample = gamma_grain.hkl_vec_as_sample_vec(plane_normal_gam_hkl)
#
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_apr_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     for direc_aprime_hkl in directions_7_17_17_aprime_hkl:
#         for direc_gam_hkl in directions_5_12_17_gamma_hkl:
#             direc_apr_sample = aprime_grain.hkl_vec_as_sample_vec(direc_aprime_hkl)
#             direc_gam_sample = gamma_grain.hkl_vec_as_sample_vec(direc_gam_hkl)
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_apr_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False


@njit(parallel=True)
def check_N_W_OR_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, misd_tol, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.bool_)

    for sample_index in prange(n_grains):
        found_match_for_this_grain = False
        gamma_grain_U_sample = gamma_grain_u_samples[sample_index]
        aprime_grain_U_sample = aprime_grain_u_samples[sample_index]

        observed_misorientation = gamma_grain_U_sample.T @ aprime_grain_U_sample

        for desired_misorientation in N_W_variant_matrices:
            if not found_match_for_this_grain:
                found_match_for_this_grain = disorientation_single_check_numba(
                    (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops)

        results_array[sample_index] = found_match_for_this_grain
        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit
def check_N_W_OR_numba(gamma_grain_u_sample, aprime_grain_u_sample, misd_tol):
    observed_misorientation = gamma_grain_u_sample.T @ aprime_grain_u_sample

    for desired_misorientation in N_W_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


def check_N_W_OR(gamma_grain, aprime_grain, misd_tol):
    observed_misorientation = gamma_grain.U_sample.T @ aprime_grain.U_sample
    for desired_misorientation in N_W_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False
# replacing below with matrix-based
# @njit(parallel=True)
# def check_N_W_OR_numba_parallel(gamma_grain_UB_samples, aprime_grain_UB_samples, misd_tol, progress_proxy=None):
#     directions_0_1bar_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     directions_1bar_1bar_2_gamma_hkl = np.array([
#         [1., 1, 2],
#         [1., 1, -2],
#         [1., -1, 2],
#         [1., -1, -2],
#         [1., 2, 1],
#         [1., 2, -1],
#         [1., -2, 1],
#         [1., -2, -1],
#         [-1., 1, 2],
#         [-1., 1, -2],
#         [-1., -1, 2],
#         [-1., -1, -2],
#         [-1., 2, 1],
#         [-1., 2, -1],
#         [-1., -2, 1],
#         [-1., -2, -1],
#         [2., 1, 1],
#         [2., 1, -1],
#         [2., -1, 1],
#         [2., -1, -1],
#         [-2., 1, 1],
#         [-2., 1, -1],
#         [-2., -1, 1],
#         [-2., -1, -1],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     n_grains = gamma_grain_UB_samples.shape[0]
#     results_array = np.zeros(n_grains, dtype=np.bool_)
#     for sample_index in prange(n_grains):
#         gamma_grain_UB_sample = gamma_grain_UB_samples[sample_index]
#         aprime_grain_UB_sample = aprime_grain_UB_samples[sample_index]
#
#         plane_normals_0_1_1_aprime_sample = (aprime_grain_UB_sample @ plane_normals_0_1_1_aprime_hkl.T).T
#         plane_normals_111_gamma_sample = (gamma_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#         plane_normals_parallel = False
#         for plane_normal_aprime_sample in plane_normals_0_1_1_aprime_sample:
#             if not plane_normals_parallel:
#                 for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                     if not plane_normals_parallel:
#                         angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                               plane_normal_gam_sample)
#                         angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                         if angle_between_normals_deg < misd_tol:
#                             plane_normals_parallel = True
#
#         # return early if no plane normals are parallel
#         if not plane_normals_parallel:
#             results_array[sample_index] = False
#         else:
#             directions_0_1bar_1_aprime_sample = (aprime_grain_UB_sample @ directions_0_1bar_1_aprime_hkl.T).T
#             directions_1bar_1bar_2_gamma_sample = (gamma_grain_UB_sample @ directions_1bar_1bar_2_gamma_hkl.T).T
#
#             # otherwise keep going:
#             direcs_parallel = False
#             for direc_aprime_sample in directions_0_1bar_1_aprime_sample:
#                 if not direcs_parallel:
#                     for direc_gam_sample in directions_1bar_1bar_2_gamma_sample:
#                         if not direcs_parallel:
#                             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#                             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#                             if angle_between_direcs_deg < misd_tol:
#                                 direcs_parallel = True
#
#             if direcs_parallel:
#                 results_array[sample_index] = True
#             else:
#                 results_array[sample_index] = False
#
#         if progress_proxy is not None:
#             progress_proxy.update(1)
#
#     return results_array
#
#
# @njit
# def check_N_W_OR_numba(gamma_grain_UB_sample, aprime_grain_UB_sample, misd_tol):
#     directions_0_1bar_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     directions_1bar_1bar_2_gamma_hkl = np.array([
#         [1., 1, 2],
#         [1., 1, -2],
#         [1., -1, 2],
#         [1., -1, -2],
#         [1., 2, 1],
#         [1., 2, -1],
#         [1., -2, 1],
#         [1., -2, -1],
#         [-1., 1, 2],
#         [-1., 1, -2],
#         [-1., -1, 2],
#         [-1., -1, -2],
#         [-1., 2, 1],
#         [-1., 2, -1],
#         [-1., -2, 1],
#         [-1., -2, -1],
#         [2., 1, 1],
#         [2., 1, -1],
#         [2., -1, 1],
#         [2., -1, -1],
#         [-2., 1, 1],
#         [-2., 1, -1],
#         [-2., -1, 1],
#         [-2., -1, -1],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_sample = (aprime_grain_UB_sample @ plane_normals_0_1_1_aprime_hkl.T).T
#     plane_normals_111_gamma_sample = (gamma_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_sample in plane_normals_0_1_1_aprime_sample:
#         if not plane_normals_parallel:
#             for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                 if not plane_normals_parallel:
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     directions_0_1bar_1_aprime_sample = (aprime_grain_UB_sample @ directions_0_1bar_1_aprime_hkl.T).T
#     directions_1bar_1bar_2_gamma_sample = (gamma_grain_UB_sample @ directions_1bar_1bar_2_gamma_hkl.T).T
#
#     for direc_aprime_sample in directions_0_1bar_1_aprime_sample:
#         for direc_gam_sample in directions_1bar_1bar_2_gamma_sample:
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False
#
#
# def check_N_W_OR(gamma_grain, aprime_grain, misd_tol):
#     directions_0_1bar_1_aprime_hkl = np.array([
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [-1, 1, 0],
#         [-1, 0, 1],
#         [0, -1, 1],
#         [1, -1, 0],
#         [1, 0, -1],
#         [0, 1, -1],
#         [-1, -1, 0],
#         [-1, 0, -1],
#         [0, -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [-1, 1, 0],
#         [-1, 0, 1],
#         [0, -1, 1],
#         [1, -1, 0],
#         [1, 0, -1],
#         [0, 1, -1],
#         [-1, -1, 0],
#         [-1, 0, -1],
#         [0, -1, -1]
#     ])
#
#     directions_1bar_1bar_2_gamma_hkl = np.array([
#         [1, 1, 2],
#         [1, 1, -2],
#         [1, -1, 2],
#         [1, -1, -2],
#         [1, 2, 1],
#         [1, 2, -1],
#         [1, -2, 1],
#         [1, -2, -1],
#         [-1, 1, 2],
#         [-1, 1, -2],
#         [-1, -1, 2],
#         [-1, -1, -2],
#         [-1, 2, 1],
#         [-1, 2, -1],
#         [-1, -2, 1],
#         [-1, -2, -1],
#         [2, 1, 1],
#         [2, 1, -1],
#         [2, -1, 1],
#         [2, -1, -1],
#         [-2, 1, 1],
#         [-2, 1, -1],
#         [-2, -1, 1],
#         [-2, -1, -1],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1, 1, 1],
#         [-1, 1, 1],
#         [1, -1, 1],
#         [1, 1, -1],
#         [-1, -1, 1],
#         [-1, 1, -1],
#         [1, -1, -1],
#         [-1, -1, -1]
#     ])
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_hkl in plane_normals_0_1_1_aprime_hkl:
#         if not plane_normals_parallel:
#             for plane_normal_gam_hkl in plane_normals_111_gamma_hkl:
#                 if not plane_normals_parallel:
#                     plane_normal_apr_sample = aprime_grain.hkl_vec_as_sample_vec(plane_normal_aprime_hkl)
#                     plane_normal_gam_sample = gamma_grain.hkl_vec_as_sample_vec(plane_normal_gam_hkl)
#
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_apr_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     for direc_aprime_hkl in directions_0_1bar_1_aprime_hkl:
#         for direc_gam_hkl in directions_1bar_1bar_2_gamma_hkl:
#             direc_apr_sample = aprime_grain.hkl_vec_as_sample_vec(direc_aprime_hkl)
#             direc_gam_sample = gamma_grain.hkl_vec_as_sample_vec(direc_gam_hkl)
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_apr_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False


@njit(parallel=True)
def check_K_S_OR_numba_parallel(gamma_grain_u_samples, aprime_grain_u_samples, misd_tol, progress_proxy=None):
    n_grains = gamma_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.bool_)

    for sample_index in prange(n_grains):
        found_match_for_this_grain = False
        gamma_grain_U_sample = gamma_grain_u_samples[sample_index]
        aprime_grain_U_sample = aprime_grain_u_samples[sample_index]

        observed_misorientation = gamma_grain_U_sample.T @ aprime_grain_U_sample

        for desired_misorientation in K_S_variant_matrices:
            if not found_match_for_this_grain:
                found_match_for_this_grain = disorientation_single_check_numba(
                    (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops)

        results_array[sample_index] = found_match_for_this_grain
        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit
def check_K_S_OR_numba(gamma_grain_u_sample, aprime_grain_u_sample, misd_tol):
    observed_misorientation = gamma_grain_u_sample.T @ aprime_grain_u_sample

    for desired_misorientation in K_S_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


def check_K_S_OR(gamma_grain, aprime_grain, misd_tol):
    observed_misorientation = gamma_grain.U_sample.T @ aprime_grain.U_sample
    for desired_misorientation in K_S_variant_matrices:
        if disorientation_single_check_numba(
                (observed_misorientation, desired_misorientation), misd_tol, cubic_symm_ops):
            return True

    return False


# replacing with matrix versions:
# @njit(parallel=True)
# def check_K_S_OR_numba_parallel(gamma_grain_UB_samples, aprime_grain_UB_samples, misd_tol, progress_proxy=None):
#     directions_1bar_1bar_1_aprime_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [.0, -1, -1]
#     ])
#
#     directions_1bar_0_1_gamma_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     n_grains = gamma_grain_UB_samples.shape[0]
#     results_array = np.zeros(n_grains, dtype=np.bool_)
#     for sample_index in prange(n_grains):
#         gamma_grain_UB_sample = gamma_grain_UB_samples[sample_index]
#         aprime_grain_UB_sample = aprime_grain_UB_samples[sample_index]
#
#         plane_normals_0_1_1_aprime_sample = (aprime_grain_UB_sample @ plane_normals_0_1_1_aprime_hkl.T).T
#         plane_normals_111_gamma_sample = (gamma_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#         plane_normals_parallel = False
#         for plane_normal_aprime_sample in plane_normals_0_1_1_aprime_sample:
#             if not plane_normals_parallel:
#                 for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                     if not plane_normals_parallel:
#                         angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                               plane_normal_gam_sample)
#                         angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                         if angle_between_normals_deg < misd_tol:
#                             plane_normals_parallel = True
#
#         # return early if no plane normals are parallel
#         if not plane_normals_parallel:
#             results_array[sample_index] = False
#         else:
#             directions_1bar_1bar_1_aprime_sample = (aprime_grain_UB_sample @ directions_1bar_1bar_1_aprime_hkl.T).T
#             directions_1bar_1bar_2_gamma_sample = (gamma_grain_UB_sample @ directions_1bar_0_1_gamma_hkl.T).T
#
#             # otherwise keep going:
#             direcs_parallel = False
#             for direc_aprime_sample in directions_1bar_1bar_1_aprime_sample:
#                 if not direcs_parallel:
#                     for direc_gam_sample in directions_1bar_1bar_2_gamma_sample:
#                         if not direcs_parallel:
#                             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#                             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#                             if angle_between_direcs_deg < misd_tol:
#                                 direcs_parallel = True
#
#             if direcs_parallel:
#                 results_array[sample_index] = True
#             else:
#                 results_array[sample_index] = False
#
#         if progress_proxy is not None:
#             progress_proxy.update(1)
#
#     return results_array
#
#
# @njit
# def check_K_S_OR_numba(gamma_grain_UB_sample, aprime_grain_UB_sample, misd_tol):
#     directions_1bar_1bar_1_aprime_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [.0, -1, -1]
#     ])
#
#     directions_1bar_0_1_gamma_hkl = np.array([
#         [1., 1, 0],
#         [1., 0, 1],
#         [0., 1, 1],
#         [-1., 1, 0],
#         [-1., 0, 1],
#         [0., -1, 1],
#         [1., -1, 0],
#         [1., 0, -1],
#         [0., 1, -1],
#         [-1., -1, 0],
#         [-1., 0, -1],
#         [0., -1, -1]
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_sample = (aprime_grain_UB_sample @ plane_normals_0_1_1_aprime_hkl.T).T
#     plane_normals_111_gamma_sample = (gamma_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_sample in plane_normals_0_1_1_aprime_sample:
#         if not plane_normals_parallel:
#             for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                 if not plane_normals_parallel:
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_aprime_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     directions_1bar_1bar_1_aprime_sample = (aprime_grain_UB_sample @ directions_1bar_1bar_1_aprime_hkl.T).T
#     directions_1bar_0_1_gamma_sample = (gamma_grain_UB_sample @ directions_1bar_0_1_gamma_hkl.T).T
#
#     for direc_aprime_sample in directions_1bar_1bar_1_aprime_sample:
#         for direc_gam_sample in directions_1bar_0_1_gamma_sample:
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_aprime_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False
#
#
# def check_K_S_OR(gamma_grain, aprime_grain, misd_tol):
#     directions_1bar_1bar_1_aprime_hkl = np.array([
#         [1, 1, 1],
#         [-1, 1, 1],
#         [1, -1, 1],
#         [1, 1, -1],
#         [-1, -1, 1],
#         [-1, 1, -1],
#         [1, -1, -1],
#         [-1, -1, -1]
#     ])
#
#     plane_normals_0_1_1_aprime_hkl = np.array([
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [-1, 1, 0],
#         [-1, 0, 1],
#         [0, -1, 1],
#         [1, -1, 0],
#         [1, 0, -1],
#         [0, 1, -1],
#         [-1, -1, 0],
#         [-1, 0, -1],
#         [0, -1, -1]
#     ])
#
#     directions_1bar_0_1_gamma_hkl = np.array([
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [-1, 1, 0],
#         [-1, 0, 1],
#         [0, -1, 1],
#         [1, -1, 0],
#         [1, 0, -1],
#         [0, 1, -1],
#         [-1, -1, 0],
#         [-1, 0, -1],
#         [0, -1, -1]
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1, 1, 1],
#         [-1, 1, 1],
#         [1, -1, 1],
#         [1, 1, -1],
#         [-1, -1, 1],
#         [-1, 1, -1],
#         [1, -1, -1],
#         [-1, -1, -1]
#     ])
#
#     plane_normals_parallel = False
#     for plane_normal_aprime_hkl in plane_normals_0_1_1_aprime_hkl:
#         if not plane_normals_parallel:
#             for plane_normal_gam_hkl in plane_normals_111_gamma_hkl:
#                 if not plane_normals_parallel:
#                     plane_normal_apr_sample = aprime_grain.hkl_vec_as_sample_vec(plane_normal_aprime_hkl)
#                     plane_normal_gam_sample = gamma_grain.hkl_vec_as_sample_vec(plane_normal_gam_hkl)
#
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_apr_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     for direc_aprime_hkl in directions_1bar_1bar_1_aprime_hkl:
#         for direc_gam_hkl in directions_1bar_0_1_gamma_hkl:
#             direc_apr_sample = aprime_grain.hkl_vec_as_sample_vec(direc_aprime_hkl)
#             direc_gam_sample = gamma_grain.hkl_vec_as_sample_vec(direc_gam_hkl)
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_apr_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False


@njit(parallel=True)
def check_S_N_OR_numba_parallel(cubic_grain_u_samples, hexagonal_grain_u_samples, misd_tol, progress_proxy=None):
    n_grains = cubic_grain_u_samples.shape[0]
    results_array = np.zeros(n_grains, dtype=np.bool_)

    for sample_index in prange(n_grains):
        found_match_for_this_grain = False
        cubic_grain_U_sample = cubic_grain_u_samples[sample_index]
        hexagonal_grain_U_sample = hexagonal_grain_u_samples[sample_index]

        # check symmetries before computing delta
        for desired_misorientation in S_N_variant_matrices:
            for cubic_symm_op in cubic_symm_ops:
                for hex_symm_op in hex_symm_ops:
                    observed_misorientation = (cubic_grain_U_sample @ cubic_symm_op.T).T @ (
                                hexagonal_grain_U_sample @ hex_symm_op.T)

                    misorientation_delta = desired_misorientation.T @ observed_misorientation
                    misd = misorientation_from_delta(misorientation_delta)
                    if misd < misd_tol:
                        found_match_for_this_grain = True

        results_array[sample_index] = found_match_for_this_grain
        if progress_proxy is not None:
            progress_proxy.update(1)

    return results_array


@njit
def check_S_N_OR_numba(cubic_grain_u_sample, hexagonal_grain_u_sample, misd_tol):
    for desired_misorientation in S_N_variant_matrices:
        for cubic_symm_op in cubic_symm_ops:
            for hex_symm_op in hex_symm_ops:
                observed_misorientation = (cubic_grain_u_sample @ cubic_symm_op.T).T @ (
                            hexagonal_grain_u_sample @ hex_symm_op.T)

                misorientation_delta = desired_misorientation.T @ observed_misorientation
                misd = misorientation_from_delta(misorientation_delta)
                if misd < misd_tol:
                    return True

    return False


def check_S_N_OR(cubic_grain, hexagonal_grain, misd_tol):
    for desired_misorientation in S_N_variant_matrices:
        for cubic_symm_op in cubic_symm_ops:
            for hex_symm_op in hex_symm_ops:
                observed_misorientation = (cubic_grain.U_sample @ cubic_symm_op.T).T @ (
                            hexagonal_grain.U_sample @ hex_symm_op.T)

                misorientation_delta = desired_misorientation.T @ observed_misorientation
                misd = misorientation_from_delta(misorientation_delta)
                if misd < misd_tol:
                    return True

    return False


# @njit(parallel=True)
# def check_S_N_OR_numba_parallel(cubic_grain_UB_samples, hexagonal_grain_UB_samples, misd_tol, progress_proxy=None):
#     plane_normals_0001_epsilon_hkl = np.array([
#         [0., 0, 1],
#         [0., 0, -1]
#     ])
#
#     directions_1bar_2_1bar_0_epsilon_hkl = np.array([
#         [-1., 2, 0],
#         [1., -2, 0],
#         [-2., 1, 0],
#         [2., -1, 0],
#         [-1., -1, 0],
#         [1., 1, 0],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1],
#     ])
#
#     directions_110_gamma_hkl = np.array([
#         [1., 1, 0],
#         [-1., 1, 0],
#         [1., -1, 0],
#         [-1., -1, 0],
#         [1., 0, 1],
#         [-1., 0, 1],
#         [1., 0, -1],
#         [-1., 0, -1],
#         [0., 1, 1],
#         [0., -1, 1],
#         [0., 1, -1],
#         [0., -1, -1],
#     ])
#
#     n_grains = cubic_grain_UB_samples.shape[0]
#     results_array = np.zeros(n_grains, dtype=np.bool_)
#     for sample_index in prange(n_grains):
#         cubic_grain_UB_sample = cubic_grain_UB_samples[sample_index]
#         hexagonal_grain_UB_sample = hexagonal_grain_UB_samples[sample_index]
#
#         plane_normals_0001_epsilon_sample = (hexagonal_grain_UB_sample @ plane_normals_0001_epsilon_hkl.T).T
#         plane_normals_111_gamma_sample = (cubic_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#         plane_normals_parallel = False
#         for plane_normal_epsilon_sample in plane_normals_0001_epsilon_sample:
#             if not plane_normals_parallel:
#                 for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                     if not plane_normals_parallel:
#                         angle_between_normals_rad = angle_between_vectors_rad(plane_normal_epsilon_sample,
#                                                                               plane_normal_gam_sample)
#                         angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                         if angle_between_normals_deg < misd_tol:
#                             plane_normals_parallel = True
#
#         # return early if no plane normals are parallel
#         if not plane_normals_parallel:
#             results_array[sample_index] = False
#         else:
#             directions_1bar_2_1bar_0_epsilon_sample = (
#                     hexagonal_grain_UB_sample @ directions_1bar_2_1bar_0_epsilon_hkl.T).T
#             directions_110_gamma_sample = (cubic_grain_UB_sample @ directions_110_gamma_hkl.T).T
#
#             # otherwise keep going:
#             direcs_parallel = False
#             for direc_eps_sample in directions_1bar_2_1bar_0_epsilon_sample:
#                 if not direcs_parallel:
#                     for direc_gam_sample in directions_110_gamma_sample:
#                         if not direcs_parallel:
#                             angle_between_direcs_rad = angle_between_vectors_rad(direc_eps_sample, direc_gam_sample)
#                             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#                             if angle_between_direcs_deg < misd_tol:
#                                 direcs_parallel = True
#
#             if direcs_parallel:
#                 results_array[sample_index] = True
#             else:
#                 results_array[sample_index] = False
#
#         if progress_proxy is not None:
#             progress_proxy.update(1)
#
#     return results_array
#
#
#
#
#
# @njit
# def check_S_N_OR_numba(cubic_grain_UB_sample, hexagonal_grain_UB_sample, misd_tol):
#     plane_normals_0001_epsilon_hkl = np.array([
#         [0., 0, 1],
#         [0., 0, -1]
#     ])
#
#     directions_1bar_2_1bar_0_epsilon_hkl = np.array([
#         [-1., 2, 0],
#         [1., -2, 0],
#         [-2., 1, 0],
#         [2., -1, 0],
#         [-1., -1, 0],
#         [1., 1, 0],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1., 1, 1],
#         [-1., 1, 1],
#         [1., -1, 1],
#         [1., 1, -1],
#         [-1., -1, 1],
#         [-1., 1, -1],
#         [1., -1, -1],
#         [-1., -1, -1],
#     ])
#
#     directions_110_gamma_hkl = np.array([
#         [1., 1, 0],
#         [-1., 1, 0],
#         [1., -1, 0],
#         [-1., -1, 0],
#         [1., 0, 1],
#         [-1., 0, 1],
#         [1., 0, -1],
#         [-1., 0, -1],
#         [0., 1, 1],
#         [0., -1, 1],
#         [0., 1, -1],
#         [0., -1, -1],
#     ])
#
#     plane_normals_0001_epsilon_sample = (hexagonal_grain_UB_sample @ plane_normals_0001_epsilon_hkl.T).T
#     plane_normals_111_gamma_sample = (cubic_grain_UB_sample @ plane_normals_111_gamma_hkl.T).T
#
#     plane_normals_parallel = False
#     for plane_normal_epsilon_sample in plane_normals_0001_epsilon_sample:
#         if not plane_normals_parallel:
#             for plane_normal_gam_sample in plane_normals_111_gamma_sample:
#                 if not plane_normals_parallel:
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_epsilon_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     # return early if no plane normals are parallel
#     if not plane_normals_parallel:
#         return False
#
#     # otherwise keep going:
#
#     directions_1bar_2_1bar_0_epsilon_sample = (hexagonal_grain_UB_sample @ directions_1bar_2_1bar_0_epsilon_hkl.T).T
#     directions_110_gamma_sample = (cubic_grain_UB_sample @ directions_110_gamma_hkl.T).T
#
#     for direc_eps_sample in directions_1bar_2_1bar_0_epsilon_sample:
#         for direc_gam_sample in directions_110_gamma_sample:
#
#             angle_between_direcs_rad = angle_between_vectors_rad(direc_eps_sample, direc_gam_sample)
#             angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#             if angle_between_direcs_deg < misd_tol:
#                 return True
#
#     return False
#
#
# def check_S_N_OR(cubic_grain, hexagonal_grain, misd_tol: float) -> bool:
#     plane_normals_0001_epsilon_hkl = np.array([
#         [0, 0, 1],
#         [0, 0, -1]
#     ])
#
#     directions_1bar_2_1bar_0_epsilon_hkl = np.array([
#         [-1, 2, 0],
#         [1, -2, 0],
#         [-2, 1, 0],
#         [2, -1, 0],
#         [-1, -1, 0],
#         [1, 1, 0],
#     ])
#
#     plane_normals_111_gamma_hkl = np.array([
#         [1, 1, 1],
#         [-1, 1, 1],
#         [1, -1, 1],
#         [1, 1, -1],
#         [-1, -1, 1],
#         [-1, 1, -1],
#         [1, -1, -1],
#         [-1, -1, -1],
#     ])
#
#     directions_110_gamma_hkl = np.array([
#         [1, 1, 0],
#         [-1, 1, 0],
#         [1, -1, 0],
#         [-1, -1, 0],
#         [1, 0, 1],
#         [-1, 0, 1],
#         [1, 0, -1],
#         [-1, 0, -1],
#         [0, 1, 1],
#         [0, -1, 1],
#         [0, 1, -1],
#         [0, -1, -1],
#     ])
#
#     plane_normals_parallel = False
#     for plane_normal_eps_hkl in plane_normals_0001_epsilon_hkl:
#         if not plane_normals_parallel:
#             for plane_normal_gam_hkl in plane_normals_111_gamma_hkl:
#                 if not plane_normals_parallel:
#                     plane_normal_eps_sample = hexagonal_grain.hkl_vec_as_sample_vec(plane_normal_eps_hkl)
#                     plane_normal_gam_sample = cubic_grain.hkl_vec_as_sample_vec(plane_normal_gam_hkl)
#
#                     angle_between_normals_rad = angle_between_vectors_rad(plane_normal_eps_sample,
#                                                                           plane_normal_gam_sample)
#                     angle_between_normals_deg = np.rad2deg(angle_between_normals_rad)
#                     if angle_between_normals_deg < misd_tol:
#                         plane_normals_parallel = True
#
#     direcs_parallel = False
#     for direc_eps_hkl in directions_1bar_2_1bar_0_epsilon_hkl:
#         if not direcs_parallel:
#             for direc_gam_hkl in directions_110_gamma_hkl:
#                 if not direcs_parallel:
#                     direc_eps_sample = hexagonal_grain.hkl_vec_as_sample_vec(direc_eps_hkl)
#                     direc_gam_sample = cubic_grain.hkl_vec_as_sample_vec(direc_gam_hkl)
#
#                     angle_between_direcs_rad = angle_between_vectors_rad(direc_eps_sample, direc_gam_sample)
#                     angle_between_direcs_deg = np.rad2deg(angle_between_direcs_rad)
#                     if angle_between_direcs_deg < misd_tol:
#                         direcs_parallel = True
#
#     if plane_normals_parallel and direcs_parallel:
#         return True
#     else:
#         return False


def get_Pitsch_variant(gamma_grain, aprime_grain):
    observed_parent_ori = gamma_grain.U_sample
    observed_child_ori = aprime_grain.U_sample

    misds_array = []

    for variant_mori in Pitsch_variant_matrices:
        calculated_child_ori = observed_parent_ori @ variant_mori

        misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

        misds_array.append(misd)

    # returns variant in 0-start array order
    # so variant 0 here is MTEX variant 1
    # variant 1 here is MTEX variant 2
    # etc
    return np.argmin(misds_array)


def get_G_T_variant(gamma_grain, aprime_grain):
    observed_parent_ori = gamma_grain.U_sample
    observed_child_ori = aprime_grain.U_sample

    misds_array = []

    for variant_mori in G_T_variant_matrices:
        calculated_child_ori = observed_parent_ori @ variant_mori

        misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

        misds_array.append(misd)

    # returns variant in 0-start array order
    # so variant 0 here is MTEX variant 1
    # variant 1 here is MTEX variant 2
    # etc
    return np.argmin(misds_array)


def get_N_W_variant(gamma_grain, aprime_grain):
    observed_parent_ori = gamma_grain.U_sample
    observed_child_ori = aprime_grain.U_sample

    misds_array = []

    for variant_mori in N_W_variant_matrices:
        calculated_child_ori = observed_parent_ori @ variant_mori

        misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

        misds_array.append(misd)

    # returns variant in 0-start array order
    # so variant 0 here is MTEX variant 1
    # variant 1 here is MTEX variant 2
    # etc
    return np.argmin(misds_array)


def get_K_S_variant(gamma_grain, aprime_grain):
    observed_parent_ori = gamma_grain.U_sample
    observed_child_ori = aprime_grain.U_sample

    misds_array = []

    for variant_mori in K_S_variant_matrices:
        calculated_child_ori = observed_parent_ori @ variant_mori

        misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), cubic_symm_ops)

        misds_array.append(misd)

    # returns variant in 0-start array order
    # so variant 0 here is MTEX variant 1
    # variant 1 here is MTEX variant 2
    # etc
    return np.argmin(misds_array)


def get_S_N_variant(cubic_grain, hexagonal_grain):
    # calculate variant ID in a similar way to MTEX
    # from mtex/geometry/misorientation/calcVariantID.m:
    # % all child variants
    # childVariants  = variants(p2c, parentOri);
    #
    # if size(childVariants,1) == 1
    #     childVariants = repmat(childVariants,length(childOri),1);
    # end
    #
    # % compute distance to all possible variants
    # d = dot(childVariants,repmat(childOri(:),1,size(childVariants,2)));
    #
    # % take the best fit
    # [~,childId] = max(d,[],2);

    observed_parent_ori = cubic_grain.U_sample
    observed_child_ori = hexagonal_grain.U_sample

    misds_array = []

    for variant_mori in S_N_variant_matrices:
        calculated_child_ori = observed_parent_ori @ variant_mori

        misd = disorientation_single_numba((calculated_child_ori, observed_child_ori), hex_symm_ops)

        misds_array.append(misd)

    # returns variant in 0-start array order
    # so variant 0 here is MTEX variant 1
    # variant 1 here is MTEX variant 2
    # etc
    return np.argmin(misds_array)


@njit
def get_S_N_deformation_tensors_from_mtex_variant_id(mtex_variant_id):
    """
    Generates deformation tensors for all epsilon martensite variants from S-N OR
    Using Table 1 of 10.1016/j.msea.2006.11.112
    # Analysis of the γ-ɛ-α' variant selection induced by 10% plastic deformation in 304 stainless steel at -60°C
    # Humbert, Petit, Bolle, Gey
    # MS&E A 2007 page 508
    """
    # we have eps in the austenite reference frame K_y
    # now we do K_y -> K_e (4 variants) -> K_y

    # e->y rotation matrix
    dg = S_N_variant_matrices[0]

    # e->y rotation matrix of this specific variant
    mori = S_N_variant_matrices[mtex_variant_id]

    # Symmetry.hexagonal.symmetry_operators().take([0, 2, 4], axis=0)
    # equivalent hexagonal symmetry on 111 lattice of y
    all_H = np.array([
        [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ],
        [
            [-0.5, 0.8660254, 0.],
            [-0.8660254, -0.5, 0.],
            [0., 0., 1.]
        ],
        [
            [-0.5, -0.8660254, 0.],
            [0.8660254, -0.5, 0.],
            [0., 0., 1.]
        ]
    ])

    # define single deformation matrix in y reference frame
    # corresponds to first mode of first epsilon variant
    eps_K_y = np.array([
        [2. / 24, 2. / 24, -1. / 24],
        [2. / 24, 2. / 24, -1. / 24],
        [-1. / 24, -1. / 24, -4. / 24]
    ])

    # make array to hold the tensors in
    deformation_tensors_array = np.zeros((3, 3, 3))
    for inc, H_i in enumerate(all_H):
        # rotate K_y into epsilon reference frame, always using first variant
        # introduce hexagonal symmetry to the variant
        # three specific symmetry elements to generate desired deformation tensors
        # eps_K_e = rotate_tensor(eps_K_y, (S_N_variant_matrices[0] @ symm_op).T)

        # rotate back into K_y using the variant
        # eps_K_y_again = rotate_tensor(eps_K_e, mori)

        overall_rotation_tensor = mori @ H_i.T @ dg.T

        eps_K_y_again = overall_rotation_tensor @ eps_K_y @ overall_rotation_tensor.T

        deformation_tensors_array[inc] = eps_K_y_again

    return deformation_tensors_array


@njit
def double_dot_product(A, B):
    """
    Equivalent to np.einsum('ij,ij',A,B)
    """
    return A[0, 0] * B[0, 0] + \
        A[0, 1] * B[0, 1] + \
        A[0, 2] * B[0, 2] + \
        A[1, 0] * B[1, 0] + \
        A[1, 1] * B[1, 1] + \
        A[1, 2] * B[1, 2] + \
        A[2, 0] * B[2, 0] + \
        A[2, 1] * B[2, 1] + \
        A[2, 2] * B[2, 2]


@njit
def predict_S_N_variant_for_grain(U, sig_K_s):
    """
    Predicts S-N martensite variant that will form in grain of orientation U
    Given a stress state in the sample reference frame of sig_K_s
    After 10.1016/j.msea.2006.11.112 and 10.1016/j.actamat.2013.11.001
    """

    # Transform stress to grain frame
    sig_K_y = U.T @ sig_K_s @ U

    # make array to store max E for each variant
    E_array = np.zeros(4)

    # iterate over variant IDs
    for mtex_var_id in range(0, 4):
        # make array to store E for each deformation matrix for this variant
        this_var_E_array = np.zeros(3)

        # get 3 deformation matrices for this variant
        eps_K_y_array = get_S_N_deformation_tensors_from_mtex_variant_id(mtex_var_id)

        # iterate over them
        for i in range(3):
            # extract individual deformation matrix
            eps_K_y_i = eps_K_y_array[i]

            # calculate work from that matrix
            E = (1. / 2) * double_dot_product(sig_K_y, eps_K_y_i)

            this_var_E_array[i] = E

        max_E_for_this_var = np.max(this_var_E_array)
        E_array[mtex_var_id] = max_E_for_this_var

    return np.max(E_array), np.argmax(E_array)
