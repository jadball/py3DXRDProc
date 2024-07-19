# py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
# University of Birmingham.
#
# Copyright (C) 2019-2024  James Ball
# Copyright (C) 2005-2019  Jon Wright
#
# This file is part of py3DXRDProc.
#
# py3DXRDProc is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# py3DXRDProc is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.


import sys

from py3DXRDProc.Indexing.custom_gip_queues import grid_index_parallel


def main(peaks_in, pars, temp_out):
    gridpars = {
        'DSTOL': 0.0025,
        'OMEGAFLOAT': 0.05,
        'COSTOL': 0.001,
        'NPKS': 35,
        'NUNIQ': 12,
        'TOLSEQ': [0.050, 0.040, 0.030],
        'SYMMETRY': "cubic",
        'RING1': [0, 3],
        'RING2': [0, 3],
        'NUL': True,
        'FITPOS': True,
        'tolangle': 1,
        'toldist': 100,
        'NTHREAD': 2,
        'NPROC': 16,
    }

    translations = [(t_x, t_y, t_z) for t_x in range(-600, 601, 300) for t_y in range(-600, 601, 300) for t_z in range(-100, 101, 100)]

    grid_index_parallel(peaks_in, pars, temp_out, gridpars, translations)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
