# py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
# University of Birmingham.
#
# Copyright (C) 2019-2024  James Ball
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

# Modified from fitpksEiger.ipynb at https://github.com/jonwright/EigerPowderSpatial

import fabio
import numpy as np
from ImageD11.transformer import transformer


def correct_flt(e2dx, e2dy, flt_in, pars, flt_out):
    df = fabio.open(e2dx).data
    ds = fabio.open(e2dy).data
    t = transformer()
    t.loadfiltered(flt_in)
    c = t.colfile
    c.sc[:] = ds[np.round(c.s_raw[:].astype(int)), np.round(c.f_raw[:].astype(int))] + c.s_raw[:]
    c.fc[:] = df[np.round(c.s_raw[:].astype(int)), np.round(c.f_raw[:].astype(int))] + c.f_raw[:]
    t.loadfileparameters(pars)
    t.compute_tth_eta()
    c.writefile(flt_out)
