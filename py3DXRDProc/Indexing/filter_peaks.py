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

import numpy as np
from ImageD11.columnfile import columnfile

import logging
log = logging.getLogger(__name__)


def filter_peaks(input_flt_path, output_flt_path, mask):
    try:
        c = columnfile(input_flt_path)
        log.info(f"{c.nrows} peaks loaded")
    except:
        # Couldn't load the columnfile (probably no peaks in it)
        return None

    # Mask is a boolean array where bad pixels are True and good pixels are false
    # It's already been dilated by 1 pixel with scipy.ndimage.binary_dilation

    # Extract the bounding box of each peak (in pixels) using c.Min_s, c.Max_s, c.Min_f, c.Max_f
    # For each bounding box, crop the dilated mask to the bounding box
    # If any pixels in the cropped mask contain a 1 (a masked-out value), then that peak is too close to a masked pixel

    # bool_list = [mask[mins:maxs, minf:maxf].any() for (mins, maxs, minf, maxf) in zip(min_s, max_s, min_f, max_f)]

    bool_list = [mask[int(mins)-1:int(maxs)+1, int(minf)-1:int(maxf)+1].any() for (mins, maxs, minf, maxf) in zip(c.Min_s, c.Max_s, c.Min_f, c.Max_f)]

    bool_list_as_filter = np.invert(bool_list)
    c.filter(bool_list_as_filter)

    log.info(f"{c.nrows} peaks survived")

    c.writefile(output_flt_path)