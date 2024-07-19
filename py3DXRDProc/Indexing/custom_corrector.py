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

# Modified from ImageD11/ImageD11/correct.py at https://github.com/FABLE-3DXRD/ImageD11/

import numpy

import logging
log = logging.getLogger(__name__)


# fixme - subtracting median filtered
# coarser medians - eg rebinned too

def correct(data_object,
            dark=None,
            monitorval=None,
            monitorcol=None):
    """
    Does the dark and flood corrections
    Also PIL filters
    """

    picture = data_object.data.astype(numpy.float32)

    # Select the masked-off regions (at DLS, all masked-off regions are set to a value of -1)
    masked_regions = picture < 0

    if dark is not None:
        # This is meant to be quicker
        log.info("Subtracting background")
        picture = numpy.subtract(picture, dark, picture)

    if monitorcol is not None and monitorval is not None:
        log.info("Trying to scale to beam current")
        if monitorcol not in data_object.header:
            log.info(f"Missing header value for normalise, {monitorcol}, {data_object.filename}")
        else:
            try:
                scal = monitorval / float(data_object.header[monitorcol])
                picture = numpy.multiply(picture, scal, picture)
            except:
                log.info(f"Scale overflow, {monitorcol}, {monitorval}, {data_object.filename}")

    # Zero-out all negative regions
    picture[picture < 0] = 0

    # Re-mask the regions afterwards
    picture[masked_regions] = -1

    data_object.data = picture

    return data_object
