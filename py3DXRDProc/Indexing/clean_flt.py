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

# Modified from ImageD11/sandbox/clean_data.py at https://github.com/FABLE-3DXRD/ImageD11/

from ImageD11 import transformer

import logging
log = logging.getLogger(__name__)


def clean_flt(merged_peaks, the_pars, rings, the_clean_peaks, tth_tol):
    """
    Cleans merged peaks file
    :param merged_peaks: merged flt file to clean
    :param the_pars: ImageD11 parameter file used to clean peaks
    :param rings: list of integers - rings to clean down to
    :param the_clean_peaks: output clean peaks flt file
    """
    my_transformer = transformer.transformer()
    my_transformer.loadfiltered(merged_peaks)
    # Also remove some small peaks
    my_transformer.colfile.filter(my_transformer.colfile.Number_of_pixels > 3)
    my_transformer.loadfileparameters(the_pars)
    this_outfile = the_clean_peaks
    my_transformer.updateparameters()
    my_transformer.colfile.updateGeometry(my_transformer.parameterobj)
    tth = my_transformer.colfile.tth
    my_transformer.addcellpeaks()
    rh = my_transformer.unitcell.ringhkls
    peaks = list(rh.keys())
    peaks.sort()
    m = 0
    for i, d in enumerate(peaks):
        this_tth = my_transformer.theorytth[i]
        multiplicity = len(rh[d])

        mask = ((this_tth + tth_tol) > tth) & (tth > (this_tth - tth_tol))
        n_peaks = mask.sum()
        log.info(f"Ring: {i} \t ds: {d} \t tth: {this_tth} \t (h k l): {rh[d][0]} {rh[d][0]} {rh[d][0]} \t multiplicity: {multiplicity} \t n_peaks: {n_peaks} \t n_peaks/multiplicity: {1.0 * n_peaks / multiplicity}")
        m += mask.sum()
    log.info(f"{m}, {tth.shape}")
    rs = rings
    this_tth = my_transformer.theorytth[rs[0]]
    mask = ((this_tth + tth_tol) > tth) & (tth > (this_tth - tth_tol))
    for r in rs[1:]:
        this_tth = my_transformer.theorytth[r]
        mask |= ((this_tth + tth_tol) > tth) & (tth > (this_tth - tth_tol))

    my_transformer.colfile.filter(mask)
    my_transformer.write_colfile(this_outfile)