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

# Modified from ImageD11/scripts/merge_flt.py at https://github.com/FABLE-3DXRD/ImageD11/

help = """
Reads in a series of flt files to merge together different threshold levels

Takes all peaks from the highest level.
For lower levels, it adds new peaks which are found if they are not 
already present
If they correspond to a previous peak, it chooses between the old and the
new according to the centre of mass. If the lower thresholded peak's centre
of mass does not agree with the higher within pixel_tol, then the lower is
rejected

Peaks are identified as unique via their co-ordinates of their strongest
pixel (f,s,omega)
"""

from ImageD11 import transformer
from ImageD11.columnfile import newcolumnfile, columnfile
import numpy
import sys

import logging
log = logging.getLogger(__name__)


def do_merge(pars_arg, stem_arg, outf_arg, dpix_arg, thres_arg):
    try:
        pars = pars_arg
        stem = stem_arg
        outf = outf_arg
        dpix = dpix_arg
        thres = [int(v) for v in thres_arg]
    except:
        log.info("Usage: pars stem outputfile pixel_tol thresholds1  thresholds2 ...")
        log.info(help)
        sys.exit()

    assert outf[-4:] == ".flt", """output file should end in ".flt" """

    thres.sort()
    thres = thres[::-1]

    log.info(f"Using parameters {pars}")
    log.info("Merging files: ")
    for v in thres:
        log.info("%s_t%d_gaps_removed.flt" % (stem, v))
    log.info("Into output file %s" % (outf))

    # if raw_input("OK? [y/n]") not in ["Y","y"]:
    #    sys.exit()

    allpks = open(outf, "w")

    allpeaks = {}
    always_ignore = {}

    goodthres = []

    for v in thres:

        mytransformer = transformer.transformer()
        mytransformer.loadfileparameters(pars)

        flt = "%s_t%d_gaps_removed.flt" % (stem, v)

        log.info(flt)
        try:
            tc = columnfile(flt)
            if tc.nrows == 0:
                log.info(f"Skipped {tc}, no peaks")
                continue
            goodthres.append(v)
            mytransformer.loadfiltered(flt)
            mytransformer.compute_tth_eta()
            mytransformer.addcellpeaks()
        except:
            log.info(f"Skipped {v}, Exception reading {flt}")
            continue

        log.info(f"npeaks: {mytransformer.colfile.nrows}")

        # mytransformer.write_colfile(flt2)

        f = mytransformer.colfile.titles.index('sc')
        s = mytransformer.colfile.titles.index('fc')
        titles = mytransformer.colfile.titles
        nignore = 0
        nnew = 0
        nold = 0
        for i in range(mytransformer.colfile.nrows):
            # Position of max intensity
            key = (int(mytransformer.colfile.IMax_o[i] * 100),
                   int(mytransformer.colfile.IMax_s[i]),
                   int(mytransformer.colfile.IMax_f[i]))

            if key in always_ignore:
                nignore = nignore + 1
                continue

            if key in allpeaks:
                if v is goodthres[0]:
                    log.info(key)
                    log.info("duplicate")
                    # raise
                # This peak is already found
                # Should we replace it, or trash the lower threshold ??
                # previous is allpeaks[key]
                # current is mytransformer.colfile.bigarray[:,i]
                df = allpeaks[key][f] - mytransformer.colfile.bigarray[f, i]
                ds = allpeaks[key][s] - mytransformer.colfile.bigarray[s, i]
                dist2 = df * df + ds * ds
                if dist2 > dpix * dpix:
                    # ignore the weaker peak
                    # log.info "Ignoring weaker",
                    nignore = nignore + 1
                    always_ignore[key] = 1
                else:
                    # Replace the stronger peak with the weaker peak
                    allpeaks[key] = mytransformer.colfile.bigarray[:, i]
                    nold = nold + 1
            else:
                nnew = nnew + 1
                allpeaks[key] = mytransformer.colfile.bigarray[:, i]

        log.info(f"total peaks: {len(list(allpeaks.keys()))} \t ignored: {nignore} \t new: {nnew} \t replacements: {nold}")
        assert nignore + nold + nnew == mytransformer.colfile.nrows

    keys = list(allpeaks.keys())

    keys.sort()

    # noinspection PyUnboundLocalVariable
    c = newcolumnfile(titles)

    assert len(titles) == len(allpeaks[keys[0]])

    bigarray = [allpeaks[k] for k in keys]

    c.bigarray = numpy.array(bigarray).T
    log.info(c.bigarray.shape)
    c.nrows = len(keys)
    c.set_attributes()
    c.setcolumn(numpy.array(list(range(len(keys)))), "spot3d_id")
    c.writefile(outf)

    mytransformer = transformer.transformer()
    mytransformer.loadfileparameters(pars)

    mytransformer.loadfiltered(outf)
    mytransformer.compute_tth_eta()
    mytransformer.addcellpeaks()
    mytransformer.computegv()

    mytransformer.write_colfile(outf)
    mytransformer.savegv(outf.replace(".flt", ".gve"))
