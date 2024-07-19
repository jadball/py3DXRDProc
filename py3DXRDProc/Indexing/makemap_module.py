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

# Modified from ImageD11/scripts/makemap.py at https://github.com/FABLE-3DXRD/ImageD11/

import argparse

import ImageD11.refinegrains
import numpy
import xfab
from ImageD11 import grain, simplex, parameters
from ImageD11.refinegrains import refinegrains

import logging

log = logging.getLogger(__name__)

xfab.CHECKS.activated = False


def triclinic(cp):
    return cp


# Custom refinegrains that can remove grains that are too small
class refinegrains_filtered(refinegrains):
    def __init__(self, tolerance=0.01, intensity_tth_range=(6.1, 6.3),
                 latticesymmetry=triclinic,
                 OmFloat=True, OmSlop=0.25):
        """

        """
        self.OMEGA_FLOAT = OmFloat
        self.slop = OmSlop
        if self.OMEGA_FLOAT:
            log.debug(f"Using {self.slop} degree slop")
        else:
            log.debug("Omega is used as observed")
        self.tolerance = tolerance
        # list of ubi matrices (1 for each grain in each scan)
        self.grainnames = []
        self.ubisread = {}
        self.translationsread = {}
        # list of scans and corresponding data
        self.scannames = []
        self.scantitles = {}
        self.scandata = {}
        # grains in each scan
        self.grains = {}
        self.grains_to_refine = []
        self.latticesymmetry = latticesymmetry
        # ?
        self.drlv = None
        self.parameterobj = parameters.parameters(**self.pars)
        self.intensity_tth_range = intensity_tth_range
        self.recompute_xlylzl = False
        for k, s in list(self.stepsizes.items()):
            self.parameterobj.stepsizes[k] = s

    def readubis_keep_names(self, filename):
        """
        Read ubi matrices from a text file
        """
        try:
            ul = grain.read_grain_file(filename)
        except:
            log.error(f"{filename}, {type(filename)}")
            raise
        for i, g in enumerate(ul):
            # name = filename + "_" + str(i)
            # Hmmm .... multiple grain files?
            name = int(g.name.rstrip("\n"))
            self.grainnames.append(name)
            self.ubisread[name] = g.ubi
            self.translationsread[name] = g.translation
        # print "Grain names",self.grainnames

    def refinepositions(self, quiet=True, maxiters=100):
        self.assignlabels(quiet=quiet)
        ks = list(self.grains.keys())
        ks.sort()
        # assignments are now fixed
        tolcache = self.tolerance
        self.tolerance = 1.0
        for key in ks:
            g = key[0]
            self.grains_to_refine = [key]
            self.parameterobj.varylist = ['t_x', 't_y', 't_z']
            self.set_translation(key[0], key[1])
            guess = self.parameterobj.get_variable_values()
            inc = self.parameterobj.get_variable_stepsizes()

            s = simplex.Simplex(self.gof, guess, inc)

            do_monitor = not quiet

            newguess, error, iter = s.minimize(maxiters=maxiters, monitor=do_monitor)

            self.grains[key].translation[0] = self.parameterobj.parameters['t_x']
            self.grains[key].translation[1] = self.parameterobj.parameters['t_y']
            self.grains[key].translation[2] = self.parameterobj.parameters['t_z']
            log.debug(f"{key}, {self.grains[key].translation}")
            self.refine(self.grains[key].ubi, quiet=quiet)
        self.tolerance = tolcache

    def generate_grains(self, quiet=False):
        t = numpy.array([self.parameterobj.parameters[s]
                         for s in ['t_x', 't_y', 't_z']])
        for grainname in self.grainnames:
            for scanname in self.scannames:
                try:
                    gr = self.grains[(grainname, scanname)]
                except KeyError:
                    if self.translationsread[grainname] is None:
                        self.grains[(grainname, scanname)] = grain.grain(
                            self.ubisread[grainname], translation=t)
                        self.grains[(grainname, scanname)].name = \
                            (str(grainname) + ":" + scanname).replace(" ", "_")
                    else:
                        self.grains[(grainname, scanname)] = grain.grain(
                            self.ubisread[grainname],
                            translation=self.translationsread[grainname])
                        self.grains[(grainname, scanname)].name = \
                            (str(grainname) + ":" + scanname).replace(" ", "_")
        for scanname in self.scannames:
            self.reset_labels(scanname)

    def reset_labels(self, scanname):
        log.debug("Resetting labels")
        try:
            x = self.scandata[scanname].xc
            y = self.scandata[scanname].yc
        except AttributeError:
            x = self.scandata[scanname].sc
            y = self.scandata[scanname].fc
        om = self.scandata[scanname].omega
        # only for this grain
        self.scandata[scanname].labels = self.scandata[scanname].labels * 0 - 2
        self.scandata[scanname].drlv2 = self.scandata[scanname].drlv2 * 0 + 1
        for g in self.grainnames:
            self.grains[(g, scanname)].x = x
            self.grains[(g, scanname)].y = y
            self.grains[(g, scanname)].om = om

    def savegrains_filtered(self, filename, minpks, sort_npks=True, filter_grains=False):
        """
        Save the refined grains
        """
        ks = list(self.grains.keys())
        # sort by number of peaks indexed to write out
        if sort_npks:
            #        npks in x array
            order = numpy.argsort([self.grains[k].npks for k in ks])
            ks = [ks[i] for i in order[::-1]]
        else:
            ks.sort()
        gl = [(self.grains[k], k) for k in ks]

        # Update the datafile and grain names reflect indices in grain list

        for g, k in gl:
            name, fltname = g.name.split(":")
            assert fltname in self.scandata, "Sorry - logical flaw"
            assert len(list(self.scandata.keys())) == 1, "Sorry - need to fix for multi data"
            self.set_translation(k[0], fltname)
            self.compute_gv(g, update_columns=True)
            numpy.put(self.scandata[fltname].gx, g.ind, self.gv[:, 0])
            numpy.put(self.scandata[fltname].gy, g.ind, self.gv[:, 1])
            numpy.put(self.scandata[fltname].gz, g.ind, self.gv[:, 2])
            hkl_real = numpy.dot(g.ubi, self.gv.T)
            numpy.put(self.scandata[fltname].hr, g.ind, hkl_real[0, :])
            numpy.put(self.scandata[fltname].kr, g.ind, hkl_real[1, :])
            numpy.put(self.scandata[fltname].lr, g.ind, hkl_real[2, :])
            hkl = numpy.floor(hkl_real + 0.5)
            numpy.put(self.scandata[fltname].h, g.ind, hkl[0, :])
            numpy.put(self.scandata[fltname].k, g.ind, hkl[1, :])
            numpy.put(self.scandata[fltname].l, g.ind, hkl[2, :])
            # Count "uniq" reflections...
            sign_eta = numpy.sign(self.scandata[fltname].eta_per_grain[g.ind])
            uniq_list = [(int(h), int(k), int(l), int(s)) for
                         (h, k, l), s in zip(hkl.T, sign_eta)]
            g.nuniq = len(set(uniq_list))

        if filter_grains:
            log.debug("Filtering grains using makemap")
            log.debug(f"Going from {len(gl)} grains to {len([g[0] for g in gl if g[0].nuniq >= minpks])} grains")
            grain.write_grain_file(filename, [g[0] for g in gl if g[0].nuniq >= minpks])
        else:
            grain.write_grain_file(filename, [g[0] for g in gl])


def makemap(peaks, pars, ubis, tol, omega_slop, tth_range, sort_npks, new_peaks, new_ubis, minpks, symmetry="triclinic",
            latticesymmetry="triclinic", filter_grains=False):
    """
    Copy of ImageD11/scripts/makemap.py
    https://github.com/FABLE-3DXRD/ImageD11/
    """
    try:
        if tth_range is None:
            tthr = (0., 180.)
        else:
            tthr = tth_range
            if len(tthr) == 1:
                tthr = (0, tthr[0])
            log.debug(f"Using tthrange {tthr}")
        func = getattr(ImageD11.refinegrains, latticesymmetry)
        o = refinegrains_filtered(intensity_tth_range=tthr,
                                  latticesymmetry=func,
                                  OmFloat=True,
                                  OmSlop=omega_slop)
    except:
        raise
    o.loadparameters(pars)
    log.debug("got pars")
    o.loadfiltered(peaks)
    log.debug("got filtered")
    o.readubis(ubis)

    if symmetry != "triclinic":
        # Grainspotter will have already done this
        log.debug("transform to uniq")
        o.makeuniq(symmetry)
    log.debug("got ubis")
    o.tolerance = float(tol)
    log.debug("generating")
    o.generate_grains()
    log.debug("Refining positions too")
    o.refinepositions()
    log.debug("Done refining positions too")
    o.savegrains_filtered(new_ubis, minpks=minpks, sort_npks=sort_npks, filter_grains=filter_grains)
    col = o.scandata[peaks].writefile(peaks + ".new")
    if new_peaks is not None:
        log.debug("re-assignlabels")
        o.assignlabels()
        col = o.scandata[peaks].copy()
        log.debug(f"Before filtering: {col.nrows}")
        col.filter(col.labels < -0.5)
        log.debug(f"After filtering: { col.nrows}")
        col.writefile(new_peaks)