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

# Modified from ImageD11/ImageD11/grid_index_parallel.py at https://github.com/FABLE-3DXRD/ImageD11/

import multiprocessing as mp
import sys

import numpy as np
import xfab
from ImageD11 import transformer, unitcell, refinegrains, indexing, grain, sym_u

if "win" in sys.platform:
    nulfile = "NUL"
else:
    nulfile = "/dev/null"


def domap(pars,
          colfile,
          grains,
          gridpars):
    """
    mapping function - does what makemap.py does, but in a function
    """
    if 'FITPOS' not in gridpars:
        gridpars['FITPOS'] = True

    OmSlop = gridpars['OMEGAFLOAT']
    OmFloat = OmSlop > 0
    #
    ss = sys.stdout  # turns off printing
    if gridpars['NUL']:
        NUL = open(nulfile, "w")
        sys.stdout = NUL
    for tol in gridpars['TOLSEQ']:
        o = refinegrains.refinegrains(OmFloat=OmFloat, OmSlop=OmSlop,
                                      tolerance=tol,
                                      intensity_tth_range=(0, 180),
                                      )
        o.parameterobj = pars
        # o.loadfiltered ...
        o.scannames = ["internal"]
        o.scantitles = colfile.titles
        o.scandata["internal"] = colfile
        o.tolerance = tol
        # o.readubis( grainsfile )
        for i, g in enumerate(grains):
            name = i
            o.grainnames.append(i)
            o.ubisread[name] = g.ubi
            o.translationsread[name] = g.translation
        if gridpars['SYMMETRY'] != "triclinic":
            o.makeuniq(gridpars['SYMMETRY'])
        o.generate_grains()
        if gridpars['FITPOS']:
            o.refinepositions()
        else:
            o.assignlabels()
            for key in o.grains.keys():
                g = o.grains[key]
                g.set_ubi(o.refine(g.ubi, quiet=False))
        # This fills in the uniq for each grain
        o.savegrains(nulfile, sort_npks=False)
        if 'NUNIQ' in gridpars:
            keep = lambda g: g.nuniq > gridpars['NUNIQ'] and g.npks > gridpars['NPKS']
        else:
            keep = lambda g: g.npks > gridpars['NPKS']
        gl = [g for g in o.grains.values() if keep(g)]
        if len(gl) == 0:
            break
        grains = gl
    if gridpars['NUL']:
        sys.stdout = ss
    return gl


def doindex(gve, x, y, z, w, gridpars):
    """
    Try to index some g-vectors
    """
    NPKS = gridpars['NPKS']
    UC = gridpars['UC']
    TOLSEQ = gridpars['TOLSEQ']
    COSTOL = gridpars['COSTOL']
    DSTOL = gridpars['DSTOL']
    if 'HKLTOL' in gridpars:
        HKLTOL = gridpars['HKLTOL']
    else:
        HKLTOL = TOLSEQ[0]
    if "2RFIT" in gridpars:
        DOFIT = gridpars['2RFIT']
    else:
        DOFIT = False
    ss = sys.stdout  # turns off printing
    if gridpars['NUL']:
        NUL = open(nulfile, "w")
        sys.stdout = NUL
    myindexer = indexing.indexer(
        wavelength=w,
        unitcell=UC,
        gv=gve.T
    )
    # added in indexer.__init__
    # myindexer.ds = np.sqrt( (gve * gve).sum(axis=0) )
    # myindexer.ga = np.zeros(len(myindexer.ds),int)-1 # Grain assignments
    for ring1 in gridpars['RING1']:
        for ring2 in gridpars['RING2']:
            myindexer.parameterobj.set_parameters({
                'cosine_tol': COSTOL,
                'ds_tol': DSTOL,
                'minpks': NPKS,
                'max_grains': 1000,
                'hkl_tol': HKLTOL,
                'ring_1': ring1,
                'ring_2': ring2
            })
            myindexer.loadpars()
            myindexer.assigntorings()
            try:
                myindexer.find()
                myindexer.scorethem(fitb4=DOFIT)
            except:
                pass
    # filter out crap
    vol = 1 / np.linalg.det(UC.B)
    grains = [grain.grain(ubi, [x, y, z]) for ubi in myindexer.ubis
              if np.linalg.det(ubi) > vol * 0.5]
    if gridpars['NUL']:
        sys.stdout = ss
    return grains


def test_many_points(args):
    """
    Grid index - loop over points
    Places the results in a multiprocessing Queue
    """
    colfile, parameters, translation, gridpars = args
    t_x, t_y, t_z = translation
    print(f"({t_x}, {t_y}, {t_z}): Started")
    mytransformer = transformer.transformer()
    mytransformer.loadfiltered(colfile)
    mytransformer.loadfileparameters(parameters)
    w = mytransformer.parameterobj.get("wavelength")

    mytransformer.updateparameters()
    mytransformer.parameterobj.set_parameters({'t_x': t_x, 't_y': t_y, 't_z': t_z})
    mytransformer.compute_tth_eta()
    mytransformer.computegv()
    #    mytransformer.savegv( tmp+".gve" )
    gve = np.vstack((mytransformer.colfile.gx, mytransformer.colfile.gy, mytransformer.colfile.gz))

    print(f"({t_x}, {t_y}, {t_z}): Indexing started")
    grains = doindex(gve, t_x, t_y, t_z, w, gridpars)
    ng = len(grains)
    if ng > 0:
        print(f"({t_x}, {t_y}, {t_z}): Indexing found {len(grains)}")
        print(f"({t_x}, {t_y}, {t_z}): Makemap started")
        grains = domap(mytransformer.parameterobj,
                       mytransformer.colfile,
                       grains,
                       gridpars)
        if len(grains) > 0:
            print(f"({t_x}, {t_y}, {t_z}): Makemap found {len(grains)}")
            return grains
        else:
            print(f"({t_x}, {t_y}, {t_z}): Makemap killed all grains")
            return "NOGRAINS"
    else:
        print(f"({t_x}, {t_y}, {t_z}): Indexing found no grains")
        return "NOGRAINS"


def worker(tasks_queue, results_queue):
    while True:
        input_data = tasks_queue.get()
        if input_data is None:  # no more tasks
            results_queue.put(None)
            break
        result = test_many_points(input_data)
        results_queue.put(result)


def initgrid(fltfile, parfile, tmp, gridpars):
    """
    Sets up a grid indexing by preparing the unitcell for indexing
    and checking the columns we want are in the colfile
    """
    print("Initialising grid")
    mytransformer = transformer.transformer()
    mytransformer.loadfiltered(fltfile)
    mytransformer.loadfileparameters(parfile)
    gridpars['UC'] = unitcell.unitcell_from_parameters(mytransformer.parameterobj)
    col = mytransformer.colfile
    if not "drlv2" in col.titles:
        col.addcolumn(np.ones(col.nrows, float), "drlv2")
    if not "labels" in col.titles:
        col.addcolumn(np.ones(col.nrows, float) - 2, "labels")
    if not "sc" in col.titles:
        assert "xc" in col.titles
        col.addcolumn(col.xc.copy(), "sc")
    if not "fc" in col.titles:
        assert "yc" in col.titles
        col.addcolumn(col.yc.copy(), "fc")
    mytransformer.colfile.writefile("%s.flt" % (tmp))
    return gridpars


class uniq_grain_list(object):
    """
    Cope with finding the same grain over and over...
    """

    def __init__(self, symmetry, toldist, tolangle, grains=None):
        self.grp = getattr(sym_u, symmetry)()
        self.dt2 = toldist * toldist
        self.tar = np.radians(tolangle)
        self.uniqgrains = []
        if grains is not None:
            self.add(grains)

    def add(self, grains):
        for i, gnew in enumerate(grains):
            newgrain = True
            for gold in self.uniqgrains:
                dt = gnew.translation - gold.translation
                dt2 = np.dot(dt, dt)
                if dt2 > self.dt2:
                    continue
                aumis = np.dot(gold.asymusT, gnew.U)
                arg = (aumis[:, 0, 0] + aumis[:, 1, 1] + aumis[:, 2, 2] - 1.) / 2.
                angle = np.arccos(np.clip(arg, -1, 1)).min()
                if angle < self.tar:
                    # too close in angle and space
                    print("           matched", i, np.degrees(angle), np.sqrt(dt2))
                    gold.nfound += 1
                    newgrain = False
                    break
            if newgrain:
                self.append_uniq(gnew)

    def append_uniq(self, g):
        symubis = [np.dot(o, g.ubi) for o in self.grp.group]
        g.asymusT = np.array([xfab.tools.ubi_to_u_b(ubi)[0].T for ubi in symubis])
        g.nfound = 1
        self.uniqgrains.append(g)


def grid_index_parallel(fltfile, parfile, tmp, gridpars, translations):
    gridpars = initgrid(fltfile, parfile, tmp, gridpars)
    print("Initialised grid")

    NPR = int(gridpars["NPROC"])

    args = [("%s.flt" % (tmp), parfile, t, gridpars) for i, t in enumerate(translations)]

    tasks_queue = mp.Queue()
    results_queue = mp.Queue()

    # Set up pool
    with mp.Pool(NPR, worker, (tasks_queue, results_queue)):
        # Add tasks to the queue
        print("Adding tasks to the queue")
        [tasks_queue.put(arg) for arg in args]

        # Empty out the queue after finishing
        for _ in range(NPR):
            tasks_queue.put(None)

        unique_grain_list_handler = uniq_grain_list(gridpars['SYMMETRY'], gridpars['toldist'], gridpars['tolangle'])

        lastsave = 0
        while NPR:
            item = results_queue.get()
            if item is None:
                NPR -= 1
                continue
            elif item == "NOGRAINS":
                continue
            else:
                grs = item
                gb4 = len(unique_grain_list_handler.uniqgrains)
                unique_grain_list_handler.add(grs)
                gnow = len(unique_grain_list_handler.uniqgrains)
                print("Got % 5d new %d from %d" % (gnow, gnow - gb4, len(grs)))
                if len(unique_grain_list_handler.uniqgrains) > lastsave:
                    lastsave = len(unique_grain_list_handler.uniqgrains)
                    grain.write_grain_file("all" + tmp + ".map", unique_grain_list_handler.uniqgrains)

    tasks_queue.close()
    tasks_queue.join_thread()
    results_queue.close()
    results_queue.join_thread()
    del tasks_queue
    del results_queue

    # write here to be on the safe side ....
    grain.write_grain_file("all" + tmp + ".map", unique_grain_list_handler.uniqgrains)


if __name__ == "__main__":
    print("#Here is an example script")
    print("""
import sys, random
from ImageD11.grid_index_parallel import grid_index_parallel

if __name__=="__main__":
    # You need this idiom to use multiprocessing on windows (script is imported again)
    gridpars = {
        'DSTOL' : 0.004,
        'OMEGAFLOAT' : 0.13,
        'COSTOL' : 0.002,
        'NPKS' : int(  sys.argv[4] ),
        'TOLSEQ' : [ 0.02, 0.015, 0.01],
        'SYMMETRY' : "cubic",
        'RING1'  : [5,10],
        'RING2' : [5,10],
        'NUL' : True,
        'FITPOS' : True,
        'tolangle' : 0.25,
        'toldist' : 100.,
        'NPROC' : None, # guess from cpu_count
        'NTHREAD' : 2 ,
    }

    # grid to search
    translations = [(t_x, t_y, t_z) 
        for t_x in range(-500, 501, 50)
        for t_y in range(-500, 501, 50) 
        for t_z in range(-500, 501, 50) ]
    # Cylinder: 
    # translations = [( x,y,z) for (x,y,z) in translations if (x*x+y*y)< 500*500 ]
    #
    random.seed(42) # reproducible
    random.shuffle(translations)

    fltfile = sys.argv[1]
    parfile = sys.argv[2]
    tmp     = sys.argv[3]
    grid_index_parallel( fltfile, parfile, tmp, gridpars, translations )
""")
