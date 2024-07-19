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

import argparse
import multiprocessing
import os
from itertools import repeat

import ImageD11
import numpy as np
from ImageD11 import ImageD11options, grain, columnfile
from py3DXRDProc.Indexing.makemap_module import refinegrains_filtered, makemap
from py3DXRDProc.cluster import get_number_of_cores
from py3DXRDProc.grain import bootstrap_mean_grain_and_errors
from py3DXRDProc.io_tools import make_folder
from py3DXRDProc.phase import Phase
from tqdm import tqdm

import logging
log = logging.getLogger(__name__)


def get_options(parser):
    # Copyright (C) 2005-2019  Jon Wright
    # Modified from ImageD11/ImageD11/refinegrains.py at https://github.com/FABLE-3DXRD/ImageD11/

    parser.add_argument("-p", "--parfile", action="store",
                        dest="parfile",
                        type=ImageD11options.ParameterFileType(mode='r'),
                        help="Name of input parameter file")
    parser.add_argument("-u", "--ubifile", action="store",
                        dest="ubifile",
                        type=ImageD11options.UbiFileType(mode='r'),
                        help="Name of ubi file")
    parser.add_argument("-U", "--newubifile", action="store",
                        dest="newubifile",
                        type=ImageD11options.UbiFileType(mode='w'),
                        help="Name of new ubi file to output")
    parser.add_argument("-f", "--fltfile", action="store",
                        dest="fltfile",
                        type=ImageD11options.ColumnFileType(mode='r'),
                        help="Name of flt file")
    parser.add_argument("-F", "--newfltfile", action="store",
                        dest="newfltfile",
                        type=ImageD11options.ColumnFileType(mode='w'),
                        help="Name of flt file containing unindexed peaks")
    lattices = ["cubic", "hexagonal", "trigonalH", "trigonalP",
                "tetragonal", "orthorhombic", "monoclinic_a",
                "monoclinic_b", "monoclinic_c", "triclinic"]
    parser.add_argument("-s", "--sym", action="store",
                        dest="symmetry",  # type="choice",
                        default="triclinic",
                        choices=lattices,
                        help="Lattice symmetry for choosing orientation")
    parser.add_argument("-t", "--tol", action="store",
                        dest="tol", type=float,
                        default=0.25,
                        help="Tolerance to use in peak assignment, default=%f" % (0.25))
    parser.add_argument("--samples", action="store",
                        dest="samples", type=int,
                        default=500,
                        help="Number of samples to take, default=%f" % (500))
    parser.add_argument("--fraction", action="store",
                        dest="fraction", type=float,
                        default=0.5,
                        help="Fraction of used peaks, float between 0 and 1, default=%f" % (0.5))
    parser.add_argument("--omega_slop", action="store", type=float,
                        dest="omega_slop",
                        default=0.5,
                        help="Omega slop (step) size")
    parser.add_argument("--phase_name", action="store", type=str,
                        dest="phase_name",
                        default="phase_name",
                        help="Name of phase of map")
    parser.add_argument("--working_dir", action="store", type=str,
                        dest="working_dir",
                        help="Path to working directory for output map")
    parser.add_argument("--filter_grains", action="store_true",
                        dest="filter_grains",
                        default=False,
                        help="Filter grains to have a minimum number of peaks")
    parser.add_argument("--minpks", action="store",
                        dest="minpks", type=int,
                        default=40,
                        help="Minimum number of peaks to filter grains to, default=%f" % (40))

    return parser


def bootstrap(input_map, parfile, peaksfile, working_dir, output_map_filename, symmetry, tol, samples, fraction, omega_slop, phase_name, filter_grains, minpks, do_parallel=True):
    # Adapted from original bootstrap script by Younes El-Hachi <younes.elhachi@toulouse-inp.fr>, included below:
    # from ImageD11 import columnfile, grain
    # import numpy as np, os, sys, multiprocessing
    #
    # try:
    #     g = grain.read_grain_file(sys.argv[1])
    #     col = columnfile.columnfile(sys.argv[2])
    #     parfile = sys.argv[3]
    #     samples = int(sys.argv[4])
    #     frac = float(sys.argv[5])
    #     if len(sys.argv) > 6:
    #         extra_arg = (" ").join(sys.argv[6:])
    #     else:
    #         extra_arg = " -t 0.02"
    # except:
    #     log.info("Usage: %s grains.map colfile.flt.new parameters.par nbr_of_samples sample_size -t 0.01 etc" % (
    #     sys.argv[0]))
    #     log.info("with \t nbr_of_samples: number of bootstrap samples. typically > 500")
    #     log.info("     \t sample_size: fraction of used peaks, float between 0 and 1.")
    #     log.info("     \t If sample_size = 1, resample with replacement.\n")
    #     sys.exit()
    #
    # cmd = "makemap.py -f boot%d_%d.flt -u %d.map -U boot%d_%d.map -p %s %s"
    # cmds = []
    # replacement = False
    # if frac == 1.:
    #     replacement = True
    # gind = range(len(g))
    # # gind = np.arange(2) #only some grains
    # for i in gind:
    #     grain.write_grain_file("%d.map" % i, [g[i]])
    #     c = col.copy()
    #     # c.filter(c.labels==int(g[i].name.split(':')[0]))   #emmm
    #     c.filter(c.labels == i)
    #     np.random.seed()
    #     for sample in range(samples):
    #         d = c.copy()
    #         size = int(round(frac * c.nrows))
    #         d.bigarray = c.bigarray[:, np.random.choice(c.nrows, size, replace=replacement)].copy()
    #         d.nrows = size
    #         d.set_attributes()
    #         d.writefile("boot%d_%d.flt" % (sample, i))
    #         log.info
    #         "sample: %04d \t grain: %03d \t " % (sample, i)
    #         cmds.append(cmd % (sample, i, i, sample, i, parfile, extra_arg))
    #
    # # platform.system() != "Windows"
    # p = multiprocessing.Pool(multiprocessing.cpu_count())
    # p.map(os.system, cmds)
    # p.close()
    # p.join()
    #
    # g_mean = []
    # g_std = []
    # for i in gind:
    #     grains = [grain.read_grain_file("boot%d_%d.map" % (x, i))[0] for x in range(samples)]
    #     ubis = np.array([x.ubi for x in grains])
    #     translations = np.array([x.translation for x in grains])
    #     mean_ubi = np.mean(ubis, axis=0)
    #     std_ubi = np.std(ubis, axis=0)
    #     mean_translation = np.mean(translations, axis=0)
    #     std_translation = np.std(translations, axis=0)
    #     g_mean.append(grain.grain(mean_ubi, mean_translation))
    #     g_std.append(grain.grain(std_ubi, std_translation))
    #     # os.system('rm %d.map'%i)
    #
    # grain.write_grain_file('%s_mean.map' % sys.argv[1][:-4], g_mean)
    # grain.write_grain_file('%s_std.map' % sys.argv[1][:-4], g_std)
    # os.system('rm boot*_*.*')
    grains_in = grain.read_grain_file(input_map)

    log.info(f"I have imported grains from {input_map}")
    log.info(f"I imported {len(grains_in)} grains")

    log.info("These are the grain IDs:")
    log.info([a_grain.name for a_grain in grains_in])

    peaks_in = columnfile.columnfile(peaksfile)

    this_bootstrap_dir = os.path.join(working_dir, f"bootstrap_{phase_name}")
    make_folder(this_bootstrap_dir)

    replacement = False
    if fraction == 1.:
        replacement = True

    maps_for_each_grain_dict = {}

    sample_peaks_output_paths = []
    grain_map_paths = []
    sample_map_output_paths = []
    unindexed_peaks_paths = []
    assigned_peaks_paths = []

    log.info("Generating input files for the makemaps")
    for grain_in in grains_in:
        gid = int(grain_in.name.split(":")[0])
        this_grain_map_path = os.path.join(this_bootstrap_dir, f"grain_{gid}_input.map")
        grain.write_grain_file(this_grain_map_path, [grain_in])
        this_grain_peaks = peaks_in.copy()
        this_grain_peaks.filter(this_grain_peaks.labels == gid)
        np.random.seed()

        maps_for_each_grain_dict[gid] = {}

        for sample in range(samples):
            this_sample_peaks_output_path = os.path.join(this_bootstrap_dir, f"grain_{gid}_sample_{sample}_peaks.flt")
            this_sample_peaks_assigned_path = os.path.join(this_bootstrap_dir, f"grain_{gid}_sample_{sample}_peaks.flt.new")

            this_grain_peaks_sampled = this_grain_peaks.copy()
            n_peaks_sampled = int(round(fraction * this_grain_peaks.nrows))
            this_grain_peaks_sampled.bigarray = this_grain_peaks.bigarray[:, np.random.choice(this_grain_peaks.nrows, n_peaks_sampled, replace=replacement)].copy()
            this_grain_peaks_sampled.nrows = n_peaks_sampled
            this_grain_peaks_sampled.set_attributes()
            this_grain_peaks_sampled.writefile(this_sample_peaks_output_path)

            this_sample_map_output_path = os.path.join(this_bootstrap_dir, f"grain_{gid}_sample_{sample}_output.map")
            maps_for_each_grain_dict[gid][sample] = this_sample_map_output_path

            this_unindexed_peaks_path = os.path.join(this_bootstrap_dir, f"grain_{gid}_sample_{sample}_peaks_unindexed.flt")

            sample_peaks_output_paths.append(this_sample_peaks_output_path)
            grain_map_paths.append(this_grain_map_path)
            sample_map_output_paths.append(this_sample_map_output_path)
            unindexed_peaks_paths.append(this_unindexed_peaks_path)
            assigned_peaks_paths.append(this_sample_peaks_assigned_path)

            log.info(f"grain: {gid} \t sample: {sample} \t")

    if do_parallel:
        log.info("Running makemaps in parallel:")
        ncpu = get_number_of_cores()
        pool = multiprocessing.Pool(ncpu - 1)
        pool.starmap(makemap, tqdm(zip(sample_peaks_output_paths,
                                       repeat(parfile),
                                       grain_map_paths,
                                       repeat(tol),
                                       repeat(omega_slop),
                                       repeat(None),
                                       repeat(True),
                                       unindexed_peaks_paths,
                                       sample_map_output_paths,
                                       repeat(minpks),
                                       repeat(symmetry),
                                       repeat("triclinic"),
                                       repeat(filter_grains)), total=len(sample_peaks_output_paths)), chunksize=1)
    else:
        for peaks_output_path, grain_map_path, unindexed_peaks_path, sample_map_output_path in tqdm(zip(sample_peaks_output_paths, grain_map_paths, unindexed_peaks_paths, sample_map_output_paths), total=len(sample_peaks_output_paths)):
            log.info(f"Peaks in: {peaks_output_path} \t Map in: {grain_map_path} \t Peaks out: {unindexed_peaks_path} \t Map out: {sample_map_output_path}")
            makemap(peaks_output_path, parfile, grain_map_path, tol, omega_slop, None, True, unindexed_peaks_path, sample_map_output_path, minpks, symmetry, "triclinic", filter_grains)

    removed_grain_maps = []

    # Now the makemaps have completed, we should delete the contributory files to save space
    for peaks_output_path, grain_map_path, unindexed_peaks_path, assigned_peaks_path in zip(sample_peaks_output_paths, grain_map_paths, unindexed_peaks_paths, assigned_peaks_paths):
        os.remove(peaks_output_path)
        if grain_map_path not in removed_grain_maps:
            os.remove(grain_map_path)
            removed_grain_maps.append(grain_map_path)
        os.remove(unindexed_peaks_path)
        os.remove(assigned_peaks_path)

    # pool.map(os.system, makemap_commands, chunksize=1)
    # pool.close()
    # pool.join()

    mean_grains = []

    grain_errors_dir = os.path.join(this_bootstrap_dir, "grain_errors")
    make_folder(grain_errors_dir)

    # Get a phase
    phase = Phase.from_id11_pars(name=phase_name, id11_path=parfile)

    log.info("Averaging grains")
    for gid, maps_dict in maps_for_each_grain_dict.items():
        this_grain_error_dir = os.path.join(grain_errors_dir, f"{gid}")
        make_folder(this_grain_error_dir)
        grain_list = [grain.read_grain_file(map_path)[0] for map_path in maps_dict.values()]
        map_list = list(maps_dict.values())

        new_grain, trans_stdev, eps_stdev, eps_lab_stdev, u_stdev, angle_error = bootstrap_mean_grain_and_errors(grain_list, phase)

        # Store the gid of the new grain
        new_grain.name = f"{gid}"

        mean_grains.append(new_grain)

        np.savetxt(os.path.join(this_grain_error_dir, "pos_errors.txt"), trans_stdev)
        np.savetxt(os.path.join(this_grain_error_dir, "eps_errors.txt"), eps_stdev)
        np.savetxt(os.path.join(this_grain_error_dir, "eps_lab_errors.txt"), eps_lab_stdev)
        np.savetxt(os.path.join(this_grain_error_dir, "u_errors.txt"), u_stdev)
        np.savetxt(os.path.join(this_grain_error_dir, "angle_error.txt"), np.array([angle_error]))

        # Remove the contributory maps:
        log.info(f"Deleting maps for grain {gid}")
        for map_path in tqdm(map_list):
            os.remove(map_path)

    temp_output_map_path = os.path.join(working_dir, f"{output_map_filename}_no_labels.map")
    final_output_map_path = os.path.join(working_dir, f"{output_map_filename}.map")

    # Grain GIDs get written to file here
    grain.write_grain_file(temp_output_map_path, mean_grains)
    tthr = (0., 180.)
    func = getattr(ImageD11.refinegrains, "triclinic")
    o = refinegrains_filtered(intensity_tth_range=tthr,
                              latticesymmetry=func,
                              OmFloat=True,
                              OmSlop=omega_slop)


    o.loadparameters(parfile)
    log.info("got pars")
    o.loadfiltered(peaksfile)
    log.info("got filtered")
    # Grain GIDs from file are preserved
    o.readubis_keep_names(temp_output_map_path)

    if symmetry != "triclinic":
        # Grainspotter will have already done this
        log.info("transform to uniq")
        o.makeuniq(symmetry)
    log.info("got ubis")
    o.tolerance = float(tol)
    log.info("generating")
    o.generate_grains()
    o.assignlabels()
    o.savegrains_filtered(final_output_map_path, minpks=minpks, sort_npks=False, filter_grains=filter_grains)


if __name__ == "__main__":
    parser = get_options(argparse.ArgumentParser())
    options = parser.parse_args()

    bootstrap(input_map=options.ubifile,
              parfile=options.parfile,
              peaksfile=options.fltfile,
              working_dir=options.working_dir,
              output_map_filename=options.newubifile,
              symmetry=options.symmetry,
              tol=options.tol,
              samples=options.samples,
              fraction=options.fraction,
              omega_slop=options.omega_slop,
              phase_name=options.phase_name,
              filter_grains=options.filter_grains,
              minpks=options.minpks)