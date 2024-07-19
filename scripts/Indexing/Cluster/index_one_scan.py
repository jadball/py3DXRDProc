#  py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
#  University of Birmingham.
#
#  Copyright (C) 2019-2024  James Ball
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
import os
import random
import shutil
import sys
import time

import numpy as np
from py3DXRDProc.Indexing import old_peaksearcher
from py3DXRDProc.Indexing.bootstrap_map import bootstrap
from py3DXRDProc.Indexing.clean_flt import clean_flt
from py3DXRDProc.Indexing.correct_module_distortion import correct_flt
from py3DXRDProc.Indexing.custom_gip_queues import grid_index_parallel
from py3DXRDProc.Indexing.filter_peaks import filter_peaks
from py3DXRDProc.Indexing.makemap_module import makemap
from py3DXRDProc.Indexing.merge_flt_module import do_merge
from py3DXRDProc.Indexing.new_peaksearcher import process
from py3DXRDProc.Indexing.ps_dset_builder import get_dls_fso
from py3DXRDProc.cluster import read_array_file, get_number_of_cores
from py3DXRDProc.conversions import ps_hdf5_to_flt
from py3DXRDProc.io_tools import make_folders
from scipy import ndimage

import logging

log = logging.getLogger(__name__)


def main(inp_file_path):
    pars, load_step, load_step_processing_dir, scan_name, sample_name, key = read_array_file(inp_file_path)

    # Work out our sample and grain_volume names
    rawdata_dir = pars.directories.raw_data

    working_root = os.getenv('TMPDIR')
    sample_name = pars.output_names.sample_name
    working_dir = os.path.join(working_root, scan_name)

    # Make the working directory if it isn't already made
    make_folders(working_dir)

    # Change to the working directory
    os.chdir(working_dir)

    # Find out how many CPUs we have
    ncpu = get_number_of_cores()

    log.info("Getting file series object")

    if pars.collection_facility == "DLS":
        fso, image_prefix = get_dls_fso(pars, scan_name)

    peaksearch_output_name = os.path.join(working_dir, pars.output_names.peak_search)
    peaksearch_h5_path = peaksearch_output_name + ".h5"
    peaksearch_flt_path = peaksearch_output_name
    merged_path = os.path.join(working_dir, pars.output_names.merged_peaks + ".flt")

    # Go for peaksearch

    if pars.collection_facility == "ESRF":
        # Single background image, new peaksearcher, spline correction
        background_image_path = pars.data.background
        spline_path = pars.data.spatial_spline
        log.info("Running new peaksearcher")

        # Get the path to the specific h5 file for this grain_volume
        input_sample_dir = os.path.join(pars.directories.raw_data, sample_name)
        input_scan_path = os.path.join(input_sample_dir, scan_name, scan_name + ".h5")

        # Call new peaksearcher
        process(input_scan_path, key, 'measurement/frelon3', 'measurement/diffrz', background_image_path,
                peaksearch_h5_path, ncpu)
        log.info("Convering to flt")
        ps_hdf5_to_flt(peaksearch_h5_path, key, 'peaks3d', merged_path, spline_path)
    else:
        # # Make individual background images
        # # Set up BGMaker options
        # scan_stem = os.path.join(rawdata_dir, pars.images.folder.prefix + volume_name + pars.images.folder.suffix)
        #
        # # Background maker options:
        # bgmaker_options = argparse.Namespace(
        #     ndigits=pars.images.folder.image_ndigits,
        #     last=pars.images.folder.total_images,
        #     format="." + pars.images.folder.image_format,
        #     images=pars.images.background.images_per_bkg,
        #     stem=scan_stem + "/" + image_prefix,
        #     kalman_error=0,
        # )
        #
        # bgmaker_starts = np.linspace(1, pars.images.folder.total_images / pars.images.background.images_per_bkg,
        #                              pars.images.background.num_bkg_images + 1)[:-1]
        #
        # log.info("Generating background...")
        # make_bg(bgmaker_starts, working_dir, bgmaker_options)
        # make_median(os.path.join(working_dir, "bkg0.edf"), "bkg", 0, len(bgmaker_starts), 1, 0, True)
        #
        # background_path = "bkg.edf"

        # Use old peaksearcher

        # Set up peaksearcher options

        parser = argparse.ArgumentParser()
        ps_parser = old_peaksearcher.get_options(parser)
        ps_options, args = ps_parser.parse_known_args()

        scan_stem = os.path.join(rawdata_dir, pars.images.folder.prefix + scan_name + pars.images.folder.suffix)

        if pars.peaksearch.subtract_background:
            setattr(ps_options, "dark", pars.peaksearch.background_path)
        setattr(ps_options, "thresholds", pars.peaksearch.thresholds)
        setattr(ps_options, "out_file", peaksearch_flt_path)
        setattr(ps_options, "stem", scan_stem + "/" + image_prefix)

        if pars.current.scale:
            setattr(ps_options, "monitor_col", "dset_current")
            setattr(ps_options, "monitor_val", pars.current.val)

        log.info("Peaksearching")
        # # Load up the mask
        # mask = np.loadtxt(pars.peaksearch.mask, dtype=bool)
        # # Invert the mask
        # mask_inverted = np.invert(mask)
        import fabio
        mask_inverted = fabio.open(pars.peaksearch.mask).data.astype(bool)

        # Dilate the mask
        mask_inverted_dilated = ndimage.binary_dilation(mask_inverted)
        old_peaksearcher.simple_peaksearcher(ps_options, fso)

        ps_flt_outputs = [f"{peaksearch_flt_path}_t{t}.flt" for t in pars.peaksearch.thresholds]
        ps_flt_gaps_removed = [f"{peaksearch_flt_path}_t{t}_gaps_removed.flt" for t in pars.peaksearch.thresholds]

        for in_flt, out_flt in zip(ps_flt_outputs, ps_flt_gaps_removed):
            filter_peaks(in_flt, out_flt, mask_inverted_dilated)

        log.info("Merging")
        do_merge(getattr(pars.parameter_files.ImageD11, load_step)[0], peaksearch_flt_path, merged_path,
                 pars.peak_merge.spot_merge_tol, pars.peaksearch.thresholds)

        # Distortion correction

        if pars.distortion.correct:
            log.info("Correcting distortion")
            merged_corrected_path = os.path.join(working_dir, pars.output_names.merged_peaks + "_corrected.flt")
            e2dx = pars.distortion.e2dx_path
            e2dy = pars.distortion.e2dy_path
            correct_flt(e2dx, e2dy, merged_path, getattr(pars.parameter_files.ImageD11, load_step)[0],
                        merged_corrected_path)
            merged_path = merged_corrected_path

    ### TEMPORARY

    # output_scan_dir = os.path.join(load_step_processing_dir, volume_name)
    #
    # # Delete the destination grain_volume directory if it exists
    # if os.path.isdir(output_scan_dir):
    #     shutil.rmtree(output_scan_dir)
    # # Copy in the parameters
    # for phase_index in range(pars.phases.total):
    #     phase_name = pars.phases.names[phase_index]
    #     id11_pars = pars.parameter_files.ImageD11[phase_index]
    #
    #     shutil.copyfile(id11_pars, os.path.join(working_dir, f"ID11_pars_{phase_name}.par"))
    #
    # # Copy the working directory to the output grain_volume directory
    # shutil.copytree(working_dir, output_scan_dir)
    #
    #
    # exit()

    ### TEMPORARY

    # SPLIT INTO PHASES AT THIS POINT

    for phase_index in range(pars.phases.total):
        phase_name = pars.phases.names[phase_index]
        keep_rings = pars.clean_peaks.keep_rings[phase_index]
        id11_pars = getattr(pars.parameter_files.ImageD11, load_step)[phase_index]

        shutil.copyfile(id11_pars, os.path.join(working_dir, f"ID11_pars_{phase_name}.par"))

        tmp_out = f"{pars.output_names.grid_output}_{phase_name}"
        clean_output_name = f"{pars.output_names.clean_peaks}_{phase_name}.flt"

        log.info(f"Cleaning peaks for phase {phase_name}")

        # clean_flt(merged_path, id11_pars, keep_rings, clean_output_name, pars.clean_peaks.tth_tol)

        if pars.clean_peaks.use_new_filter[phase_index]:
            from py3DXRDProc.Indexing.clean_flt2 import clean_flt2
            print('Using new peak cleaning routine!')
            clean_flt2(infile=merged_path,
                       parfile=id11_pars,
                       outfile=clean_output_name,
                       frac=pars.clean_peaks.frac[phase_index],
                       dsmax=pars.clean_peaks.dsmax[phase_index],
                       dstol=pars.clean_peaks.dstol[phase_index],
                       rings=keep_rings)

            clean_output_name_allrings = f"{pars.output_names.clean_peaks}_{phase_name}_allrings.flt"
            clean_flt2(infile=merged_path,
                       parfile=id11_pars,
                       outfile=clean_output_name_allrings,
                       frac=pars.clean_peaks.frac[phase_index],
                       dsmax=10.0,
                       dstol=pars.clean_peaks.dstol[phase_index],
                       rings=None)
            merged_path_thisphase = clean_output_name_allrings
        else:
            clean_flt(merged_path, id11_pars, keep_rings, clean_output_name, pars.clean_peaks.tth_tol)
            merged_path_thisphase = merged_path

        # Grid index
        gridpars = {
            'DSTOL': pars.grid_index.dstol[phase_index],
            # accounts for strains at d-spacing of 1 A, and roughly step/distance
            'OMEGAFLOAT': pars.omegas.manual.step / 2.,
            'COSTOL': pars.grid_index.costol[phase_index],
            'HKLTOL': pars.grid_index.hkltol[phase_index],
            'NPKS': pars.grid_index.npks[phase_index],
            'TOLSEQ': pars.grid_index.tolseq[phase_index],
            'SYMMETRY': pars.grid_index.symmetry[phase_index],
            'RING1': pars.grid_index.ring1[phase_index],
            'RING2': pars.grid_index.ring2[phase_index],
            'NUL': pars.grid_index.nul[phase_index],
            'FITPOS': pars.grid_index.fitpos[phase_index],
            'tolangle': pars.grid_index.tolangle[phase_index],
            'toldist': pars.grid_index.toldist[phase_index],
            'NPROC': ncpu,  # guess from cpu_count
            'NTHREAD': pars.grid_index.nthread[phase_index],
        }

        log.info("Running grid index with parameters:")
        log.info(gridpars)

        # grid to search
        translations = [(t_x, t_y, t_z)
                        for t_x in range(pars.grid_index.dimensions.x[0], pars.grid_index.dimensions.x[1],
                                         pars.grid_index.dimensions.x[2])
                        for t_y in range(pars.grid_index.dimensions.y[0], pars.grid_index.dimensions.y[1],
                                         pars.grid_index.dimensions.y[2])
                        for t_z in range(pars.grid_index.dimensions.z[0], pars.grid_index.dimensions.z[1],
                                         pars.grid_index.dimensions.z[2])]

        random.seed(time.time())  # reproducible
        random.shuffle(translations)

        grid_index_parallel(clean_output_name, id11_pars, tmp_out, gridpars, translations)

        input_map_name = f"all{pars.output_names.grid_output}_{phase_name}.map"
        input_map_path = os.path.join(working_dir, input_map_name)

        output_map_name = f"all{pars.output_names.grid_output}_{phase_name}_mademap.map"
        output_map_path = os.path.join(working_dir, output_map_name)

        unindexed_peaks_path = merged_path_thisphase + ".unindexed"

        log.info("Running makemap")
        for tol_index, tol in enumerate(pars.makemap.tolseq[phase_index]):
            if tol_index == 0:
                # First time makemap has been run
                input_map = input_map_path
                output_map = output_map_path
            else:
                input_map = output_map_path
                output_map = output_map_path
            if tol_index == len(pars.makemap.tolseq[phase_index]) - 1:
                # Final tolerance, do our first makemap and filter the grains to remove junk:
                makemap(peaks=merged_path_thisphase,
                        pars=id11_pars,
                        ubis=input_map,
                        tol=tol,
                        omega_slop=gridpars["OMEGAFLOAT"],
                        tth_range=None,
                        sort_npks=False,
                        new_peaks=unindexed_peaks_path,
                        new_ubis=output_map,
                        minpks=pars.makemap.minpeaks[phase_index],
                        symmetry=gridpars["SYMMETRY"],
                        filter_grains=True)
                # Then run bootstrap:
                merged_labelled_peaks_path = merged_path_thisphase + ".new"
                mean_map_assigned_file_name = f"all{pars.output_names.grid_output}_{phase_name}_mademap_means_assigned"

                if pars.bootstrap.quiet[phase_index]:
                    # Change log level to WARN temporarily
                    log.setLevel(logging.WARN)

                bootstrap(input_map=output_map,
                          parfile=id11_pars,
                          peaksfile=merged_labelled_peaks_path,
                          working_dir=working_dir,
                          output_map_filename=mean_map_assigned_file_name,
                          symmetry=gridpars["SYMMETRY"],
                          tol=tol,
                          samples=pars.bootstrap.samples[phase_index],
                          fraction=pars.bootstrap.fraction[phase_index],
                          omega_slop=gridpars["OMEGAFLOAT"],
                          phase_name=phase_name,
                          filter_grains=False,
                          minpks=pars.makemap.minpeaks[phase_index],
                          do_parallel=pars.bootstrap.parallel_makemap[phase_index])

                if pars.bootstrap.quiet[phase_index]:
                    log.setLevel(logging.INFO)

            else:
                makemap(peaks=merged_path_thisphase,
                        pars=id11_pars,
                        ubis=input_map,
                        tol=tol,
                        omega_slop=gridpars["OMEGAFLOAT"],
                        tth_range=None,
                        sort_npks=False,
                        new_peaks=unindexed_peaks_path,
                        new_ubis=output_map,
                        minpks=pars.makemap.minpeaks[phase_index],
                        symmetry=gridpars["SYMMETRY"],
                        filter_grains=False)

        log.info(f"Finished indexing phase {phase_name}")

    output_scan_dir = os.path.join(load_step_processing_dir, scan_name)

    # Delete the destination grain_volume directory if it exists
    if os.path.isdir(output_scan_dir):
        shutil.rmtree(output_scan_dir)

    # Copy the working directory to the output grain_volume directory
    shutil.copytree(working_dir, output_scan_dir)

    exit()


if __name__ == "__main__":
    # Call get_options function to get arguments:
    inp_file_path = sys.argv[1]
    main(inp_file_path)
