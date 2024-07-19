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
import shutil

import logging
log = logging.getLogger(__name__)

import pandas as pd
from py3DXRDProc.cluster import write_array_file, dls_submit_cluster_job_simple, bham_submit_cluster_job, dls_submit_cluster_job_slurm, esrf_submit_cluster_job_slurm
from py3DXRDProc.io_tools import make_folder, make_folders, clean_folder, clean_folder_fast
from py3DXRDProc.parse_input_file import parameter_attribute_builder


def get_options(the_parser):
    """
    Specify options for this script with ArgParse
    :param the_parser: argument parser in use
    :return parsed arguments
    """
    the_parser.add_argument(
        "-l",
        "--load_step",
        help="Specify load steps to index from database, e.g -l no_load yield_point",
        action="store",
        dest="loadsteps",
        type=str,
        nargs="+",
    )
    the_parser.add_argument(
        "-c",
        "--clean",
        help="Deletes everything in processing dir",
        action="store",
        dest="clean_processing",
        type=str,
        choices=["Y", "N"],
        default="N",
    )
    the_parser.add_argument(
        "-i",
        "--input",
        help="Input file path",
        dest="input_file",
        type=str,
        required=True,
    )
    return the_parser


def main():
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()

    # Load parameters, file locations etc from input file:

    # Define processing directories:
    pars = parameter_attribute_builder(options.input_file)
    rawdata_root = pars.directories.raw_data
    processing_root = pars.directories.processing

    # Check raw directories exist:
    if not os.path.exists(rawdata_root):
        raise FileNotFoundError("Raw data folder does not exist!")

    # Check processing root exists, and make it if it doesn't:
    make_folders(processing_root)

    # Clean this sample folder if requested:
    sample_name = pars.output_names.sample_name
    sample_dir = os.path.join(processing_root, sample_name)

    if options.clean_processing == "Y":
        log.info("Cleaning output sample folder...")
        if os.path.isdir(sample_dir):
            clean_folder_fast(sample_dir)
        else:
            make_folder(sample_dir)
    else:
        make_folder(sample_dir)

    # Copy input file to the sample folder
    shutil.copy(options.input_file, os.path.join(sample_dir, "input_file.json"))

    # Initialise numeric input files directory:
    index_files_dir = os.path.join(sample_dir, "index_files")

    # Make the folder if it doesn't exist:
    make_folder(index_files_dir)

    # Clean the folder if this is not a fresh processing folder
    if options.clean_processing == "N":
        clean_folder_fast(index_files_dir)

    # Read the scans dataframe from file
    scans_df = pd.read_excel(pars.parameter_files.scan_database, dtype=str)

    # If this is ESRF data, read the dataframe of keys
    # Format is two columns: grain_volume and key
    if pars.collection_facility == "ESRF":
        keys_df = pd.read_excel(pars.parameter_files.keys_database, dtype=str)

    if options.loadsteps is None:
        load_steps_to_index = scans_df.columns.to_list()
    else:
        load_steps_to_index = options.loadsteps

    # Iterate through scans:
    files_index = 1
    # Iterate over each column
    for load_step in scans_df:
        if load_step in load_steps_to_index:
            # Get the column for this load step
            this_load_scans = scans_df[load_step]
            # Iterate over the entries in this column
            for scan in this_load_scans:
                if not pd.isna(scan):
                    log.info(scan)
                    load_step_path = os.path.join(sample_dir, load_step)
                    if os.path.exists(load_step_path):
                        # Load step folder exists
                        if options.clean_processing == "Y":
                            clean_folder(load_step_path)
                    else:
                        make_folder(load_step_path)
                    # Default key
                    key = None
                    if pars.collection_facility == "ESRF":
                        key = keys_df[scan].to_list()[0]
                    write_array_file(load_step, scan, options, index_files_dir, files_index, load_step_path, sample_name, key)

                    files_index += 1

    if pars.processing_facility == "DLS":
        if pars.cluster.use_test_cluster:
            dls_submit_cluster_job_slurm(pars.scripts.index, files_index-1, index_files_dir, pars.cluster.simul_job_limit, pars.directories.cluster_stdio)
        else:
            dls_submit_cluster_job_simple(pars.scripts.index, files_index-1, index_files_dir, pars.cluster.simul_job_limit)
    elif pars.processing_facility == "ESRF":
        esrf_submit_cluster_job_slurm(pars.scripts.index, files_index - 1, index_files_dir,
                                        pars.cluster.simul_job_limit, pars.directories.cluster_stdio)
    else:
        bham_submit_cluster_job(pars.scripts.index, files_index-1, index_files_dir, pars.cluster.simul_job_limit)


if __name__ == "__main__":
    main()