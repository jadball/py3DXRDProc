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

import logging
log = logging.getLogger(__name__)

import pandas as pd
from py3DXRDProc.io_tools import make_folder
from py3DXRDProc.parse_input_file import parameter_attribute_builder
from py3DXRDProc.sample import Sample


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
        "-i",
        "--input",
        help="Input file path",
        dest="input_file",
        type=str,
        required=True,
    )
    return the_parser


def main(input_path, load_steps):
    # Read the scans dataframe from file
    # Define processing directories:
    log.info("Reading pars")
    pars = parameter_attribute_builder(input_path)
    with open(input_path, "r") as pars_file:
        pars_string = pars_file.read()

    log.info("Reading scans dataframe")
    scans_df = pd.read_excel(pars.parameter_files.scan_database, dtype=str, engine='openpyxl')
    scans_df = scans_df[scans_df.columns.drop(list(scans_df.filter(regex='Unnamed:')))]

    if load_steps is None:
        load_steps = scans_df.columns.to_list()

    log.info(f"Load steps are {load_steps}")

    log.info("Importing sample")
    sample = Sample.from_files_df(pars, pars_string, scans_df, load_steps, pars.output_names.sample_name, with_errors=True)

    sample.clean(dist_tol=0.3, angle_tol=2.0)

    sample.stitch(dist_tol_xy=0.212, dist_tol_z=0.212, angle_tol=2.0)

    # Export to HDF5
    hdf5_dir = os.path.join(pars.directories.processing, sample.name, "hdf5")
    hdf5_path = os.path.join(hdf5_dir, "all_scans.hdf5")
    make_folder(hdf5_dir)
    sample.export_to_hdf5(file_path=hdf5_path)

    # make merged scans output dir
    merged_dest_dir = os.path.join(pars.directories.processing, sample.name, "gffs")
    make_folder(merged_dest_dir)
    sample.export_merged_maps_to_gff(dest_dir=merged_dest_dir, filter_too=False)

    exit()


if __name__ == "__main__":
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()

    main(options.input_file, options.loadsteps)
