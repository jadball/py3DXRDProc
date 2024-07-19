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

import numpy as np

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
    sample = Sample.from_files_df(pars, pars_string, scans_df, load_steps, pars.output_names.sample_name,
                                  with_errors=True)

    log.info("Cleaning")
    sample.clean(dist_tol=0.1, angle_tol=1.0)

    log.info("Stitching")
    sample.stitch(dist_tol_xy=0.1, dist_tol_z=0.2, angle_tol=1.0)

    log.info("Aligning reference frames")

    # guess initial affine transformations

    # load_1 is well aligned
    R1_guess = np.eye(3)
    T1_guess = np.array([0, 0, 0])
    D1_guess = np.zeros((4, 4))
    D1_guess[0:3, 0:3] = R1_guess
    D1_guess[0:3, 3] = T1_guess

    # load_2 needs a shift in x
    R2_guess = np.eye(3)
    T2_guess = np.array([-0.04, 0, 0])
    D2_guess = np.zeros((4, 4))
    D2_guess[0:3, 0:3] = R2_guess
    D2_guess[0:3, 3] = T2_guess

    # load_3 needs a shift in x and y
    R3_guess = np.eye(3)
    T3_guess = np.array([-0.1, -0.05, 0])
    D3_guess = np.zeros((4, 4))
    D3_guess[0:3, 0:3] = R3_guess
    D3_guess[0:3, 3] = T3_guess

    # load_4
    R4_guess = np.eye(3)
    T4_guess = np.array([-0.15, -0.05, 0])
    D4_guess = np.zeros((4, 4))
    D4_guess[0:3, 0:3] = R4_guess
    D4_guess[0:3, 3] = T4_guess

    # load_5
    R5_guess = np.eye(3)
    T5_guess = np.array([-0.17, -0.05, 0])
    D5_guess = np.zeros((4, 4))
    D5_guess[0:3, 0:3] = R5_guess
    D5_guess[0:3, 3] = T5_guess

    # load_6
    R6_guess = np.eye(3)
    T6_guess = np.array([-0.18, -0.07, 0])
    D6_guess = np.zeros((4, 4))
    D6_guess[0:3, 0:3] = R6_guess
    D6_guess[0:3, 3] = T6_guess

    # load_7
    R7_guess = np.eye(3)
    T7_guess = np.array([-0.19, -0.07, 0])
    D7_guess = np.zeros((4, 4))
    D7_guess[0:3, 0:3] = R7_guess
    D7_guess[0:3, 3] = T7_guess

    # load_8
    R8_guess = np.eye(3)
    T8_guess = np.array([-0.2, -0.07, 0])
    D8_guess = np.zeros((4, 4))
    D8_guess[0:3, 0:3] = R8_guess
    D8_guess[0:3, 3] = T8_guess

    # load_9
    R9_guess = np.eye(3)
    T9_guess = np.array([-0.21, -0.08, 0])
    D9_guess = np.zeros((4, 4))
    D9_guess[0:3, 0:3] = R9_guess
    D9_guess[0:3, 3] = T9_guess

    # load_10
    R10_guess = np.eye(3)
    T10_guess = np.array([-0.15, -0.03, 0])
    D10_guess = np.zeros((4, 4))
    D10_guess[0:3, 0:3] = R10_guess
    D10_guess[0:3, 3] = T10_guess

    trans_dict = {
        "Load_1": D1_guess,
        "Load_2": D2_guess,
        "Load_3": D3_guess,
        "Load_4": D4_guess,
        "Load_5": D5_guess,
        "Load_6": D6_guess,
        "Load_7": D7_guess,
        "Load_8": D8_guess,
        "Load_9": D9_guess,
        "Load_10": D10_guess,

    }

    sample.guess_affine_transformations(trans_dict)

    sample.optimize_sample_reference_frames("austenite")

    log.info("Tracking")
    sample.track(filter_before_merge=False, dist_tol=0.10, angle_tol=1.0)

    log.info("Exporting")
    # Export to HDF5
    hdf5_dir = os.path.join(pars.directories.processing, sample.name, "hdf5")
    hdf5_path = os.path.join(hdf5_dir, "all_scans_aligned_tracked.hdf5")
    make_folder(hdf5_dir)
    sample.export_to_hdf5(file_path=hdf5_path)

    exit()


if __name__ == "__main__":
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()

    main(options.input_file, options.loadsteps)
