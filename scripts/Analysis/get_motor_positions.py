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
###

import argparse
import os

import h5py
import pandas as pd
from py3DXRDProc.parse_input_file import parameter_attribute_builder


def get_options(the_parser):
    """
    Specify options for this script with ArgParse
    :param the_parser: argument parser in use
    :return parsed arguments
    """
    the_parser.add_argument(
        "-i",
        "--input",
        help="Input file path",
        dest="input_file",
        type=str,
        required=True,
    )
    return the_parser


def motor_pos_from_h5(filename, target_key, motor_pos_dataset):
    key_split = target_key.split("_")
    if len(key_split) != 1:
        key1, key2 = key_split
        target_key = key1
    h = h5py.File(filename, "r")
    motor_pos = h[target_key][motor_pos_dataset][()]
    return motor_pos


def main():
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()

    # Load parameters, file locations etc from input file:

    # Define processing directories:
    pars = parameter_attribute_builder(options.input_file)

    # Read the scans dataframe from file
    scans_df = pd.read_excel(pars.parameter_files.scan_database, dtype=str)

    # Read the keys dataframe from file
    keys_df = pd.read_excel(pars.parameter_files.keys_database, dtype=str)

    # Make a positions dataframe with the same columns as the scans df
    positions_df = pd.DataFrame(columns=["load_step", "letterbox", "scan_number", "y_pos"])

    # Get the path to the specific h5 file for this grain_volume
    sample_name = pars.output_names.sample_name
    input_sample_dir = os.path.join(pars.directories.raw_data, sample_name)

    # Iterate through scans:

    # Iterate over each column
    for load_step in scans_df:
        # Get the column for this load step
        this_load_scans = scans_df[load_step]
        # Iterate over the entries in this column
        for index, scan in enumerate(this_load_scans):
            if not pd.isna(scan):
                key = keys_df[scan].to_list()[0]
                input_scan_path = os.path.join(input_sample_dir, scan, scan + ".h5")
                motor_pos = motor_pos_from_h5(input_scan_path, key, "instrument/positioners/samtz")

                this_row = {"load_step": load_step, "letterbox": index, "scan_number": scan, "y_pos": motor_pos}
                positions_df = positions_df.append(this_row, verify_integrity=True, ignore_index=True)


    print(scans_df)
    print(positions_df)

    positions_df_dest_path = os.path.join("~/my-rds/Code/UoB_GitLab/input_files/ESRF_Run", f"{sample_name}_motor_positions.xlsx")
    positions_df.to_excel(positions_df_dest_path)



if __name__ == "__main__":
    main()