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

from py3DXRDProc.io_tools import make_folder
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
    sample = Sample.import_from_hdf5(input_path, load_steps)

    # # make merged scans output dir
    # merged_dest_dir = os.path.join(sample.pars.directories.processing, sample.name, "gffs")
    # make_folder(merged_dest_dir)
    # sample.export_merged_maps_to_gff(dest_dir=merged_dest_dir, filter_too=False)

    sample.track(filter_before_merge=False, dist_tol=0.10, angle_tol=1.0)

    # Export to HDF5
    hdf5_dir = os.path.join(sample.pars.directories.processing, sample.name, "hdf5")
    hdf5_path = os.path.join(hdf5_dir, "all_scans_tracked.hdf5")
    make_folder(hdf5_dir)
    sample.export_to_hdf5(file_path=hdf5_path)

    exit()


if __name__ == "__main__":
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()

    main(options.input_file, options.loadsteps)
