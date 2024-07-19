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
import json

from jsmin import jsmin

import logging
log = logging.getLogger(__name__)


class Struct(object):
    # This class (https://stackoverflow.com/a/6993694) by xeye (https://stackoverflow.com/users/885564/xeye) is licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


def get_options(the_parser):
    """
    Specify command-line options for this script.
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


def string_to_struct(the_string):
    minified = jsmin(the_string)
    json_obj = json.loads(minified)
    the_pars = Struct(json_obj)
    return the_pars


def parameter_attribute_builder(input_file):
    with open(input_file) as js_file:
        the_string = js_file.read()
    the_pars = string_to_struct(the_string)

    return the_pars


def main():
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    my_parser = get_options(parser)
    options, args = my_parser.parse_known_args()
    pars = parameter_attribute_builder(options.input_file)
    log.info(pars.parameter_files.scan_database)


if __name__ == "__main__":
    main()
