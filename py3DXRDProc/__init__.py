"""
py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
University of Birmingham.

Copyright (C) 2019-2024  James Ball unless otherwise stated

This file is part of py3DXRDProc.

py3DXRDProc is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

py3DXRDProc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = "0.1.0"
__author__ = 'James Ball',
__author_email__ = 'jadball@gmail.com'

__all__ = ['grain', 'grain_map', 'grain_volume', 'load_step', 'sample', 'io_tools', 'conversions', 'parse_input_file', 'cluster', 'Indexing', 'phase']

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)