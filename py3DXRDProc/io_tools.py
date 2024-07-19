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

import os
import shutil

import logging
log = logging.getLogger(__name__)


def make_folder(folder):
    """Makes folder if it doesn't exist.

    :param folder: path to folder to create
    """
    if not os.path.exists(folder):
        log.info(f"Creating folder {folder}")
        os.mkdir(folder)


def make_folders(folder):
    """
    Makes folder tree if it doesn't exist.
    :param folder: folder to create
    """
    if not os.path.exists(folder):
        log.info(f"Creating folder {folder}")
        os.makedirs(folder)


def clean_folder(folder):
    """
    Cleans folders.
    :param folder: folder to clean
    """
    if os.listdir(folder):
        log.info(f"{folder} not empty, deleting")
        for root, dirs, files in os.walk(folder):
            for file in files:
                os.remove(os.path.join(root, file))


def clean_folder_fast(folder):
    shutil.rmtree(folder)
    make_folder(folder)
