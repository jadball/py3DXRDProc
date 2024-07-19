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

import os

import fabio
import h5py
import numpy as np
from fabio.fabioimage import fabioimage

import logging
log = logging.getLogger(__name__)


def fso_from_datasets(image_files_list, omega_dset, current_dset):
    # Copyright (C) 2005-2019  Jon Wright
    # Modified from ImageD11/sandbox/hdfscan.py at https://github.com/FABLE-3DXRD/ImageD11/

    assert len(image_files_list) == len(omega_dset)
    assert len(omega_dset) == len(current_dset)
    order = np.argsort(np.array(omega_dset))

    def frm(index):
        header = {'dset_omega': omega_dset[index], 'dset_current': current_dset[index], 'dset_filename': image_files_list[index]}
        f = fabio.open(image_files_list[index])
        f.header = header
        f.currentframe = index
        return f

    #
    yield frm(order[0])  # first
    for i in order:
        yield frm(i)


def fso_from_h5(filename, target_key, image_dataset, omega_dataset):
    # Copyright (C) 2005-2019  Jon Wright
    # Modified from ImageD11/sandbox/hdfscan.py at https://github.com/FABLE-3DXRD/ImageD11/

    log.info(f"Opening image dataset {image_dataset}")
    log.info(f"Opening omega dataset {omega_dataset}")
    h = h5py.File(filename, "r")
    omega = h[target_key][omega_dataset][:] % 360  # forwards grain_volume

    nframes = len(omega)
    data = h[target_key][image_dataset]
    assert data.shape[0] == nframes
    header = {'Omega': omega[0]}
    filename = "%s::%s" % (filename, image_dataset)
    order = np.argsort(omega)

    def frm(index):
        header['Omega'] = omega[index]
        f = fabioimage(data[index], header)
        f.filename = filename
        f.currentframe = index
        return f

    #
    yield frm(order[0])  # first
    for i in order:
        yield frm(i)


def get_dls_fso(pars, scan_number_string=None):
    if pars.images.use_nexus:
        # Read image paths from the nexus file
        nexus_parent_dir = pars.images.nexus.parent_directory
        if pars.images.nexus.in_subfolders:
            nexus_subfolder_name = pars.images.nexus.subfolder_prefix + scan_number_string + pars.images.nexus.subfolder_suffix
            nexus_dir = os.path.join(nexus_parent_dir, nexus_subfolder_name)
        else:
            nexus_dir = nexus_parent_dir
        nexus_path = os.path.join(nexus_dir, pars.images.nexus.file_prefix + scan_number_string + "." + pars.images.nexus.extension)

        nexus_file = h5py.File(nexus_path, "r")
        group = nexus_file[pars.images.nexus.group_path]
        image_dset = group.get(pars.images.nexus.image_dset)
        image_list = [os.path.join(nexus_dir, x.decode("utf-8")) for x in image_dset[..., :]]

        if pars.current.scale:
            # Must also be using nexus
            if not pars.images.use_nexus:
                raise ValueError("Can't correct for beam current without using nexus for the images!")
            group = nexus_file[pars.current.nexus.group_path]
            current_dset = group.get(pars.current.nexus.current_dset)
            current_list = [x for x in current_dset[..., :]]
        else:
            current_list = np.ones(len(image_list)) * 300
        image_prefix = ""
    else:
        # Read image paths from the image directory
        image_list = []
        scan_directory = os.path.join(pars.directories.raw_data, pars.images.folder.prefix + scan_number_string + pars.images.folder.suffix)
        for path in sorted(os.listdir(scan_directory)):
            full_path = os.path.join(scan_directory, path)
            if os.path.isfile(full_path):
                filename, file_ext = os.path.splitext(path)
                if file_ext in ['.gz', '.bz2']:
                    file_ext = os.path.splitext(filename)[1] + file_ext
                for possible_prefix in pars.images.folder.image_prefix:
                    if filename.startswith(possible_prefix):
                        if file_ext == "." + pars.images.folder.image_format:
                            image_list.append(full_path)
                            image_prefix = possible_prefix
        current_list = np.ones(len(image_list)) * 300

    if pars.omegas.use_nexus:
        # Read omegas from the nexus file
        nexus_parent_dir = pars.images.nexus.parent_directory
        if pars.images.nexus.in_subfolders:
            nexus_subfolder_name = pars.images.nexus.subfolder_prefix + scan_number_string + pars.images.nexus.subfolder_suffix
            nexus_dir = os.path.join(nexus_parent_dir, nexus_subfolder_name)
        else:
            nexus_dir = nexus_parent_dir
        nexus_path = os.path.join(nexus_dir, pars.images.nexus.file_prefix + scan_number_string + "." + pars.images.nexus.extension)

        nexus_file = h5py.File(nexus_path, "r")
        group = nexus_file[pars.omegas.nexus.group_path]
        omega_dset = group.get(pars.omegas.nexus.omega_dset)
        omega_list = [x for x in omega_dset[..., :]]
    else:
        omega_list = list(np.arange(pars.omegas.manual.start, pars.num_params.end, pars.num_params.step))

    return fso_from_datasets(image_list, omega_list, current_list), image_prefix