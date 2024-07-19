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
from __future__ import annotations

import logging
import os
from typing import List, Dict, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from ImageD11.finite_strain import DeformationGradientTensor
from numba import set_num_threads

from py3DXRDProc.cluster import get_number_of_cores
from py3DXRDProc.conversions import are_grains_duplicate_array_2lists_numba_wrapper

log = logging.getLogger(__name__)

from py3DXRDProc.grain import RawGrain, CleanGrain, StitchedGrain, filter_grain_list, TrackedGrain, \
    combine_matching_grain_pairs_into_groups
from py3DXRDProc.grain_map import RawGrainsMap, GrainsCollection
from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume, TrackedGrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.parse_input_file import string_to_struct, Struct
from py3DXRDProc.phase import Phase


class Sample:
    """Sample class - contains information about an overall 3DXRD sample.

     :param name: The name of the sample
     :raises TypeError: If `name` isn't a ``str``
     """

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Sample name must be a string!")
        self.pars = None
        self.pars_string = None
        self._load_steps_dict: Dict[str, LoadStep] = {}
        self.name = name
        self.tracked_grain_volume: Optional[TrackedGrainVolume] = None

    @property
    def load_steps(self) -> Dict[str, LoadStep]:
        """Gets all the load steps

        :return: The list of load steps from the dictionary
        """

        return self._load_steps_dict

    @property
    def load_steps_list(self) -> List[LoadStep]:
        """Gets all the load steps

        :return: The list of load steps from the dictionary
        """

        return list(self._load_steps_dict.values())

    @property
    def load_step_names(self) -> List[str]:
        """Gets all the load step names

        :return: The list of load step names from the dictionary
        """

        return list(self._load_steps_dict.keys())

    @property
    def all_unstitched_grain_volumes_list(self) -> List[GrainVolume]:
        """Gets all the :class:`~py3DXRDProc.grain_volume.GrainVolume` from each :class:`~py3DXRDProc.load_step.LoadStep` in :attr:`~.Sample.load_steps_list`

        :return: All :class:`~py3DXRDProc.grain_volume.GrainVolume` in the sample
        """

        all_grain_volumes = []
        for load_step in self.load_steps_list:
            all_grain_volumes.extend(load_step.grain_volumes_list)
        return all_grain_volumes

    @property
    def all_stitched_grain_volumes_list(self) -> List[StitchedGrainVolume]:
        """Gets all the :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` from each :class:`~py3DXRDProc.load_step.LoadStep` in :attr:`~.Sample.load_steps_list`

        :return: All :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in the sample
        """

        all_stitched_grain_volumes = []
        for load_step in self.load_steps_list:
            all_stitched_grain_volumes.extend(load_step.stitched_grain_volumes_list)
        return all_stitched_grain_volumes

    @property
    def all_grain_volumes_list(self) -> List[GrainVolume | StitchedGrainVolume]:
        """Gets all the :class:`~py3DXRDProc.grain_volume.GrainVolume` and :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` from each :class:`~py3DXRDProc.load_step.LoadStep` in :attr:`~.Sample.load_steps_list`

        :return: All :class:`~py3DXRDProc.grain_volume.GrainVolume` and :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in the sample
        """

        all_stitched_grain_volumes = []
        for load_step in self.load_steps_list:
            all_stitched_grain_volumes.extend(load_step.all_grain_volumes_list)
        return all_stitched_grain_volumes

    @property
    def all_raw_grains(self) -> List[RawGrain]:
        """List of all :class:`~py3DXRDProc.grain.RawGrain` from :attr:`~.Sample.load_steps_list`

        :return: List of all :class:`~py3DXRDProc.grain.RawGrain`
        """

        grains_list: List[RawGrain] = []
        for load_step in self.load_steps_list:
            grains_list.extend(load_step.all_raw_grains)
        return grains_list

    @property
    def all_clean_grains(self) -> List[CleanGrain]:
        """List of all :class:`~py3DXRDProc.grain.CleanGrain` from :attr:`~.Sample.load_steps_list`

        :return: List of all :class:`~py3DXRDProc.grain.CleanGrain`
        """

        grains_list: List[CleanGrain] = []
        for load_step in self.load_steps_list:
            grains_list.extend(load_step.all_clean_grains)
        return grains_list

    @property
    def all_raw_and_clean_grains(self) -> List[RawGrain | CleanGrain]:
        """List of all :class:`~py3DXRDProc.grain.RawGrain` and :class:`~py3DXRDProc.grain.CleanGrain` from :attr:`~.Sample.load_steps_list`

        :return: List of all :class:`~py3DXRDProc.grain.CleanGrain`
        """

        grains_list: List[RawGrain | CleanGrain] = []
        for load_step in self.load_steps_list:
            grains_list.extend(load_step.all_raw_and_clean_grains)
        return grains_list

    @property
    def all_stitched_grains(self) -> List[StitchedGrain]:
        """List of all :class:`~py3DXRDProc.grain.StitchedGrain` from :attr:`~.Sample.load_steps_list`

        :return: List of all :class:`~py3DXRDProc.grain.StitchedGrain`
        """

        grains_list: List[StitchedGrain] = []
        for load_step in self.load_steps_list:
            grains_list.extend(load_step.all_stitched_grains)
        return grains_list

    @property
    def all_tracked_grains(self) -> List[TrackedGrain]:
        """List of all :class:`~py3DXRDProc.grain.TrackedGrain` from :attr:`~.Sample.tracked_grain_volume`

        :return: List of all :class:`~py3DXRDProc.grain.TrackedGrain`
        :raises ValueError: If there isn't a :attr:`~.Sample.tracked_grain_volume` yet
        """

        if self.tracked_grain_volume is None:
            raise ValueError("tracked_grain_volume doesn't exist!")
        return self.tracked_grain_volume.all_grains

    @property
    def all_grains(self) -> List[RawGrain | CleanGrain | StitchedGrain | TrackedGrain]:
        """List of all grains from :attr:`~.Sample.load_steps_list`

        :return: List of all grains in the load step
        """

        grains_list: List[RawGrain | CleanGrain | StitchedGrain | TrackedGrain] = []
        for load_step in self.load_steps_list:
            grains_list.extend(load_step.all_grains)
        if self.tracked_grain_volume is not None:
            grains_list.extend(self.all_tracked_grains)
        return grains_list

    @property
    def all_phases(self) -> List[Phase]:
        """List of all phases from :attr:`~.Sample.all_grain_volumes_list` in the sample

        :return: List of all phases in the sample
        """

        all_phases = []
        for grain_volume in self.all_grain_volumes_list:
            for phase in grain_volume.all_phases:
                if phase not in all_phases:
                    all_phases.append(phase)
        return all_phases

    @property
    def all_pos_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = np.concatenate([load_step.all_pos_errors for load_step in self.load_steps_list], axis=0)
        return pos_error_array

    @property
    def all_eps_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = np.concatenate([load_step.all_eps_errors for load_step in self.load_steps_list],
                                         axis=0)
        return eps_error_array

    @property
    def all_eps_lab_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = np.concatenate(
            [load_step.all_eps_lab_errors for load_step in self.load_steps_list],
            axis=0)
        return eps_lab_error_array

    @property
    def all_U_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = np.concatenate([load_step.all_U_errors for load_step in self.load_steps_list],
                                       axis=0)
        return U_error_array

    @property
    def all_angle_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = np.concatenate(
            [load_step.all_angle_errors for load_step in self.load_steps_list], axis=0)
        return angle_error_array

    @property
    def all_sig_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = np.concatenate([load_step.all_sig_errors for load_step in self.load_steps_list],
                                         axis=0)
        return sig_error_array

    @property
    def all_sig_lab_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = np.concatenate(
            [load_step.all_sig_lab_errors for load_step in self.load_steps_list],
            axis=0)
        return sig_lab_error_array

    @property
    def average_pos_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = self.all_pos_errors
        return np.mean(pos_error_array, axis=0)

    @property
    def average_eps_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = self.all_eps_errors
        return np.mean(eps_error_array, axis=0)

    @property
    def average_eps_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = self.all_eps_lab_errors
        return np.mean(eps_lab_error_array, axis=0)

    @property
    def average_U_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = self.all_U_errors
        return np.mean(U_error_array, axis=0)

    @property
    def average_angle_error(self) -> float:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = self.all_angle_errors
        return float(np.mean(angle_error_array, axis=0))

    @property
    def average_sig_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = self.all_sig_errors
        return np.mean(sig_error_array, axis=0)

    @property
    def average_sig_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all raw grains (from :attr:`~.Sample.all_raw_grains`) in the sample

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = self.all_sig_lab_errors
        return np.mean(sig_lab_error_array, axis=0)

    # Fetch methods:

    def get_load_step(self, load_step_name: str) -> LoadStep:
        """Returns the corresponding load step dictionary entry for `load_step_name`

        :param load_step_name: The name to use as the key to the :attr:`~.Sample.load_steps` dictionary
        :raises TypeError: If `load_step_name` isn't a ``str``
        :raises KeyError: If that load step couldn't be found
        :return: The :class:`~py3DXRDProc.load_step.LoadStep` instance
        """

        if not isinstance(load_step_name, str):
            raise TypeError("Load step name should be a string!")
        try:
            return self.load_steps[load_step_name]
        except KeyError:
            raise KeyError(f"Couldn't find a load step with the name {load_step_name} in the dictionary")

    def get_load_steps(self, load_step_name_list: List[str]) -> List[LoadStep]:
        """Returns the corresponding load step dictionary entries for a list of `load_step_name`

        :param load_step_name_list: The name to use as the key to the :attr:`~.Sample.load_steps` dictionary
        :raises TypeError: If `load_step_name_list` isn't a ``list``
        :raises TypeError: If any entry in `load_step_name_list` isn't a ``str``
        :return: The :class:`~py3DXRDProc.load_step.LoadStep` instance
        """

        if not isinstance(load_step_name_list, list):
            raise TypeError("load_step_name_list should be a list!")
        for load_step_name in load_step_name_list:
            if not isinstance(load_step_name, str):
                raise TypeError("load_step_name_list must be a list of strings!")
        load_step_list = [self.get_load_step(load_step_name) for load_step_name in load_step_name_list]
        return load_step_list

    def get_stitched_grain_volume(self, vol_name: str) -> StitchedGrainVolume:
        """Get a :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` from :attr:`~.Sample.all_stitched_grain_volumes_list` given the volume name to look for

        :param vol_name: The name to look for in :attr:`~.Sample.all_stitched_grain_volumes_list`
        :raises KeyError: If a matching :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` couldn't be found
        :return: The matching :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` if one was found
        """

        for vol in self.all_stitched_grain_volumes_list:
            if vol.name == vol_name:
                return vol
        raise KeyError(f"Couldn't find stitched grain volume with name {vol_name}")

    # Add methods:
    def add_load_step(self, load_step: LoadStep) -> None:
        """Adds a :class:`~py3DXRDProc.load_step.LoadStep` to the :attr:`~.Sample.load_steps` dictionary

        :param load_step: The :class:`~py3DXRDProc.load_step.LoadStep` instance to add
        :raises TypeError: If `load_step` isn't a :class:`~py3DXRDProc.load_step.LoadStep` instance
        :raises ValueError: If `load_step` is already in the :attr:`~.Sample.load_steps` dictionary
        """

        if not isinstance(load_step, LoadStep):
            raise TypeError("load_step should be a LoadStep instance!")
        if load_step.name not in self.load_step_names:
            self.load_steps[load_step.name] = load_step
        else:
            raise ValueError(f"Load step {load_step} already in the load step dictionary!")

    def add_load_steps(self, load_step_list: List[LoadStep]):
        """Adds multiple :class:`~py3DXRDProc.load_step.LoadStep` to the :attr:`~.Sample.load_steps` dictionary

        :param load_step_list: The list of :class:`~py3DXRDProc.load_step.LoadStep` instance to add
        :raises TypeError: If `load_step_list` isn't a ``list`` instance
        """

        if not isinstance(load_step_list, list):
            raise TypeError("load_step_list should be a list!")
        for load_step in load_step_list:
            self.add_load_step(load_step)

    # Calculate methods:
    def clean(self, dist_tol: float = 0.1,
              angle_tol: float = 1.0) -> None:
        """Take all the :class:`~py3DXRDProc.load_step.LoadStep` in the :attr:`~.Sample.load_steps` dictionary.
        For each :class:`~py3DXRDProc.load_step.LoadStep`, call :meth:`py3DXRDProc.load_step.LoadStep.clean`

        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        """

        log.info(f"Cleaning")
        for load_step in self.load_steps_list:
            load_step.clean(dist_tol=dist_tol, angle_tol=angle_tol)

    def stitch(self, filter_before_merge: bool = False,
               filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
               dist_tol_xy: float = 0.1,
               dist_tol_z: float = 0.2,
               angle_tol: float = 1.0) -> None:
        """Take all :class:`~py3DXRDProc.load_step.LoadStep` in the :attr:`~.Sample.load_steps` dictionary.
                For each :class:`~py3DXRDProc.load_step.LoadStep`, call :meth:`py3DXRDProc.load_step.LoadStep.stitch`

        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~py3DXRDProc.grain_volume.GrainVolume` before stitching, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol_xy: The tolerance in grain centre-centre XY distance (mm)
        :param dist_tol_z: The tolerance in grain centre-centre Z distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        """

        log.info(f"Stitching")
        for load_step in self.load_steps_list:
            load_step.stitch(filter_before_merge=filter_before_merge,
                             filter_bounds=filter_bounds,
                             dist_tol_xy=dist_tol_xy,
                             dist_tol_z=dist_tol_z,
                             angle_tol=angle_tol)

    def track(self, filter_before_merge: bool = False,
              filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
              dist_tol: float = 0.1,
              angle_tol: float = 1.0) -> None:
        """Generate a :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` from the :attr:`~.Sample.all_stitched_grains_volume` dictionary using meth:`py3DXRDProc.grain_volume.TrackedGrainVolume.from_grainvolume_list`.

:param filter_before_merge: Whether to filter the grains geometrically in each :class:`~.StitchedGrainVolume` before stitching, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises ValueError: If there's only one :class:`~.StitchedGrainVolume` in the :attr:`~.Sample.all_stitched_grains_volume` dictionary
        """

        log.info(f"Tracking")
        if len(self.load_steps_list) == 1:
            raise ValueError("Only one load step, nothing to track!")
        tracked_vol = TrackedGrainVolume.from_grainvolume_list(
            list_of_grain_volumes=self.all_stitched_grain_volumes_list,
            filter_before_merge=filter_before_merge,
            filter_bounds=filter_bounds,
            dist_tol=dist_tol,
            angle_tol=angle_tol)
        self.tracked_grain_volume = tracked_vol

    def guess_affine_transformations(self, affine_trans_dict):
        for key, value in affine_trans_dict.items():
            self.get_load_step(key).D = value

    def optimize_sample_reference_frames(self, phase_name):

        from py3DXRDProc.conversions import rot_array_to_matrix, disorientation_single_numba
        from scipy.optimize import direct
        from pycpd import RigidRegistration

        def evaluate_misorien(rot_array, load_step_2, grain_pairs):
            rot_matrix = rot_array_to_matrix(rot_array)

            # modify the rotation part of the second load step
            # probably better if this didn't modify D directly

            D_rot_local = rot_matrix @ load_step_2.D[0:3, 0:3]

            # evaluate mean misorientation between grain pairs
            misoriens = []
            for grain_pair in grain_pairs:
                grain_b, grain_a = grain_pair

                misorientation = disorientation_single_numba(matrix_tuple=(grain_a.U_sample, D_rot_local @ grain_b.U),
                                                             symmetries=grain_a.phase.symmetry.symmetry_operators())

                misoriens.append(misorientation)

            return np.mean(misoriens)

        # for each grain in first list, find best match in second list
        # position limited, orientation based

        def find_unique_orientation_matches(grains_a, grains_b, dist_tol, angle_tol):
            # modified wrapper around conversions.are_grains_duplicate_array_2lists_numba_wrapper

            ncpu = get_number_of_cores() - 1
            set_num_threads(ncpu)

            matching_grain_pair_indices = are_grains_duplicate_array_2lists_numba_wrapper(grains_a, grains_b,
                                                                                          dist_tol=dist_tol,
                                                                                          angle_tol=angle_tol)
            matching_grain_pairs = [(grains_a[a], grains_b[b]) for (a, b) in matching_grain_pair_indices]
            matching_grain_pairs_deduped = []
            for grain_pair in matching_grain_pairs:
                grain_a, grain_b = grain_pair
                already_in_list = False
                for existing_grain_pair in matching_grain_pairs_deduped:
                    existing_grain_a, existing_grain_b = existing_grain_pair
                    if grain_a == existing_grain_a:
                        already_in_list = True
                if not already_in_list:
                    matching_grain_pairs_deduped.append(grain_pair)

            return matching_grain_pairs_deduped

        # def find_best_orientation_matches(grains_a, grains_b, dist_tol, angle_tol):
        #     matches = []
        #     # b may have already been matched
        #     b_matches = []
        #     for grain_a in tqdm(grains_a):
        #         best_grain_in_b = None
        #         best_angle = np.degrees(np.pi)
        #
        #         for grain_b in grains_b:
        #             # check grain b hasn't already been matched (this ensures 1-to-1 correspondence)
        #             if grain_b not in b_matches:
        #                 if grain_a.phase == grain_b.phase:
        #                     if np.linalg.norm(grain_a.pos_sample - grain_b.pos_sample) < dist_tol:
        #                         misorientation = disorientation_single_numba(
        #                             matrix_tuple=(grain_a.U_sample, grain_b.U_sample),
        #                             symmetries=grain_a.phase.symmetry.symmetry_operators())
        #
        #                         if misorientation < best_angle:
        #                             best_angle = misorientation
        #                             best_grain_in_b = grain_b
        #
        #         if best_angle < angle_tol:
        #             matches.append((grain_a, best_grain_in_b))
        #             b_matches.append(best_grain_in_b)
        #
        #     return matches

        first_load_step = self.load_steps_list[0]

        for second_load_step in self.load_steps_list[1:]:
            log.info(f"Optimizing between load steps {first_load_step.name} and {second_load_step.name}")
            grains_list_1 = first_load_step.all_stitched_grains
            grains_list_2 = second_load_step.all_stitched_grains

            # limit to grains of our phase

            grains_list_1_phased = [grain for grain in grains_list_1 if grain.phase.name == phase_name]
            grains_list_2_phased = [grain for grain in grains_list_2 if grain.phase.name == phase_name]

            # sort the grains lists by volume

            grains_list_1_phased_sorted = sorted(grains_list_1_phased, key=lambda x: x.volume, reverse=True)
            grains_list_2_phased_sorted = sorted(grains_list_2_phased, key=lambda x: x.volume, reverse=True)

            # pick the 100 largest grains from each grains list

            grains_list_1 = grains_list_1_phased_sorted[0:100]
            grains_list_2 = grains_list_2_phased_sorted[0:100]

            log.info("Finding matching grain pairs")

            # uses load_step.D:

            initial_matching_grain_pairs = find_unique_orientation_matches(grains_list_2,
                                                                           grains_list_1,
                                                                           dist_tol=0.1,
                                                                           angle_tol=5.0)

            log.info(f"Initial grain pair search found {len(initial_matching_grain_pairs)} grains")

            bounds = [(-45, 45), (0, 90), (-45, 45)]

            # randomly select 50 pairs to optimize
            # rng = np.random.default_rng()
            # random_pairs = rng.choice(initial_matching_grain_pairs, 50)
            random_pairs = initial_matching_grain_pairs

            log.info("Optimizing rotation using grain orientations")
            res = direct(func=evaluate_misorien,
                         bounds=bounds,
                         args=(second_load_step, random_pairs))
            log.info(f"Optimized rotation found is {res.x}")

            new_rot_matrix = rot_array_to_matrix(res.x)

            # modify D
            new_D = second_load_step.D.copy()
            new_D[0:3, 0:3] = new_rot_matrix @ second_load_step.D[0:3, 0:3]
            second_load_step.D = new_D

            # find new matching pairs
            log.info("Searching again for matches with optimized rotation")
            secondary_matching_grain_pairs = find_unique_orientation_matches(grains_list_2,
                                                                             grains_list_1,
                                                                             dist_tol=0.1,
                                                                             angle_tol=1.0)

            log.info(f"Secondary grain pair search found {len(secondary_matching_grain_pairs)} grains")
            log.info("Optimizing translation using grain positions")

            matched_grain_1_centroids = np.array(
                [grain_1.pos_sample for (grain_1, grain_2) in secondary_matching_grain_pairs])
            matched_grain_2_centroids = np.array(
                [grain_2.pos_sample for (grain_1, grain_2) in secondary_matching_grain_pairs])

            reg = RigidRegistration(X=matched_grain_1_centroids, Y=matched_grain_2_centroids, scale=False)
            TY, (s_reg, R_reg, t_reg) = reg.register()

            log.info(f"Found translation {t_reg}")
            log.info(f"Remaining rotation from positions is {R_reg}")

            second_load_step.D[0:3, 3] = second_load_step.D[0:3, 3] - t_reg

            log.info("Searching again for matches with optimized rotation and translation")
            final_matching_grain_pairs = find_unique_orientation_matches(grains_list_2,
                                                                         grains_list_1,
                                                                         dist_tol=0.1,
                                                                         angle_tol=1.0)

            log.info(f"Final grain pair search found {len(final_matching_grain_pairs)} grains")

    def determine_modified_rotations_from_positions(self):
        """Determine whether load steps are rotated by some rigid body transform compared to the first load step"""
        from pycpd import RigidRegistration

        target_grains = [grain for grain in self.tracked_grain_volume.all_grains if grain.is_fully_tracked]

        first_load_step = self.load_steps_list[0]
        first_load_step_grains = [target_grain.parent_stitch_grains[first_load_step.name] for target_grain in
                                  target_grains]
        first_load_step_points = np.array([grain.pos_offset for grain in first_load_step_grains])

        for load_step in self.load_steps_list[1:]:
            load_step_grains = [target_grain.parent_stitch_grains[load_step.name] for target_grain in target_grains]
            load_step_points = np.array([grain.pos_offset for grain in load_step_grains])
            reg = RigidRegistration(X=load_step_points, Y=first_load_step_points, scale=False)
            TY, (s_reg, R_reg, t_reg) = reg.register()
            load_step.U_adjust = np.linalg.inv(R_reg)

    def determine_modified_rotations_from_U_matrices(self):
        """Determine whether load steps are rotated by some rigid body transform compared to the first load step"""
        rotation_matrix_lists_dict = {n: [] for n in self.load_step_names[1:]}
        target_grains = [grain for grain in self.tracked_grain_volume.all_grains if grain.is_fully_tracked]
        for target_grain in target_grains:
            # for each load step
            initial_grain = target_grain.parent_stitch_grains_list[0]
            for load_step in self.load_steps_list[1:]:
                parent_grain = target_grain.parent_stitch_grains[load_step.name]
                # work out the deformation gradient tensor between this grain and the no load grain
                F = DeformationGradientTensor(parent_grain.UBI, initial_grain.UB)
                # decompose to get the rotation matrix
                V, R, S = F.VRS
                # append to the list at this load step
                rotation_matrix_lists_dict[load_step.name].append(R)

        # https://stackoverflow.com/questions/51517466/what-is-the-correct-way-to-average-several-rotation-matrices

        average_rotation_matrices_dict = {}

        for key, matrix_list in rotation_matrix_lists_dict.items():
            matrix_array = np.array(matrix_list)
            average_rot_matrix = np.mean(matrix_array, axis=0)
            U, D, V = np.linalg.svd(average_rot_matrix)
            Q = U @ V
            average_rotation_matrices_dict[key] = Q

        # Set the adjustments

        for load_step_name, rot_mat in average_rotation_matrices_dict.items():
            self.load_steps[load_step_name].U_adjust = rot_mat

    def filter_all_grains(self, xmin: float,
                          xmax: float,
                          ymin: float,
                          ymax: float,
                          zmin: float,
                          zmax: float,
                          use_adjusted_pos: bool = False) -> List[RawGrain | CleanGrain | StitchedGrain | TrackedGrain]:
        """Filter all grains by positional bounds to remove outlying grains.
        Uses :func:`py3DXRDProc.grain.filter_grain_list`

        :param xmin: Minimum x position of the grain (mm)
        :param xmax: Maximum x position of the grain (mm)
        :param ymin: Minimum y position of the grain (mm)
        :param ymax: Maximum y position of the grain (mm)
        :param zmin: Minimum z position of the grain (mm)
        :param zmax: Maximum z position of the grain (mm)
        :param use_adjusted_pos: Decide whether you filter on the grain position in the map frame (`False`) or sample frame (`True`), defaults to `False`
        :return: A list of grains that survived the filter
        """

        filtered_grains_list = filter_grain_list(grain_list=self.all_grains,
                                                 xmin=xmin,
                                                 xmax=xmax,
                                                 ymin=ymin,
                                                 ymax=ymax,
                                                 zmin=zmin,
                                                 zmax=zmax,
                                                 use_adjusted_pos=use_adjusted_pos)
        return filtered_grains_list

        # Make arrays of groups of grains that survived full tracking

    # Import methods:
    @classmethod
    def from_files_df(cls, pars: Struct,
                      pars_string: str,
                      maps_df: pd.DataFrame,
                      load_steps: List[str],
                      sample_name: str,
                      phase_name: Optional[str] = None,
                      with_errors: bool = False) -> Sample:
        """Imports a :class:`~py3DXRDProc.sample.Sample` from a dataframe that specifies the :class:`~py3DXRDProc.grain_volume.GrainVolume` and :class:`~py3DXRDProc.load_step.LoadStep` order.
        Uses `pars` ``Struct`` containing indexing parameter information to find :mod:`ImageD11` map files and imports them.

        :param pars: Parameters from parsed input file as a ``Struct`` object
        :param pars_string: Parameters from parsed input file as a ``str``
        :param maps_df: Dataframe of scans
        :param load_steps: List of the load step names that you want to import
        :param sample_name: Name of the :class:`~py3DXRDProc.sample.Sample`
        :param phase_name: Import only a specific phase, defaults to `None`
        :param with_errors: Imports grain errors if `True`, defaults to `False`
        :raises TypeError: If `pars` is not a ``Struct``
        :raises TypeError: If `pars_string` is not a ``str``
        :raises TypeError: If `maps_df` is not a Pandas DataFrame
        :raises TypeError: If `sample_name` is not a ``str``
        :raises TypeError: If `phase_name` is supplied but not a ``str``
        :raises ValueError: If `phase_name` couldn't be found in the `pars` ``Struct``
        :return sample_object: The :class:`~py3DXRDProc.sample.Sample` object that the function builds
        """

        if not isinstance(pars, Struct):
            raise TypeError("pars should be a Struct!")
        if not isinstance(pars_string, str):
            raise TypeError("pars_string should be a string!")
        if not isinstance(maps_df, pd.DataFrame):
            raise TypeError("maps_df should be a Pandas DataFrame!")
        if not isinstance(sample_name, str):
            raise TypeError("sample_name should be a string!")
        if phase_name is not None:
            if not isinstance(phase_name, str):
                raise TypeError("phase_name should be a string!")
        # initialise sample object
        sample_object = Sample(sample_name)

        # Associate the parameters
        sample_object.pars = pars
        sample_object.pars_string = pars_string

        # Iterate over the load steps, get the scans in each load step

        # Determine which phases to import
        phase_list = pars.phases.names

        if phase_name is not None:
            # Import a specific phase
            if phase_name not in phase_list:
                raise ValueError("Phase you want to import is not in the input file!")
            else:
                phases_to_import = [phase_name]
        else:
            phases_to_import = phase_list

        # Read the phase information from the ImageD11 parameters
        first_load_step_name = load_steps[0]
        phase_objects = [Phase.from_id11_pars(name=a_phase,
                                              id11_path=getattr(pars.parameter_files.ImageD11, first_load_step_name)[
                                                  index]) for index, a_phase in enumerate(phases_to_import)]

        # Try to populate the stiffness constants for each phase
        for phase_object in phase_objects:
            crystal_structure = phase_object.symmetry.to_string()
            if crystal_structure == "cubic":
                c11 = getattr(pars.stiffness, phase_object.name).c11
                c12 = getattr(pars.stiffness, phase_object.name).c12
                c44 = getattr(pars.stiffness, phase_object.name).c44
                phase_object.stiffness_constants = {"c11": c11, "c12": c12, "c44": c44}
            elif crystal_structure == "hexagonal":
                c11 = getattr(pars.stiffness, phase_object.name).c11
                c12 = getattr(pars.stiffness, phase_object.name).c12
                c13 = getattr(pars.stiffness, phase_object.name).c13
                c33 = getattr(pars.stiffness, phase_object.name).c33
                c44 = getattr(pars.stiffness, phase_object.name).c44
                phase_object.stiffness_constants = {"c11": c11, "c12": c12, "c13": c13, "c33": c33, "c44": c44}

        # The index_dimensions that we used in makemap, to make sure we got every single grain
        # These index_dimensions are in the lab frame.
        # They may be larger than the actual index_dimensions of the sample to accommodate rotated square/rectangular samples
        # e.g if the sample is 1 mm wide, but is mounted off-angle at theta=0deg, then the indexing bounds should be larger
        # We should also filter out grains that are more than a certain amount outside of the index_dimensions
        # In grid_index, units are um, so we divide by 1000 to convert to mm
        index_dimensions = (
            (pars.grid_index.dimensions.x[0] / 1000, pars.grid_index.dimensions.x[1] / 1000),
            (pars.grid_index.dimensions.y[0] / 1000, pars.grid_index.dimensions.y[1] / 1000),
            (pars.grid_index.dimensions.z[0] / 1000, pars.grid_index.dimensions.z[1] / 1000)
        )

        # Get the actual material dimensions in the sample reference frame (used for scaling volumes of individual grains)
        material_dimensions = ((pars.sample.dimensions.x[0], pars.sample.dimensions.x[1]),
                               (pars.sample.dimensions.y[0], pars.sample.dimensions.y[1]),
                               (pars.sample.dimensions.z[0], pars.sample.dimensions.z[1]))

        # Read the motor positions
        log.info("Calculating offsets")
        spreadsheet_path = pars.parameter_files.motor_positions
        # Read motor positions dataframe from file
        motor_positions_df = pd.read_excel(spreadsheet_path, dtype={'scan_number': str}, engine='openpyxl')

        extra_column_names = ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1", "Unnamed: 0.2"]

        # Drop random extra columns
        for column in extra_column_names:
            if column in motor_positions_df:
                motor_positions_df.drop(column, axis=1, inplace=True)

        motor_positions_df.dropna(how='all', inplace=True)

        ### The overall sample reference frame is centered vertically on the sample so the middle of the sample has z=0
        ### The origin of the eventual merged grain volume will be the centre of the no-loaded scans
        # So for an odd number of scans, the origin will the the origin of the middle scan

        # Scan 1
        # Scan 2
        # Scan 3 --- origin
        # Scan 4
        # Scan 5

        # For an even number of scans, the origin will be between the middle two scans:

        # Scan 1
        # Scan 2
        # Scan 3
        #        --- origin
        # Scan 4
        # Scan 5
        # Scan 6

        # So e.g scans 1-3 will have a positive offset (their centre of masses have positive z value)
        # and scans 4-6 will have a negative offset

        def offset(group):
            mid_position = (group["y_pos"].to_list()[0] + group["y_pos"].to_list()[-1]) / 2
            group["new_pos"] = mid_position - group["y_pos"]
            return group

        motor_positions_df = motor_positions_df.groupby("load_step").apply(offset)

        ordered_vols_list = []
        for load_step in maps_df:
            if load_step in load_steps:
                log.info(f"Importing load step {load_step}")
                # Create a load step object with this name and parent sample
                load_step_object = LoadStep(load_step, sample_object)

                # Get the column for this load step
                this_load_volumes = maps_df[load_step]
                # Iterate over the entries in this column
                for vol in this_load_volumes:
                    if not pd.isna(vol):
                        log.info(f"At grain volume number {vol}")
                        # Make a GrainVolume object

                        # Work out what the origin is for this GrainVolume
                        this_offset = motor_positions_df.loc[motor_positions_df.scan_number == vol]["new_pos"].values[0]
                        this_origin = np.array([0, 0, this_offset])

                        grain_volume_object = GrainVolume(name=vol, load_step=load_step_object,
                                                          index_dimensions=index_dimensions,
                                                          material_dimensions=material_dimensions,
                                                          offset_origin=this_origin)

                        ordered_vols_list.append(grain_volume_object.name)
                        map_dir = os.path.join(pars.directories.processing, sample_name, load_step, vol)
                        for phase_object in phase_objects:
                            if with_errors:
                                map_path = os.path.join(map_dir,
                                                        f"all{pars.output_names.grid_output}_{phase_object.name}_mademap_means_assigned.map")

                                errors_folder = os.path.join(map_dir, f"bootstrap_{phase_object.name}", "grain_errors")

                                map_object = RawGrainsMap.import_from_map(map_path=map_path, phase=phase_object,
                                                                          grain_volume=grain_volume_object,
                                                                          errors_folder=errors_folder)
                                if map_object is not None:
                                    grain_volume_object.add_raw_map(map_object)
                            else:
                                map_path = os.path.join(map_dir,
                                                        f"all{pars.output_names.grid_output}_{phase_object.name}_mademap.map")

                                map_object = RawGrainsMap.import_from_map(map_path=map_path, phase=phase_object,
                                                                          grain_volume=grain_volume_object)
                                if map_object is not None:
                                    grain_volume_object.add_raw_map(map_object)

                        load_step_object.add_grain_volume(grain_volume_object)

                # Now this load step is full of grains, work out what transformation you'd have to do to centre this load step horizontally
                load_step_grains = load_step_object.all_raw_and_clean_grains
                GrainsCollection.validate_grains_list(load_step_grains)
                load_step_grain_positions = GrainsCollection.get_attribute_list_from_validated_grain_list(
                    attribute="pos", grains_list=load_step_grains)
                load_step_grains_average_position = np.mean(load_step_grain_positions, axis=0)
                horizontal_centering_translation = np.array(
                    [load_step_grains_average_position[0], load_step_grains_average_position[1], 0])
                load_step_object.centering_translation = horizontal_centering_translation

                sample_object.add_load_step(load_step_object)

        sample_object.ordered_vols_list = ordered_vols_list

        return sample_object

    @classmethod
    def import_from_hdf5(cls, file_path: str,
                         specific_load_step_names: Optional[List[str]] = None) -> Sample:
        """Import a :class:`~.Sample` from an HDF5 file

        :param file_path: Path to the HDF5 file to import
        :param specific_load_step_names: List of names of specific load steps to import
        :raises TypeError: If `file_path` is not a ``str``
        :raises TypeError: If `specific_load_step_names` is not a ``list``
        :raises ValueError: If `specific_load_step_names` is an empty ``list``
        :raises TypeError: If any load step name isn't a ``str``
        :raises ValueError: If any load step name is an empty ``str``
        :return: The :class:`~.Sample` created from the HDF5 group
        """

        if not isinstance(file_path, str):
            raise TypeError("file_path should be a string!")
        if specific_load_step_names is not None:
            if not isinstance(specific_load_step_names, list):
                raise TypeError("specific_load_step_names should be a list!")
            if len(specific_load_step_names) == 0:
                raise ValueError("specific_load_step_names cannot be empty!")
            for load_step_name in specific_load_step_names:
                if not isinstance(load_step_name, str):
                    raise TypeError("Each load step name must be a str!")
                if load_step_name == "":
                    raise ValueError("Each load step name must be a non-empty str!")

        log.info("Importing from HDF5")
        # Make a new Sample object
        with h5py.File(file_path, 'r') as hf:
            sample_group_name = list(hf.keys())[0]
            sample_name = sample_group_name
            sample_object = Sample(sample_name)

            sample_object.pars_string = hf[sample_group_name]["Pars String"][()].decode("utf-8")

            sample_object.pars = string_to_struct(sample_object.pars_string)

            if specific_load_step_names is not None:
                load_step_names = specific_load_step_names
            else:
                load_step_names = list(hf[sample_group_name]["Load Step Names"].asstr())

            load_step_object_list = []
            for load_step_name in load_step_names:
                load_step_group = hf[sample_group_name]["Load Steps"][load_step_name]
                load_step_object = LoadStep.import_from_hdf5_group(load_step_group, sample_object)
                load_step_object_list.append(load_step_object)

            # Add the load steps to the sample object
            sample_object.add_load_steps(load_step_object_list)

            expected_tracked_group_name = f"{sample_name}_tracked"
            if (expected_tracked_group_name in hf[sample_group_name].keys()) and (specific_load_step_names is None):
                log.info(f"Importing tracked grain volume")
                tracked_volume = TrackedGrainVolume.import_from_hdf5_group(
                    hf[sample_group_name][expected_tracked_group_name], sample_object)
                sample_object.tracked_grain_volume = tracked_volume

        return sample_object

    # Export methods:
    def export_to_hdf5(self, file_path: str) -> None:
        """Exports the :class:`~.Sample` from an HDF5 file

        :param file_path: Path to the HDF5 file to export to
        """

        log.info("Exporting to HDF5")
        hf = h5py.File(file_path, 'w')
        this_sample_group = hf.create_group(self.name)

        this_sample_group['Pars String'] = self.pars_string

        load_step_names = self.load_step_names
        dt = h5py.special_dtype(vlen=str)
        dset = this_sample_group.create_dataset("Load Step Names", (len(load_step_names),), dtype=dt)
        dset[:] = [str(value) for value in load_step_names]

        this_load_steps_group = this_sample_group.create_group("Load Steps")

        for load_step in self.load_steps_list:
            log.info(f"Exporting load step {load_step.name}")
            load_step.export_to_hdf5_group(this_load_steps_group)

        if self.tracked_grain_volume is not None:
            log.info(f"Exporting tracked grain volume")
            self.tracked_grain_volume.export_to_hdf5_group(this_sample_group)

        hf.close()

    def export_merged_maps_to_gff(self, dest_dir: str,
                                  use_adjusted_position: bool = True,
                                  filter_too: bool = False,
                                  filter_bounds: Optional[
                                      Tuple[float, float, float, float, float, float]] = None) -> None:
        """Calls :meth:`~.py3DXRDProc.load_step.LoadStep.export_merged_volume_to_gff` for each :class:`~py3DXRDProc.load_step.LoadStep` in :attr:`~.Sample.load_steps_list`

        :param dest_dir: The output directory to export the GFFs to
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        :param filter_too: Whether to filter the grains geometrically in each :class:`~py3DXRDProc.load_step.LoadStep` before exporting, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to before export, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        """

        for load_step in self.load_steps_list:
            log.info(f"Exporting load step {load_step.name} to gff")
            load_step.export_merged_volume_to_gff(output_directory=dest_dir,
                                                  use_adjusted_position=use_adjusted_position,
                                                  filter_too=filter_too,
                                                  filter_bounds=filter_bounds)

    def __str__(self) -> str:
        return str(self.load_steps)
