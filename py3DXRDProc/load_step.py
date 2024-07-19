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

import os

from typing import List, Dict, TYPE_CHECKING, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt

import logging
log = logging.getLogger(__name__)

from py3DXRDProc.grain import filter_grain_list, StitchedGrain, RawGrain, CleanGrain
from py3DXRDProc.grain_map import RawGrainsMap, StitchedGrainsMap, GrainsCollection, CleanedGrainsMap
from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume
from py3DXRDProc.phase import Phase

if TYPE_CHECKING:
    from py3DXRDProc.sample import Sample


class LoadStep:
    """LoadStep class - contains information about a single load step in a 3DXRD scan.

    :param name: The name of the load step
    :param sample: The :class:`~py3DXRDProc.sample.Sample` instance for this load step
    :raises TypeError: If `name` isn't a ``str``
    :raises ValueError: If `name` is empty
    :raises TypeError: If `sample` isnt' a :class:`~py3DXRDProc.sample.Sample` instance
    """

    def __init__(self, name: str, sample: Sample) -> None:

        if not isinstance(name, str):
            raise TypeError("Load Step should be a string!")
        if name == "":
            raise ValueError("Load Step name cannot be empty!")
        self._name = name
        self._grain_vols_dict: Dict[str, GrainVolume] = {}
        self._stitched_grain_vols_dict: Dict[str, StitchedGrainVolume] = {}

        if not sample.__class__.__name__ == "Sample":
            raise TypeError("Sample should be a Sample instance!")
        self._sample = sample

        # specify D as identity for now
        self.D = np.zeros((4, 4))
        self.D[0:3, 0:3] = np.eye(3)

    @property
    def name(self) -> str:
        """The load step name

        :return: The load step name
        """

        return self._name

    @property
    def sample(self):
        """The load step :class:`~py3DXRDProc.sample.Sample`

        :return: The load step :class:`~py3DXRDProc.sample.Sample`
        """

        return self._sample

    @property
    def grain_volumes(self) -> Dict[str, GrainVolume]:
        """The dictionary of :class:`~py3DXRDProc.grain_volume.GrainVolume`
        inside this load step, indexed by grain volume name

        :return: The :class:`~py3DXRDProc.grain_volume.GrainVolume` dictionary
        """

        return self._grain_vols_dict

    @property
    def grain_volume_names(self) -> List[str]:
        """The grain volume names from the :attr:`~.LoadStep.grain_volumes` dictionary

        :return: List of names of all :class:`~py3DXRDProc.grain_volume.GrainVolume` in the load step
        """

        return list(self._grain_vols_dict.keys())

    @property
    def grain_volumes_list(self) -> List[GrainVolume]:
        """The grain volumes from the :attr:`~.LoadStep.grain_volumes` dictionary

        :return: List of all :class:`~py3DXRDProc.grain_volume.GrainVolume` in the load step
        """

        return list(self._grain_vols_dict.values())

    @property
    def stitched_grain_volumes(self) -> Dict[str, StitchedGrainVolume]:
        """The dictionary of :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume`
        inside this load step, indexed by grain volume name

        :return: The :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` dictionary
        """

        return self._stitched_grain_vols_dict

    @property
    def stitched_grain_volume_names(self) -> List[str]:
        """The grain volume names from the :attr:`~.LoadStep.stitched_grain_volumes` dictionary

        :return: List of names of all :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in the load step
        """

        return list(self._stitched_grain_vols_dict.keys())

    @property
    def stitched_grain_volumes_list(self) -> List[StitchedGrainVolume]:
        """The grain volumes from the :attr:`~.LoadStep.stitched_grain_volumes` dictionary

        :return: List of all :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in the load step
        """

        return list(self._stitched_grain_vols_dict.values())

    @property
    def all_grain_volumes(self) -> Dict[str, GrainVolume] | Dict[str, StitchedGrainVolume]:
        """The dictionary of all grain volumes inside this grain volume, indexed by map name.
        Generated on the fly from :attr:`~.LoadStep.grain_volumes` and :attr:`~.LoadStep.stitched_grain_volumes`

        :return: Grain volumes dictionary
        """

        return {**self.grain_volumes, **self.stitched_grain_volumes}

    @property
    def all_grain_volume_names(self) -> List[str]:
        """List of all grain volume names inside this load step.
        Generated on the fly from :attr:`~.LoadStep.all_grain_volumes`

        :return: List of names of all grain volumes in the load step
        """

        return list(self.all_grain_volumes.keys())

    @property
    def all_grain_volumes_list(self) -> List[GrainVolume | StitchedGrainVolume]:
        """List of all grain volumes inside this load step.
        Generated on the fly from :attr:`~.LoadStep.all_grain_volumes`

        :return: List of all grain maps in the volume
        """

        return list(self.all_grain_volumes.values())

    @property
    def all_raw_grains(self) -> List[RawGrain]:
        """List of all :class:`~py3DXRDProc.grain.RawGrain` from :attr:`~.LoadStep.grain_volumes`

        :return: List of all :class:`~py3DXRDProc.grain.RawGrain`
        """

        grains_list: List[RawGrain] = []
        for grain_volume in self.grain_volumes_list:
            grains_list.extend(grain_volume.all_raw_grains)
        return grains_list

    @property
    def all_clean_grains(self) -> List[CleanGrain]:
        """List of all :class:`~py3DXRDProc.grain.CleanGrain` from :attr:`~.LoadStep.grain_volumes`

        :return: List of all :class:`~py3DXRDProc.grain.CleanGrain`
        """

        grains_list: List[CleanGrain] = []
        for grain_volume in self.grain_volumes_list:
            grains_list.extend(grain_volume.all_clean_grains)
        return grains_list

    @property
    def all_raw_and_clean_grains(self) -> List[RawGrain | CleanGrain]:
        """List of all :class:`~py3DXRDProc.grain.RawGrain` and :class:`~py3DXRDProc.grain.CleanGrain` from :attr:`~.LoadStep.grain_volumes`

        :return: List of all :class:`~py3DXRDProc.grain.CleanGrain`
        """

        grains_list: List[RawGrain | CleanGrain] = []
        for grain_volume in self.grain_volumes_list:
            grains_list.extend(grain_volume.all_grains)
        return grains_list

    @property
    def all_stitched_grains(self) -> List[StitchedGrain]:
        """List of all :class:`~py3DXRDProc.grain.StitchedGrain` from :attr:`~.LoadStep.stitched_grain_volumes`

        :return: List of all :class:`~py3DXRDProc.grain.StitchedGrain`
        """

        grains_list: List[StitchedGrain] = []
        for grain_volume in self.stitched_grain_volumes_list:
            grains_list.extend(grain_volume.all_grains)
        return grains_list

    @property
    def all_grains(self) -> List[RawGrain | CleanGrain | StitchedGrain]:
        """List of all grains from :attr:`~.LoadStep.all_grain_volumes`

        :return: List of all grains in the load step
        """

        grains_list: List[RawGrain | CleanGrain | StitchedGrain] = []
        for grain_volume in self.all_grain_volumes_list:
            grains_list.extend(grain_volume.all_grains)
        return grains_list

    @property
    def all_raw_grain_maps(self) -> List[RawGrainsMap]:
        """List of all :class:`~py3DXRDProc.grain.RawGrainsMap` from :attr:`~.LoadStep.grain_volumes_list`

        :return: List of all :class:`~py3DXRDProc.grain_map.RawGrainsMap`
        """

        raw_grain_maps_list: List[RawGrainsMap] = []
        for grain_volume in self.grain_volumes_list:
            for raw_grain_map in grain_volume.raw_maps_list:
                raw_grain_maps_list.append(raw_grain_map)
        return raw_grain_maps_list

    @property
    def all_clean_grain_maps(self) -> List[CleanedGrainsMap]:
        """List of all :class:`~py3DXRDProc.grain.CleanedGrainsMap` from :attr:`~.LoadStep.grain_volumes_list`

        :return: List of all :class:`~py3DXRDProc.grain_map.CleanedGrainsMap`
        """

        clean_grain_maps_list: List[CleanedGrainsMap] = []
        for grain_volume in self.grain_volumes_list:
            for clean_grain_map in grain_volume.clean_maps_list:
                clean_grain_maps_list.append(clean_grain_map)
        return clean_grain_maps_list

    @property
    def all_stitched_grain_maps(self) -> List[StitchedGrainsMap]:
        """List of all :class:`~py3DXRDProc.grain.StitchedGrainsMap` from :attr:`~.LoadStep.grain_volumes_list`

        :return: List of all :class:`~py3DXRDProc.grain_map.StitchedGrainsMap`
        """

        stitch_grain_maps_list: List[StitchedGrainsMap] = []
        for stitched_grain_volume in self.stitched_grain_volumes_list:
            for stitched_grain_map in stitched_grain_volume.maps_list:
                stitch_grain_maps_list.append(stitched_grain_map)
        return stitch_grain_maps_list

    @property
    def all_grain_maps(self) -> List[RawGrainsMap | CleanedGrainsMap | StitchedGrainsMap]:
        """List of all grain maps in the load step

        :return: List of all grain maps in the load step
        """

        return self.all_raw_grain_maps + self.all_clean_grain_maps + self.all_stitched_grain_maps

    @property
    def all_phases(self) -> List[Phase]:
        """List of all phases from :attr:`~.LoadStep.grain_volumes_list` in the load step

        :return: List of all phases in the load step
        """

        phase_list = []
        for volume in self.all_grain_volumes_list:
            for phase in volume.all_phases:
                if phase not in phase_list:
                    phase_list.append(phase)
        return phase_list

    @property
    def all_pos_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = np.concatenate([raw_grain_map.all_pos_errors for raw_grain_map in self.all_raw_grain_maps], axis=0)
        return pos_error_array

    @property
    def all_eps_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = np.concatenate([raw_grain_map.all_eps_errors for raw_grain_map in self.all_raw_grain_maps], axis=0)
        return eps_error_array

    @property
    def all_eps_lab_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = np.concatenate(
            [raw_grain_map.all_eps_lab_errors for raw_grain_map in self.all_raw_grain_maps],
            axis=0)
        return eps_lab_error_array

    @property
    def all_U_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = np.concatenate([raw_grain_map.all_U_errors for raw_grain_map in self.all_raw_grain_maps],
                                       axis=0)
        return U_error_array

    @property
    def all_angle_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = np.concatenate(
            [raw_grain_map.all_angle_errors for raw_grain_map in self.all_raw_grain_maps], axis=0)
        return angle_error_array

    @property
    def all_sig_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = np.concatenate([raw_grain_map.all_sig_errors for raw_grain_map in self.all_raw_grain_maps],
                                         axis=0)
        return sig_error_array

    @property
    def all_sig_lab_errors(self) -> npt.NDArray[np.float64]:
        """Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Numpy array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = np.concatenate(
            [raw_grain_map.all_sig_lab_errors for raw_grain_map in self.all_raw_grain_maps],
            axis=0)
        return sig_lab_error_array

    @property
    def average_pos_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = self.all_pos_errors
        return np.mean(pos_error_array, axis=0)

    @property
    def average_eps_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = self.all_eps_errors
        return np.mean(eps_error_array, axis=0)

    @property
    def average_eps_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = self.all_eps_lab_errors
        return np.mean(eps_lab_error_array, axis=0)

    @property
    def average_U_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = self.all_U_errors
        return np.mean(U_error_array, axis=0)

    @property
    def average_angle_error(self) -> float:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = self.all_angle_errors
        return float(np.mean(angle_error_array, axis=0))

    @property
    def average_sig_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = self.all_sig_errors
        return np.mean(sig_error_array, axis=0)

    @property
    def average_sig_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all raw grains (from :attr:`~.LoadStep.all_raw_grains`) in the load step

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = self.all_sig_lab_errors
        return np.mean(sig_lab_error_array, axis=0)

    # Fetch methods:
    def get_grain_volume(self, vol_name: str) -> GrainVolume:
        """Get a grain volume in :attr:`~.LoadStep.grain_volumes` by the volume name

        :param vol_name: The name of the grain volume to look for
        :raises TypeError: If `vol_name` isn't a ``str``
        :return: The :class:`~py3DXRDProc.grain_volume.GrainVolume` in :attr:`~.LoadStep.grain_volumes`
        """

        if not isinstance(vol_name, str):
            raise TypeError("vol_name name should be a string!")
        try:
            return self.grain_volumes[vol_name]
        except KeyError:
            raise KeyError(f"Could not find grain volume {vol_name} in vols dict")

    def get_stitched_grain_volume(self, vol_name: str) -> StitchedGrainVolume:
        """Get a grain volume in :attr:`~.LoadStep.stitched_grain_volumes` by the volume name

        :param vol_name: The name of the grain volume to look for
        :raises TypeError: If `vol_name` isn't a ``str``
        :return: The :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in :attr:`~.LoadStep.stitched_grain_volumes`
        """

        if not isinstance(vol_name, str):
            raise TypeError("vol_name name should be a string!")
        try:
            return self.stitched_grain_volumes[vol_name]
        except KeyError:
            raise KeyError(f"Could not find grain volume {vol_name} in vols dict")

    def get_any_grain_volume(self, vol_name: str) -> GrainVolume | StitchedGrainVolume:
        """Get a grain volume in :attr:`~.LoadStep.all_grain_volumes` by the volume name

        :param vol_name: The name of the grain volume to look for
        :raises TypeError: If `vol_name` isn't a ``str``
        :return: The :class:`~py3DXRDProc.grain_volume.GrainVolume` or :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` in :attr:`~.LoadStep.all_grain_volumes`
        """

        if not isinstance(vol_name, str):
            raise TypeError("vol_name name should be a string!")
        try:
            return self.all_grain_volumes[vol_name]
        except KeyError:
            raise KeyError(f"Could not find grain volume {vol_name} in vols dict")

    def get_all_grains_in_volume_name(self, volume_name) -> List[RawGrain | CleanGrain] | List[StitchedGrain]:
        """Get all grains in a volume given a volume name

        :param volume_name: The name of the grain volume to look for
        :return: The list of :class:`~py3DXRDProc.grain.RawGrain` or :class:`~py3DXRDProc.grain.CleanGrain` or :class:`~py3DXRDProc.grain.StitchedGrain` found
        """

        grain_volume = self.get_any_grain_volume(volume_name)
        volume_grains = grain_volume.all_grains
        return volume_grains

    def get_all_raw_grain_maps_with_phase(self, phase: Phase) -> List[RawGrainsMap]:
        """Get all raw grain maps from :attr:`~.LoadStep.all_raw_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain_map.RawGrainsMap` that all have the matching `phase`
        """

        matching_grain_maps = [grain_map for grain_map in self.all_raw_grain_maps if grain_map.phase == phase]
        return matching_grain_maps

    def get_all_clean_grain_maps_with_phase(self, phase: Phase) -> List[CleanedGrainsMap]:
        """Get all clean grain maps from :attr:`~.LoadStep.all_clean_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` that all have the matching `phase`
        """

        matching_grain_maps = [grain_map for grain_map in self.all_clean_grain_maps if grain_map.phase == phase]
        return matching_grain_maps

    def get_all_stitched_grain_maps_with_phase(self, phase: Phase) -> List[StitchedGrainsMap]:
        """Get all stitched grain maps from :attr:`~.LoadStep.all_stitched_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` that all have the matching `phase`
        """

        matching_grain_maps = [grain_map for grain_map in self.all_stitched_grain_maps if grain_map.phase == phase]
        return matching_grain_maps

    def get_all_grain_maps_with_phase(self, phase: Phase) -> List[RawGrainsMap | CleanedGrainsMap | StitchedGrainsMap]:
        """Get all grain maps from :attr:`~.LoadStep.all_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain_map.RawGrainsMap`, :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` and :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` that all have the matching `phase`
        """

        matching_grain_maps = [grain_map for grain_map in self.all_grain_maps if grain_map.phase == phase]
        return matching_grain_maps

    def get_all_raw_grains_with_phase(self, phase) -> List[RawGrain]:
        """Get all raw grains from :attr:`~.LoadStep.all_raw_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain.RawGrain` that all have the matching `phase`
        """

        raw_grains_list = []
        for raw_grain_map in self.get_all_raw_grain_maps_with_phase(phase):
            raw_grains_list.extend(raw_grain_map.grains)
        return raw_grains_list

    def get_all_clean_grains_with_phase(self, phase) -> List[CleanGrain]:
        """Get all clean grains from :attr:`~.LoadStep.all_clean_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain.CleanGrain` that all have the matching `phase`
        """

        clean_grains_list = []
        for clean_grain_map in self.get_all_clean_grain_maps_with_phase(phase):
            clean_grains_list.extend(clean_grain_map.grains)
        return clean_grains_list

    def get_all_stitched_grains_with_phase(self, phase) -> List[StitchedGrain]:
        """Get all stitch grains from :attr:`~.LoadStep.all_stitch_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain.StitchedGrain` that all have the matching `phase`
        """

        stitch_grains_list = []
        for stitch_grain_map in self.get_all_stitched_grain_maps_with_phase(phase):
            stitch_grains_list.extend(stitch_grain_map.grains)
        return stitch_grains_list

    def get_all_grains_with_phase(self, phase) -> List[RawGrain | CleanGrain | StitchedGrain]:
        """Get all grains from :attr:`~.LoadStep.all_grain_maps` that have the right `phase`

        :param phase: The `phase` to look for
        :return: A list of :class:`~py3DXRDProc.grain.RawGrain`, :class:`~py3DXRDProc.grain.CleanGrain` and :class:`~py3DXRDProc.grain.StitchedGrain` that all have the matching `phase`
        """

        grains_list: List[RawGrain | CleanGrain | StitchedGrain] = []
        for grain_map in self.get_all_grain_maps_with_phase(phase):
            grains_list.extend(grain_map.grains)
        return grains_list

    def filter_all_grains(self, xmin: float,
                          xmax: float,
                          ymin: float,
                          ymax: float,
                          zmin: float,
                          zmax: float,
                          use_adjusted_pos: bool = False) -> List[RawGrain | CleanGrain | StitchedGrain]:
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

    # Add methods:
    def add_grain_volume(self, grain_volume: GrainVolume) -> None:
        """Add a :class:`~py3DXRDProc.grain_volume.GrainVolume` to the :attr:`~.LoadStep.grain_volumes` dictionary, indexed by the :attr:`~py3DXRDProc.grain_volume.GrainVolume.name`

        :param grain_volume: The :class:`~py3DXRDProc.grain_volume.GrainVolume` to add
        :raises TypeError: If `grain_volume` isn't a :class:`~py3DXRDProc.grain_volume.GrainVolume` instance
        :raises ValueError: If `grain_volume` was already found in :attr:`~.LoadStep.grain_volumes`
        """

        if not isinstance(grain_volume, GrainVolume):
            raise TypeError("grain_volume should be a GrainVolume instance!")
        if grain_volume.name not in self.grain_volumes:
            self.grain_volumes[grain_volume.name] = grain_volume
        else:
            raise ValueError(f"GrainVolume {grain_volume} already exists in this load step!")

    def add_grain_volumes(self, list_of_grain_volumes: List[GrainVolume]) -> None:
        """Add a list of :class:`~py3DXRDProc.grain_volume.GrainVolume` to the :attr:`~.LoadStep.grain_volumes` dictionary, indexed by the :attr:`~py3DXRDProc.grain_volume.GrainVolume.name`

        :param list_of_grain_volumes: The list of :class:`~py3DXRDProc.grain_volume.GrainVolume` to add
        :raises TypeError: If `list_of_grain_volumes` isn't a ``list`` instance
        """

        if not isinstance(list_of_grain_volumes, list):
            raise TypeError("list_of_grain_volumes should be a list!")
        for vol in list_of_grain_volumes:
            self.add_grain_volume(vol)

    def add_stitched_grain_volume(self, stitched_grain_volume: StitchedGrainVolume) -> None:
        """Add a list of :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` to the :attr:`~.LoadStep.stitched_grain_volumes` dictionary, indexed by the :attr:`~py3DXRDProc.grain_volume.StitchedGrainVolume.name`

        :param stitched_grain_volume: The :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` to add
        :raises TypeError: If `stitched_grain_volume` isn't a :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` instance
        :raises ValueError: If `stitched_grain_volume` was already found in :attr:`~.LoadStep.stitched_grain_volumes`
        """

        if not isinstance(stitched_grain_volume, StitchedGrainVolume):
            raise TypeError("stitched_grain_volume should be a StitchedGrainVolume instance!")
        if stitched_grain_volume.name not in self.stitched_grain_volumes:
            self.stitched_grain_volumes[stitched_grain_volume.name] = stitched_grain_volume
        else:
            raise ValueError(f"StitchedGrainVolume {stitched_grain_volume} already exists in this load step!")

    def add_stitched_grain_volumes(self, list_of_stitched_grain_volumes: List[StitchedGrainVolume]) -> None:
        """Add a list of :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` to the :attr:`~.LoadStep.stitched_grain_volumes` dictionary, indexed by the :attr:`~py3DXRDProc.grain_volume.StitchedGrainVolume.name`

        :param list_of_stitched_grain_volumes: The :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` to add
        :raises TypeError: If `list_of_stitched_grain_volumes` isn't a ``list`` instance
        """

        if not isinstance(list_of_stitched_grain_volumes, list):
            raise TypeError("list_of_stitched_grain_volumes should be a list!")
        for vol in list_of_stitched_grain_volumes:
            self.add_stitched_grain_volume(vol)

    # Calculate methods:

    def clean(self, dist_tol: float = 0.1,
              angle_tol: float = 1) -> None:
        """Take all the :class:`~py3DXRDProc.grain_volume.GrainVolume` in the :attr:`~.LoadStep.grain_volumes` dictionary.
        For each :class:`~py3DXRDProc.grain_volume.GrainVolume`, generate a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from each :class:`~py3DXRDProc.grain_map.RawGrainsMap`.

        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        """

        for vol in self.grain_volumes_list:
            vol.clean(dist_tol=dist_tol,
                      angle_tol=angle_tol)

    def stitch(self, filter_before_merge: bool = False,
               filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
               dist_tol_xy: float = 0.1,
               dist_tol_z: float = 0.2,
               angle_tol: float = 1) -> None:
        """Take all :class:`~py3DXRDProc.grain_volume.GrainVolume` from :attr:`~.LoadStep.grain_volumes_list`,
        stitch them together using :func:`~py3DXRDProc.grain_volume.StitchedGrainVolume.from_grainvolume_list`,
        add it to the :attr:`~.LoadStep.stitched_grain_volumes` dictionary

        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~py3DXRDProc.grain_volume.GrainVolume` before stitching, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol_xy: The tolerance in grain centre-centre XY distance (mm)
        :param dist_tol_z: The tolerance in grain centre-centre Z distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        """

        all_raw_volumes = self.grain_volumes_list
        stiched_grain_volume = StitchedGrainVolume.from_grainvolume_list(list_of_grain_volumes=all_raw_volumes,
                                                                         filter_before_merge=filter_before_merge,
                                                                         filter_bounds=filter_bounds,
                                                                         dist_tol_xy=dist_tol_xy,
                                                                         dist_tol_z=dist_tol_z,
                                                                         angle_tol=angle_tol)
        self.add_stitched_grain_volume(stiched_grain_volume)

    # IO methods:

    @classmethod
    def import_from_hdf5_group(cls, load_step_group: h5py.Group, sample_object: Sample) -> LoadStep:
        """Import a :class:`~.LoadStep` from a :class:`h5py.Group`.
        Imports the underlying grain volumes too.

        :param load_step_group: :class:`h5py.Group` containing the load step data
        :param sample_object: :class:`~py3DXRDProc.sample.Sample` object that this load step belongs to
        :raises TypeError: If `load_step_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `sample_object` is not a :class:`~py3DXRDProc.sample.Sample` instance

        :return: The :class:`~.LoadStep` created from the HDF5 group
        """

        if not isinstance(load_step_group, h5py.Group):
            raise TypeError("load_step_group should be an h5py.Group instance")
        if not sample_object.__class__.__name__ == "Sample":
            raise TypeError("sample_object should be a LoadStep instance!")

        load_step_name = load_step_group.name.split("/")[-1]
        logging.info(f"Importing load step {load_step_name}")
        load_step_object = LoadStep(name=load_step_name, sample=sample_object)

        # Import the regular grain volumes first
        grain_volume_names = list(load_step_group["Grain Volume Names"].asstr())
        grain_volume_object_list = []
        for grain_volume_name in grain_volume_names:
            grain_volume_group = load_step_group["Grain Volumes"][grain_volume_name]
            grain_volume_object = GrainVolume.import_from_hdf5_group(grain_volume_group, load_step_object)

            grain_volume_object_list.append(grain_volume_object)

        load_step_object.add_grain_volumes(grain_volume_object_list)

        # Import the stitched grain volumes last
        stitched_grain_volume_names = list(load_step_group["Stitched Grain Volume Names"].asstr())
        stitched_grain_volume_object_list = []
        for stitched_grain_volume_name in stitched_grain_volume_names:
            stitched_grain_volume_group = load_step_group["Stitched Grain Volumes"][stitched_grain_volume_name]
            stitched_grain_volume_object = StitchedGrainVolume.import_from_hdf5_group(stitched_grain_volume_group,
                                                                                      load_step_object)

            stitched_grain_volume_object_list.append(stitched_grain_volume_object)

        load_step_object.add_stitched_grain_volumes(stitched_grain_volume_object_list)

        try:
            load_step_object.D = load_step_group["D"][:]
        except KeyError:
            # working with older HDF5 file, D not written
            log.warning("D not found in load step attribute! Will be defaulted")

        return load_step_object

    def export_to_hdf5_group(self, this_sample_group: h5py.Group) -> h5py.Group:
        """Export the :class:`~.LoadStep` to a :class:`h5py.Group`

        :param this_sample_group: The :class:`h5py.Group` to export this load step to
        :return: The :class:`h5py.Group` with the data filled in
        """

        this_load_step_group = this_sample_group.create_group(self.name)

        this_load_step_group.create_dataset("All Grain Volume Names", data=self.all_grain_volume_names, dtype="S256")
        this_load_step_group.create_dataset("Grain Volume Names", data=self.grain_volume_names, dtype="S256")
        this_load_step_group.create_dataset("Stitched Grain Volume Names", data=self.stitched_grain_volume_names,
                                            dtype="S256")
        this_load_step_group.create_dataset("D", data=self.D)

        this_grain_volumes_group = this_load_step_group.create_group("Grain Volumes")
        this_stitched_grain_volumes_group = this_load_step_group.create_group("Stitched Grain Volumes")

        # Export GrainVolumes first, then StitchedGrainVolumes
        for volume in self.grain_volumes_list:
            volume.export_to_hdf5_group(this_grain_volumes_group)

        for stitched_volume in self.stitched_grain_volumes_list:
            stitched_volume.export_to_hdf5_group(this_stitched_grain_volumes_group)

        return this_load_step_group

    def export_merged_volume_to_gff(self, output_directory: str,
                                    use_adjusted_position: bool = True,
                                    filter_too: bool = False,
                                    filter_bounds: Optional[
                                        Tuple[float, float, float, float, float, float]] = None) -> None:
        """Export all the :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` from the
        :attr:`~.LoadStep.stitched_grain_volumes` dictionary to the `output_directory`

        :param output_directory: The output directory to export the GFFs to
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        :param filter_too: Whether to filter the grains geometrically in each :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` before exporting, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to before export, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        """

        for stitched_volume in self.stitched_grain_volumes_list:
            for phase in stitched_volume.all_phases:
                this_stitched_map = stitched_volume.get_map_from_phase(phase)
                this_output_path = os.path.join(output_directory, f"{self.name}_{phase.name}_merged.gff")
                this_filtered_output_path = os.path.join(output_directory,
                                                         f"{self.name}_{phase.name}_merged_filtered.gff")
                merged_grains_to_export = this_stitched_map.grains
                StitchedGrainsMap.validate_grains_list(merged_grains_to_export)
                GrainsCollection.export_validated_grain_list_to_gff(grains_list=merged_grains_to_export,
                                                                    gff_path=this_output_path,
                                                                    use_adjusted_position=use_adjusted_position)
                if filter_too is True:
                    if filter_bounds is None:
                        raise ValueError("Must supply filter bounds if filter_too is true!")
                    filtered_merged_grains_to_export = filter_grain_list(merged_grains_to_export,
                                                                         filter_bounds[0],
                                                                         filter_bounds[1],
                                                                         filter_bounds[2],
                                                                         filter_bounds[3],
                                                                         filter_bounds[4],
                                                                         filter_bounds[5],
                                                                         use_adjusted_position)
                    StitchedGrainsMap.validate_grains_list(filtered_merged_grains_to_export)
                    GrainsCollection.export_validated_grain_list_to_gff(grains_list=filtered_merged_grains_to_export,
                                                                        gff_path=this_filtered_output_path,
                                                                        use_adjusted_position=use_adjusted_position)

    def __repr__(self) -> str:
        return self.name
