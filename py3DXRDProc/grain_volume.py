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

from typing import List, Dict, Tuple, TYPE_CHECKING, Optional

import h5py
import numpy as np
import numpy.typing as npt

import logging
log = logging.getLogger(__name__)

from py3DXRDProc.grain import CleanGrain, RawGrain, StitchedGrain, \
    TrackedGrain
from py3DXRDProc.grain_map import RawGrainsMap, CleanedGrainsMap, StitchedGrainsMap, TrackedGrainsMap
from py3DXRDProc.phase import Phase

if TYPE_CHECKING:
    from py3DXRDProc.load_step import LoadStep
    from py3DXRDProc.sample import Sample


class GrainVolume:
    """Class representing a contiguous volume of microstructures, containing one or more phases as maps and extra info
    such as volume size. Attached to a load step and therefore a wider sample.
    Holds both :class:`~py3DXRDProc.grain_map.RawGrainsMap` and :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` in separate ``dict``.

    :param name: Name of the grain volume
    :param load_step: Load step this grain volume is attached to
    :param index_dimensions: Indexing limits used during the grid indexing process in the lab frame in mm
    :param material_dimensions: Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes
    :param offset_origin: The coordinates of the origin of this volume in the :class:`~py3DXRDProc.sample.Sample` reference frame
    :raises TypeError: If `name` isn't a ``str``
    :raises TypeError: If `index_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `material_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `offset_origin` isn't a numpy array of ``float64``
    :raises TypeError: If `load_step` isn't a :class:`~py3DXRDProc.load_step.LoadStep` instance
    """

    def __init__(self, name: str,
                 load_step: LoadStep,
                 index_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 material_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 offset_origin: npt.NDArray[np.float64]):

        if not isinstance(name, str):
            raise TypeError("GrainVolume name should be a string!")

        if not load_step.__class__.__name__ == "LoadStep":
            raise TypeError("load_step should be a LoadStep instance!")

        if not isinstance(index_dimensions, tuple):
            raise TypeError("Index index_dimensions should be a tuple!")
        for axis in index_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the index_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the index_dimensions should be a 2-tuple")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        if not isinstance(offset_origin, np.ndarray):
            raise TypeError("Offset origin should be a Numpy array!")
        if not offset_origin.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")

        if not isinstance(material_dimensions, tuple):
            raise TypeError("Material index_dimensions should be a tuple!")
        for axis in material_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the material_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the material_dimensions should be a 2-tuple")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        self._name = name
        self._load_step = load_step
        self._index_dimensions = index_dimensions
        self._material_dimensions = material_dimensions
        self._offset_origin = offset_origin

        self._raw_maps_dict: Dict[str, RawGrainsMap] = {}
        self._clean_maps_dict: Dict[str, CleanedGrainsMap] = {}

    @property
    def name(self) -> str:
        """The grain volume name

        :return: The grain volume name
        """

        return self._name

    @property
    def index_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Indexing limits used during the grid indexing process in the lab frame in mm

        :return: The indexing limits as a tuple of 2-tuples"""

        return self._index_dimensions

    @property
    def material_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes

        :return: The material as a tuple of 2-tuples"""

        return self._material_dimensions

    @property
    def offset_origin(self) -> npt.NDArray[np.float64]:
        """The coordinates of the origin of this volume in the :class:`~py3DXRDProc.sample.Sample` reference frame

        :return: The offset origin as a Numpy array of floats
        """

        return self._offset_origin

    @property
    def material_volume(self) -> float:
        """The actual volume (in cubic mm) of material represented by this grain volume

        :return: The volume of physical material in this GrainVolume, in cubic mm
        """

        material_volume = (self.material_dimensions[0][1] - self.material_dimensions[0][0]) * \
                          (self.material_dimensions[1][1] - self.material_dimensions[1][0]) * \
                          (self.material_dimensions[2][1] - self.material_dimensions[2][0])

        return material_volume

    @property
    def load_step(self) -> LoadStep:
        """The :class:`~py3DXRDProc.load_step.LoadStep` instance this grain volume belongs to

        :return: The :class:`~py3DXRDProc.load_step.LoadStep` instance of the grain volume
        """

        return self._load_step

    @property
    def sample(self) -> Sample:
        """The grain volume :class:`~py3DXRDProc.sample.Sample` from the :attr:`~.GrainVolume.load_step`

        :return: The grain volume :class:`~py3DXRDProc.sample.Sample`
        """

        return self.load_step.sample

    @property
    def raw_maps(self) -> Dict[str, RawGrainsMap]:
        """The dictionary of :class:`~py3DXRDProc.grain_map.RawGrainsMap`
        inside this grain volume, indexed by map name

        :return: The :class:`~py3DXRDProc.grain_map.RawGrainsMap` dictionary
        """

        return self._raw_maps_dict

    @property
    def raw_map_names(self) -> List[str]:
        """The raw map names from the :attr:`~.GrainVolume.raw_maps` dictionary

        :return: List of names of all :class:`~py3DXRDProc.grain_map.RawGrainsMap` in the volume
        """

        return list(self._raw_maps_dict.keys())

    @property
    def raw_maps_list(self) -> List[RawGrainsMap]:
        """The raw maps from the :attr:`~.GrainVolume.raw_maps` dictionary

        :return: List of all :class:`~py3DXRDProc.grain_map.RawGrainsMap` in the volume
        """

        return list(self._raw_maps_dict.values())

    @property
    def clean_maps(self) -> Dict[str, CleanedGrainsMap]:
        """The dictionary of :class:`~py3DXRDProc.grain_map.CleanedGrainsMap`
        inside this grain volume, indexed by map name

        :return: The :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` dictionary
        """

        return self._clean_maps_dict

    @property
    def clean_map_names(self) -> List[str]:
        """The clean map names from the :attr:`~.GrainVolume.clean_maps` dictionary

        :return: List of names of all :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` in the volume
        """

        return list(self._clean_maps_dict.keys())

    @property
    def clean_maps_list(self) -> List[CleanedGrainsMap]:
        """The clean maps from the :attr:`~.GrainVolume.clean_maps` dictionary

        :return: List of all :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` in the volume
        """

        return list(self._clean_maps_dict.values())

    @property
    def maps(self) -> Dict[str, RawGrainsMap | CleanedGrainsMap]:
        """The dictionary of all maps inside this grain volume, indexed by map name.
        Generated on the fly from :attr:`~.GrainVolume.raw_maps` and :attr:`~.GrainVolume.clean_maps`

        :return: Maps dictionary
        """

        return {**self.raw_maps, **self.clean_maps}

    @property
    def map_names(self) -> List[str]:
        """List of all map names inside this grain volume.
        Generated on the fly from :attr:`~.GrainVolume.maps`

        :return: List of names of all grain maps in the volume
        """

        return list(self.maps.keys())

    @property
    def maps_list(self) -> List[RawGrainsMap | CleanedGrainsMap]:
        """List of all maps inside this grain volume.
        Generated on the fly from :attr:`~.GrainVolume.maps`

        :return: List of all grain maps in the volume
        """

        return list(self.maps.values())

    @property
    def all_raw_grains(self) -> List[RawGrain]:
        """Gets all :attr:`~py3DXRDProc.grain.RawGrain` in the grain volume

        :return: List of all the :class:`~py3DXRDProc.grain.RawGrain` from all the :attr:`~.GrainVolume.raw_maps`
        """

        grains_list: List[RawGrain] = []
        for raw_grain_map in self.raw_maps_list:
            grains_list.extend(raw_grain_map.grains)
        return grains_list

    @property
    def all_clean_grains(self) -> List[CleanGrain]:
        """Gets all :class:`~py3DXRDProc.grain.CleanedGrain` in the grain volume

        :return: List of all the :class:`~py3DXRDProc.grain.CleanedGrain` from all the :attr:`~.GrainVolume.clean_maps`
        """

        grains_list: List[CleanGrain] = []
        for clean_grain_map in self.clean_maps_list:
            grains_list.extend(clean_grain_map.grains)
        return grains_list

    @property
    def all_grains(self) -> List[RawGrain | CleanGrain]:
        """Gets all grains in the grain volume

        :return: List of all the grains from all the maps
        """

        grains_list: List[RawGrain | CleanGrain] = []
        for grain_map in self.maps_list:
            grains_list.extend(grain_map.grains)
        return grains_list

    @property
    def all_raw_phases(self) -> List[Phase]:
        """Gets the :class:`~py3DXRDProc.phase.Phase` from all raw maps in the volume

        :return: List of :class:`~py3DXRDProc.phase.Phase`
        """

        return list(set([a_map.phase for a_map in self.raw_maps_list]))

    @property
    def all_clean_phases(self) -> List[Phase]:
        """Gets the :class:`~py3DXRDProc.phase.Phase` from all cleaned maps in the volume.

        :return: List of :class:`~py3DXRDProc.phase.Phase`
        """

        return list(set([a_map.phase for a_map in self.clean_maps_list]))

    @property
    def all_phases(self) -> List[Phase]:
        """Gets the :class:`~py3DXRDProc.phase.Phase` from all maps in the volume.

        :return: List of :class:`~py3DXRDProc.phase.Phase`
        """
        return list(set(self.all_raw_phases + self.all_clean_phases))

    @property
    def mean_peak_intensity_sum_raw(self) -> float:
        if len(self.raw_maps_list) == 0:
            raise ValueError("This GrainVolume doesn't have any raw maps! Did you add them?")
        return float(np.sum([grain.mean_peak_intensity for grain in self.all_raw_grains]))

    def add_raw_map(self, raw_grain_map: RawGrainsMap) -> None:
        """Add a :class:`~py3DXRDProc.grain_map.RawGrainsMap` to the :attr:`~.GrainVolume.raw_maps` dictionary.

        :param raw_grain_map: The :class:`~py3DXRDProc.grain_map.RawGrainsMap` to add to the dictionary
        :raises TypeError: If `raw_grain_map` isn't a :class:`~py3DXRDProc.grain_map.RawGrainsMap` instance
        :raises ValueError: If `raw_grain_map` already exists in :attr:`~.GrainVolume.raw_maps`
        """

        if not isinstance(raw_grain_map, RawGrainsMap):
            raise TypeError("raw_grain_map should be a RawGrainsMap instance!")

        if raw_grain_map.name not in self.raw_maps:
            self._raw_maps_dict[raw_grain_map.name] = raw_grain_map
        else:
            raise ValueError(f"RawGrainsMap {raw_grain_map} already exists in this grain volume!")

    def add_clean_map(self, clean_grain_map: CleanedGrainsMap) -> None:
        """Add a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to the :attr:`~.GrainVolume.clean_maps` dictionary.

        :param clean_grain_map: The :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to add to the dictionary
        :raises TypeError: If `clean_grain_map` isn't a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` instance
        :raises ValueError: If `clean_grain_map` already exists in :attr:`~.GrainVolume.clean_maps`
        """

        if not isinstance(clean_grain_map, CleanedGrainsMap):
            raise TypeError("clean_grain_map should be a CleanedGrainsMap instance!")

        if clean_grain_map.name not in self.clean_maps:
            self._clean_maps_dict[clean_grain_map.name] = clean_grain_map
        else:
            raise ValueError(f"CleanedGrainsMap {clean_grain_map} already exists in this grain volume!")

    def add_raw_maps(self, list_of_raw_maps: List[RawGrainsMap]) -> None:
        """Add multiple :class:`~py3DXRDProc.grain_map.RawGrainsMap` to the :attr:`~.GrainVolume.raw_maps` dictionary.

        :param list_of_raw_maps: List of :class:`~py3DXRDProc.grain_map.RawGrainsMap` to add to the dictionary
        :raises TypeError: If `list_of_raw_maps` isn't a list instance
        :raises ValueError: If `list_of_raw_maps` is empty
        """

        if not isinstance(list_of_raw_maps, list):
            raise TypeError("list_of_raw_maps should be a list!")
        if len(list_of_raw_maps) == 0:
            raise ValueError("list_of_raw_maps cannot be empty!")
        for grain_map in list_of_raw_maps:
            self.add_raw_map(grain_map)

    def add_clean_maps(self, list_of_clean_maps: List[CleanedGrainsMap]) -> None:
        """Add multiple :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to the :attr:`~.GrainVolume.clean_maps` dictionary.

        :param list_of_clean_maps: List of :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to add to the dictionary
        :raises TypeError: If `list_of_clean_maps` isn't a list instance
        :raises ValueError: If `list_of_clean_maps` is empty
        """

        if not isinstance(list_of_clean_maps, list):
            raise TypeError("list_of_clean_maps should be a list!")
        if len(list_of_clean_maps) == 0:
            raise ValueError("list_of_clean_maps cannot be empty!")
        for grain_map in list_of_clean_maps:
            self.add_clean_map(grain_map)

    # Fetch methods:

    def get_raw_map(self, raw_map_name: str) -> RawGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.RawGrainsMap` from the :attr:`~.GrainVolume.raw_maps` dictionary given the map name.

        :param raw_map_name: The name of the :class:`~py3DXRDProc.grain_map.RawGrainsMap` to add to the :attr:`~.GrainVolume.raw_maps` dictionary
        :raises TypeError: If `raw_map_name` isn't a ``str``
        :raises KeyError: If `raw_map_name` couldn't be found in :attr:`~.GrainVolume.raw_maps`
        :return: The :class:`~py3DXRDProc.grain_map.RawGrainsMap`
        """

        if not isinstance(raw_map_name, str):
            raise TypeError("raw_map_name should be a string!")
        try:
            return self.raw_maps[raw_map_name]
        except KeyError:
            raise KeyError(f"Could not find raw map {raw_map_name} in raw maps dict")

    def get_clean_map(self, clean_map_name: str) -> CleanedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from the :attr:`~.GrainVolume.clean_maps` dictionary given the map name.

        :param clean_map_name: The name of the :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to add to the :attr:`~.GrainVolume.clean_maps` dictionary
        :raises TypeError: If `clean_map_name` isn't a ``str``
        :raises KeyError: If `clean_map_name` couldn't be found in :attr:`~.GrainVolume.clean_maps`
        :return: The :class:`~py3DXRDProc.grain_map.CleanedGrainsMap`
        """

        if not isinstance(clean_map_name, str):
            raise TypeError("clean_map_name should be a string!")
        try:
            return self.clean_maps[clean_map_name]
        except KeyError:
            raise KeyError(f"Could not find clean map {clean_map_name} in clean maps dict")

    def get_raw_map_from_phase(self, phase: Phase) -> RawGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.RawGrainsMap` from the :attr:`~.GrainVolume.raw_maps` dictionary given its :class:`~py3DXRDProc.phase.Phase`

        :param phase: The :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase` isn't a :class:`~py3DXRDProc.phase.Phase`
        :raises KeyError: If a raw map couldn't be found with a matching `phase`
        :return: The :class:`~py3DXRDProc.grain_map.RawGrainsMap` with a matching `phase`
        """

        if not isinstance(phase, Phase):
            raise TypeError("phase should be a Phase instance!")
        for raw_map in self.raw_maps_list:
            if raw_map.phase == phase:
                return raw_map
        raise KeyError("Raw map with this phase name not found!")

    def get_raw_map_from_phase_name(self, phase_name: str) -> RawGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.RawGrainsMap` from the :attr:`~.GrainVolume.raw_maps` dictionary given its :class:`~py3DXRDProc.phase.Phase` name

        :param phase_name: The name of the :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase_name` isn't a ``str``
        :raises KeyError: If a raw map couldn't be found with a matching `phase_name`
        :return: The :class:`~py3DXRDProc.grain_map.RawGrainsMap` with a matching `phase_name`
        """

        if not isinstance(phase_name, str):
            raise TypeError("phase_name should be a string!")
        for raw_map in self.raw_maps_list:
            if raw_map.phase.name == phase_name:
                return raw_map
        raise KeyError("Raw map with this phase name not found!")

    def get_clean_map_from_phase(self, phase: Phase) -> CleanedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from the :attr:`~.GrainVolume.clean_maps` dictionary given its :class:`~py3DXRDProc.phase.Phase`

        :param phase: The :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase` isn't a :class:`~py3DXRDProc.phase.Phase`
        :raises KeyError: If a raw map couldn't be found with a matching `phase`
        :return: The :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` with a matching `phase`
        """

        if not isinstance(phase, Phase):
            raise TypeError("phase should be a Phase instance!")
        for clean_map in self.clean_maps_list:
            if clean_map.phase == phase:
                return clean_map
        raise KeyError("Clean map with this phase name not found!")

    def get_clean_map_from_phase_name(self, phase_name: str) -> CleanedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from the :attr:`~.GrainVolume.clean_maps` dictionary given its :class:`~py3DXRDProc.phase.Phase` name

        :param phase_name: The name of the :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase_name` isn't a ``str``
        :raises KeyError: If a clean map couldn't be found with a matching `phase_name`
        :return: The :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` with a matching `phase_name`
        """

        if not isinstance(phase_name, str):
            raise TypeError("phase_name should be a string!")
        for clean_map in self.clean_maps_list:
            if clean_map.phase.name == phase_name:
                return clean_map
        raise KeyError("Clean map with this phase name not found!")

    def get_map(self, map_name: str) -> RawGrainsMap | CleanedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.RawGrainsMap` or :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from the :attr:`~.GrainVolume.maps` dictionary given the map name.

        :param map_name: The name of the :class:`~py3DXRDProc.grain_map.RawGrainsMap` or :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to look for
        :raises TypeError: If `map_name` isn't a ``str``
        :raises KeyError: If `map_name` couldn't be found in :attr:`~.GrainVolume.maps`
        :return: The :class:`~py3DXRDProc.grain_map.RawGrainsMap` or :class:`~py3DXRDProc.grain_map.CleanedGrainsMap`
        """

        if not isinstance(map_name, str):
            raise TypeError("map_name should be a string!")
        try:
            return self.maps[map_name]
        except KeyError:
            raise KeyError(f"Could not find map {map_name} in raw_maps or clean_maps dicts")

    def clean(self, dist_tol: float = 0.1, angle_tol: float = 1.0) -> None:
        """Take all the :class:`~py3DXRDProc.grain_map.RawGrainsMap` in the :attr:`~.GrainVolume.raw_maps` dictionary.
        Generate a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` from each :class:`~py3DXRDProc.grain_map.RawGrainsMap`.
        Add each :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` to the :attr:`~.GrainVolume.clean_maps` dictionary.

        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises ValueError: If the :attr:`~.GrainVolume.raw_maps` dictionary is empty so there's nothing to clean
        """

        if len(self.raw_maps_list) == 0:
            raise ValueError("The raw maps list is empty! There's nothing to clean!")

        for raw_grain_map in self.raw_maps_list:
            clean_grain_map = CleanedGrainsMap.from_cleaning_grain_map(input_grain_map=raw_grain_map,
                                                                       dist_tol=dist_tol,
                                                                       angle_tol=angle_tol)
            log.info(f"Went from {len(raw_grain_map)} to {len(clean_grain_map)} grains")
            self.add_clean_map(clean_grain_map)

    @classmethod
    def import_from_hdf5_group(cls, grain_volume_group: h5py.Group, load_step_object: LoadStep) -> GrainVolume:
        """Import a :class:`~.GrainVolume` from a :class:`h5py.Group`.
        Imports the underlying maps too.

        :param grain_volume_group: :class:`h5py.Group` containing the grain volume data
        :param load_step_object: :class:`~py3DXRDProc.load_step.LoadStep` object that this grain volume belongs to
        :raises TypeError: If `grain_volume_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `load_step_object` is not a :class:`~py3DXRDProc.load_step.LoadStep` instance

        :return: The :class:`~.GrainVolume` created from the HDF5 group
        """

        if not isinstance(grain_volume_group, h5py.Group):
            raise TypeError("grain_volume_group should be an h5py.Group instance")
        if not load_step_object.__class__.__name__ == "LoadStep":
            raise TypeError("load_step_object should be a LoadStep instance!")

        grain_volume_name = grain_volume_group.name.split("/")[-1]
        log.info(f"Importing grain volume {grain_volume_name}")
        index_dimensions = grain_volume_group["Index Dimensions"][()]
        index_dimensions_proper_type = ((index_dimensions[0][0], index_dimensions[0][1]),
                                        (index_dimensions[1][0], index_dimensions[1][1]),
                                        (index_dimensions[2][0], index_dimensions[2][1]))
        material_dimensions = grain_volume_group["Material Dimensions"][()]
        material_dimensions_proper_type = ((material_dimensions[0][0], material_dimensions[0][1]),
                                           (material_dimensions[1][0], material_dimensions[1][1]),
                                           (material_dimensions[2][0], material_dimensions[2][1]))
        offset_origin = grain_volume_group["Offset Origin"][:]

        grain_volume_object = GrainVolume(name=grain_volume_name,
                                          load_step=load_step_object,
                                          index_dimensions=index_dimensions_proper_type,
                                          material_dimensions=material_dimensions_proper_type,
                                          offset_origin=offset_origin)
        phase_names = list(grain_volume_group["Phases"].keys())

        raw_grain_maps_list = []
        clean_grain_maps_list = []

        have_clean_maps = True
        for phase_name in phase_names:
            log.info(f"Importing phase {phase_name}")
            phase_group = grain_volume_group["Phases"][phase_name]
            phase_object = Phase.import_from_hdf5_group(phase_group)

            # Import the raw maps first
            raw_map_group = grain_volume_group["Raw Grain Maps"][phase_name]
            raw_map_object = RawGrainsMap.import_from_hdf5_group(map_group=raw_map_group,
                                                                 grain_volume_object=grain_volume_object,
                                                                 phase_object=phase_object)
            raw_grain_maps_list.append(raw_map_object)
            if have_clean_maps:
                try:
                    clean_map_group = grain_volume_group["Clean Grain Maps"][phase_name]

                    clean_map_object = CleanedGrainsMap.import_from_hdf5_group(map_group=clean_map_group,
                                                                               grain_volume_object=grain_volume_object,
                                                                               raw_map=raw_map_object)
                    clean_grain_maps_list.append(clean_map_object)
                except KeyError:
                    # no clean maps
                    have_clean_maps = False

        grain_volume_object.add_raw_maps(raw_grain_maps_list)
        if have_clean_maps:
            grain_volume_object.add_clean_maps(clean_grain_maps_list)

        return grain_volume_object

    def export_to_hdf5_group(self, this_load_step_group: h5py.Group) -> h5py.Group:
        """Export the :class:`~.GrainVolume` to a :class:`h5py.Group`

        :param this_load_step_group: The :class:`h5py.Group` to export this grain volume to
        :return: The :class:`h5py.Group` with the data filled in
        """

        this_grain_volume_group = this_load_step_group.create_group(self.name)
        this_raw_grain_maps_group = this_grain_volume_group.create_group("Raw Grain Maps")
        this_phases_group = this_grain_volume_group.create_group("Phases")

        this_grain_volume_group.create_dataset("Index Dimensions", data=self.index_dimensions)
        this_grain_volume_group.create_dataset("Material Dimensions", data=self.material_dimensions)
        this_grain_volume_group.create_dataset("Offset Origin", data=self.offset_origin)

        for raw_grain_map in self.raw_maps_list:
            raw_grain_map.export_to_hdf5_group(this_raw_grain_maps_group)

        if len(self.clean_maps_list) != 0:
            this_clean_grain_maps_group = this_grain_volume_group.create_group("Clean Grain Maps")

            for clean_grain_map in self.clean_maps_list:
                clean_grain_map.export_to_hdf5_group(this_clean_grain_maps_group)

        phase_list = [grain_map.phase for grain_map in self.maps_list]
        for phase in phase_list:
            if phase.name not in this_phases_group.keys():
                phase.export_to_hdf5_group(this_phases_group)

        return this_grain_volume_group


class StitchedGrainVolume:
    """Class representing a stitched volume of microstructures, generated from a bunch of contiguous
    :class:`~.GrainVolume`, containing one or more phases as maps and extra info
    such as volume size. Attached to a load step and therefore a wider sample.
    Holds one or more :class:`~py3DXRDProc.grain_map.StitchedGrainsMap`

    :param index_dimensions: Indexing limits used during the grid indexing process in the lab frame in mm
    :param material_dimensions: Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes
    :param offset_origin: The coordinates of the origin of this volume in the :class:`~py3DXRDProc.sample.Sample` reference frame
    :param contrib_vols_list: A list of contributory :class:`~.GrainVolume` that made up this :class:`~.StitchedGrainVolume`
    :raises TypeError: If `name` isn't a ``str``
    :raises TypeError: If `index_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `material_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `offset_origin` isn't a numpy array of ``float64``
    :raises TypeError: If `contrib_vols_list` isn't a ``list``
    :raises TypeError: If anything in `contrib_vols_list` isn't a :class:`~.GrainVolume` instance
    :raises ValueError: If any :class:`~.GrainVolume` in `contrib_vols_list` has a different :attr:`~.GrainVolume.load_step` value
    """

    def __init__(self, index_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 material_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 offset_origin: npt.NDArray[np.float64],
                 contrib_vols_list: List[GrainVolume]):

        if not isinstance(index_dimensions, tuple):
            raise TypeError("Dimensions should be a tuple!")
        for axis in index_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the index_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the index_dimensions should be a 2-tuple!")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        if not isinstance(offset_origin, np.ndarray):
            raise TypeError("Offset origin should be a Numpy array!")
        if not offset_origin.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")

        if not isinstance(contrib_vols_list, list):
            raise TypeError("contrib_vols_list should be a list instance!")
        for grainvol in contrib_vols_list:
            if not isinstance(grainvol, GrainVolume):
                raise TypeError(f"Contributory volume is not a GrainVolume instance")

        first_grainvol_load_step = contrib_vols_list[0].load_step
        for grainvol in contrib_vols_list:
            if grainvol.load_step != first_grainvol_load_step:
                raise ValueError(f"Contributory GrainVolume {grainvol.name} has a different load step!")

        if not isinstance(material_dimensions, tuple):
            raise TypeError("Material index_dimensions should be a tuple!")
        for axis in material_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the material_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the material_dimensions should be a 2-tuple")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        name = f"{first_grainvol_load_step.name}_stitched"

        self._name = name
        self._load_step = first_grainvol_load_step
        self._index_dimensions = index_dimensions
        self._material_dimensions = material_dimensions
        self.offset_origin = offset_origin

        self._maps_dict: Dict[str, StitchedGrainsMap] = {}
        self._contrib_volumes = contrib_vols_list

    @property
    def name(self) -> str:
        """The grain volume name

        :return: The grain volume name
        """

        return self._name

    @property
    def index_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Indexing limits used during the grid indexing process in the lab frame in mm

        :return: The indexing limits as a tuple of 2-tuples"""

        return self._index_dimensions

    @property
    def material_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes

        :return: The material as a tuple of 2-tuples"""

        return self._material_dimensions

    @property
    def material_volume(self) -> float:
        """The actual volume (in cubic mm) of material represented by this grain volume

        :return: The volume of physical material in this GrainVolume, in cubic mm
        """

        material_volume = (self.material_dimensions[0][1] - self.material_dimensions[0][0]) * \
                          (self.material_dimensions[1][1] - self.material_dimensions[1][0]) * \
                          (self.material_dimensions[2][1] - self.material_dimensions[2][0])

        return material_volume

    @property
    def maps(self) -> Dict[str, StitchedGrainsMap]:
        """The dictionary of maps inside this grain volume, indexed by map name

        :return: Maps dictionary
        """

        return self._maps_dict

    @property
    def map_names(self) -> List[str]:
        """The map names from the :attr:`~.FloatingGrainVolume.maps` dictionary

        :return: List of names of all maps in the volume
        """

        return list(self._maps_dict.keys())

    @property
    def maps_list(self) -> List[StitchedGrainsMap]:
        """The maps from the :attr:`~.FloatingGrainVolume.maps` dictionary

        :return: List of all maps in the volume
        """

        return list(self._maps_dict.values())

    @property
    def all_contrib_volumes(self) -> List[GrainVolume]:
        """The list of contributory :class:`~.GrainVolume` that made up this :class:`~.StitchedGrainVolume`

        :return: The list of contributory :class:`~.GrainVolume` that made up this :class:`~.StitchedGrainVolume`
        """

        return self._contrib_volumes

    @property
    def all_contrib_volume_names(self) -> List[str]:
        """The name of each :class:`~.GrainVolume` in :attr:`~.StitchedGrainVolume.all_contrib_volumes`

        :return: The name of each :class:`~.GrainVolume` in :attr:`~.StitchedGrainVolume.all_contrib_volumes`
        """

        return [volume.name for volume in self.all_contrib_volumes]

    @property
    def load_step(self) -> LoadStep:
        """The :class:`~py3DXRDProc.load_step.LoadStep` instance this grain volume belongs to

        :return: The :class:`~py3DXRDProc.load_step.LoadStep` instance of the grain volume
        """

        return self._load_step

    @property
    def sample(self) -> Sample:
        """The grain volume :class:`~py3DXRDProc.sample.Sample` from the :attr:`~.StitchedGrainVolume.load_step`

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        return self.load_step.sample

    @property
    def all_phases(self) -> List[Phase]:
        """Gets the :class:`~py3DXRDProc.phase.Phase` from all maps in the volume

        :return: List of :class:`~py3DXRDProc.phase.Phase`
        """

        return [a_map.phase for a_map in self.maps_list]

    @property
    def all_grains(self) -> List[StitchedGrain]:
        """Gets all grains in the grain volume

        :return: List of all the grains from all the maps
        """

        grains_list: List[StitchedGrain] = []
        for grain_map in self.maps_list:
            grains_list.extend(grain_map.grains)
        return grains_list

    def get_map(self, map_name: str) -> StitchedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` from the :attr:`~.StitchedGrainVolume.maps` dictionary given the map name.

        :param map_name: The name of the :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` to look for
        :raises TypeError: If `map_name` isn't a ``str``
        :raises KeyError: If `map_name` couldn't be found in :attr:`~.StitchedGrainVolume.maps`
        :return: The :class:`~py3DXRDProc.grain_map.StitchedGrainsMap`
        """

        if not isinstance(map_name, str):
            raise TypeError("map_name should be a string!")
        try:
            return self.maps[map_name]
        except KeyError:
            raise KeyError(f"Could not find map {map_name} in map dict")

    def get_map_from_phase(self, phase: Phase) -> StitchedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` from the :attr:`~.StitchedGrainVolume.maps` dictionary given its :class:`~py3DXRDProc.phase.Phase`

        :param phase: The :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase` isn't a :class:`~py3DXRDProc.phase.Phase`
        :raises KeyError: If a map couldn't be found with a matching `phase`
        :return: The :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` with a matching `phase`
        """

        if not isinstance(phase, Phase):
            raise TypeError("phase should be a Phase instance!")
        for single_map in self.maps_list:
            if single_map.phase == phase:
                return single_map
        raise ValueError("Phase not found in this volume")

    def get_map_from_phase_name(self, phase_name: str) -> StitchedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` from the :attr:`~.StitchedGrainVolume.maps` dictionary given its :class:`~py3DXRDProc.phase.Phase` name

        :param phase_name: The name of the :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase_name` isn't a ``str``
        :raises KeyError: If a raw map couldn't be found with a matching `phase_name`
        :return: The :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` with a matching `phase_name`
        """

        if not isinstance(phase_name, str):
            raise TypeError("phase_name should be a string!")
        for single_map in self.maps_list:
            if single_map.phase.name == phase_name:
                return single_map
        raise ValueError("Phase name not found in this volume")

    # Add methods:

    def add_map(self, grain_map: StitchedGrainsMap) -> None:
        """Add a :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` to the :attr:`~.StitchedGrainVolume.maps` dictionary.

        :param grain_map: The :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` to add to the dictionary
        :raises TypeError: If `grain_map` isn't a :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` instance
        :raises ValueError: If `grain_map` already exists in :attr:`~.GrainVolume.maps`
        """

        if not isinstance(grain_map, StitchedGrainsMap):
            raise TypeError("grain_map should be a StitchedGrainsMap instance!")

        if grain_map.name not in self.maps:
            self.maps[grain_map.name] = grain_map
        else:
            raise ValueError(f"StitchedGrainsMap {grain_map} already exists in this grain volume!")

    def add_maps(self, grain_maps_list: List[StitchedGrainsMap]) -> None:
        """Add multiple :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` to the :attr:`~.StitchedGrainVolume.maps` dictionary.

        :param grain_maps_list: List of :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` to add to the dictionary
        :raises TypeError: If `grain_maps_list` isn't a list instance
        :raises ValueError: If `grain_maps_list` is empty
        """

        if not isinstance(grain_maps_list, list):
            raise TypeError("grain_maps_list should be a list!")
        if len(grain_maps_list) == 0:
            raise ValueError("grain_maps_list cannot be empty!")
        for grain_map in grain_maps_list:
            self.add_map(grain_map)

    @classmethod
    def from_grainvolume_list(cls, list_of_grain_volumes: List[GrainVolume],
                              filter_before_merge: bool = False,
                              filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                              dist_tol_xy: float = 0.1,
                              dist_tol_z: float = 0.2,
                              angle_tol: float = 1.0) -> StitchedGrainVolume:
        """Generate a :class:`~.StitchedGrainVolume` from a list of :class:`~.GrainVolume` by stitching them together.
        Works out all the :class:`~py3DXRDProc.phase.Phase` present in `list_of_grain_volumes`,
        groups all the :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` in `list_of_grain_volumes` by phase,
        then calls :meth:`~py3DXRDProc.grain_map.StitchedGrainsMap.from_clean_maps_list` from each group of :class:`~py3DXRDProc.grain_map.CleanedGrainsMap`.

        :param list_of_grain_volumes: A list of :class:`~.GrainVolume` to stitch together
        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~.GrainVolume` before stitching, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol_xy: The tolerance in grain centre-centre XY distance (mm)
        :param dist_tol_z: The tolerance in grain centre-centre Z distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises TypeError: If `load_step` isn't a :class:`~py3DXRDProc.load_step.LoadStep` instance
        :raises TypeError: If `list_of_grain_volumes` isn't a list
        :raises TypeError: If anything in `list_of_grain_volumes` isn't a :class:`~.GrainVolume`
        :raises TypeError: If any volume in `list_of_grain_volumes` has a different load step
        :raises ValueError: If any :class:`~.GrainVolume` in `list_of_grain_volumes` after the first :class:`~.GrainVolume` is missing a :class:`~py3DXRDProc.phase.Phase` that the first :class:`~.GrainVolume` has.
        :return: The new :class:`~.StitchedGrainVolume`
        """

        # Check the integrity of the list of grain volumes
        if not isinstance(list_of_grain_volumes, list):
            raise TypeError("list_of_grain_volumes must be a list")
        for vol in list_of_grain_volumes:
            if not isinstance(vol, GrainVolume):
                raise TypeError("Not all grain volumes are GrainVolume instances!")

        first_vol_load_step = list_of_grain_volumes[0].load_step
        for vol in list_of_grain_volumes:
            if vol.load_step != first_vol_load_step:
                raise ValueError("All grain volumes must have the same load step!")

        # the following check has been removed to allow stitching where some grain volumes are missing a phase
        # this is due to empty grain maps etc.
        # Get all phases from each grain volume
        # Check that all phases exist in all volumes

        # Get all clean phases for each grain volume
        # [[austenite, ferrite], [austenite, ferrite], [austenite]]
        phases_for_each_grain_volume = [vol.all_clean_phases for vol in list_of_grain_volumes]
        # [austenite, ferrite]
        # phases_in_first_volume = phases_for_each_grain_volume[0]
        # for phases in phases_for_each_grain_volume:
        #     # [austenite, ferrite]
        #     for phase_in_first_volume in phases_in_first_volume:
        #         # [austenite]
        #         if phase_in_first_volume not in phases:
        #             raise ValueError("Grain volumes in list have different phases!")
        if len(list_of_grain_volumes) == 1:
            merged_volume_index_dimensions = list_of_grain_volumes[0].index_dimensions
            merged_volume_material_dimensions = list_of_grain_volumes[0].material_dimensions
            merged_origin = list_of_grain_volumes[0].offset_origin

        else:
            # Determine offset origin and index_dimensions
            # Determine the index_dimensions of the merged volume
            all_volumes = list_of_grain_volumes

            # Sort the volumes by their vertical origin in the sample frame
            all_volumes = sorted(all_volumes, key=lambda x: x.offset_origin[2])

            lower_x_bound = all_volumes[0].offset_origin[0] + all_volumes[0].index_dimensions[0][0]
            upper_x_bound = all_volumes[0].offset_origin[0] + all_volumes[0].index_dimensions[0][1]

            lower_y_bound = all_volumes[0].offset_origin[1] + all_volumes[0].index_dimensions[1][0]
            upper_y_bound = all_volumes[0].offset_origin[1] + all_volumes[0].index_dimensions[1][1]

            lower_z_bound = all_volumes[0].offset_origin[2] + all_volumes[0].index_dimensions[2][0]
            upper_z_bound = all_volumes[-1].offset_origin[2] + all_volumes[-1].index_dimensions[2][1]

            lower_x_bound_mat = all_volumes[0].offset_origin[0] + all_volumes[0].material_dimensions[0][0]
            upper_x_bound_mat = all_volumes[0].offset_origin[0] + all_volumes[0].material_dimensions[0][1]

            lower_y_bound_mat = all_volumes[0].offset_origin[1] + all_volumes[0].material_dimensions[1][0]
            upper_y_bound_mat = all_volumes[0].offset_origin[1] + all_volumes[0].material_dimensions[1][1]

            lower_z_bound_mat = all_volumes[0].offset_origin[2] + all_volumes[0].material_dimensions[2][0]
            upper_z_bound_mat = all_volumes[-1].offset_origin[2] + all_volumes[-1].material_dimensions[2][1]

            merged_volume_index_dimensions = ((lower_x_bound, upper_x_bound),
                                              (lower_y_bound, upper_y_bound),
                                              (lower_z_bound, upper_z_bound))

            merged_volume_material_dimensions = ((lower_x_bound_mat, upper_x_bound_mat),
                                                 (lower_y_bound_mat, upper_y_bound_mat),
                                                 (lower_z_bound_mat, upper_z_bound_mat))

            merged_origin = np.array([0., 0., 0.])

        stitched_grain_volume_obj = StitchedGrainVolume(index_dimensions=merged_volume_index_dimensions,
                                                        material_dimensions=merged_volume_material_dimensions,
                                                        offset_origin=merged_origin,
                                                        contrib_vols_list=list_of_grain_volumes)

        # For each phase, get a list of cleaned maps
        all_phases = phases_for_each_grain_volume[0]
        stitched_maps_for_each_phase = []
        for phase in all_phases:
            # Get the grain map with this phase from each volume in list_of_grain_volumes
            this_phase_cleaned_maps = []
            for vol in list_of_grain_volumes:
                # try and get the clean grain map with that phase from each grain volume
                try:
                    this_vol_clean_map_with_phase = vol.get_clean_map_from_phase(phase)
                    this_phase_cleaned_maps.append(this_vol_clean_map_with_phase)
                # but if you can't (no grains in that phase in that volume), carry on
                except KeyError:
                    continue
            this_phase_stitched_map = StitchedGrainsMap.from_clean_maps_list(clean_maps_list=this_phase_cleaned_maps,
                                                                             merged_volume=stitched_grain_volume_obj,
                                                                             filter_before_merge=filter_before_merge,
                                                                             filter_bounds=filter_bounds,
                                                                             dist_tol_xy=dist_tol_xy,
                                                                             dist_tol_z=dist_tol_z,
                                                                             angle_tol=angle_tol)
            stitched_maps_for_each_phase.append(this_phase_stitched_map)

        stitched_grain_volume_obj.add_maps(stitched_maps_for_each_phase)
        return stitched_grain_volume_obj

    @classmethod
    def import_from_hdf5_group(cls, grain_volume_group: h5py.Group, load_step_object: LoadStep) -> StitchedGrainVolume:
        """Import a :class:`~.StitchedGrainVolume` from a :class:`h5py.Group`.
        Imports the underlying maps too.

        :param grain_volume_group: :class:`h5py.Group` containing the grain volume data
        :param load_step_object: :class:`~py3DXRDProc.load_step.LoadStep` object that this grain volume belongs to
        :raises TypeError: If `grain_volume_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `load_step_object` is not a :class:`~py3DXRDProc.load_step.LoadStep` instance

        :return: The :class:`~.StitchedGrainVolume` created from the HDF5 group
        """

        if not isinstance(grain_volume_group, h5py.Group):
            raise TypeError("grain_volume_group should be an h5py.Group instance")
        if not load_step_object.__class__.__name__ == "LoadStep":
            raise TypeError("load_step_object should be a LoadStep instance!")

        grain_volume_name = grain_volume_group.name.split("/")[-1]
        log.info(f"Importing stitched grain volume {grain_volume_name}")
        index_dimensions = grain_volume_group["Index Dimensions"][()]
        index_dimensions_proper_type = ((index_dimensions[0][0], index_dimensions[0][1]),
                                        (index_dimensions[1][0], index_dimensions[1][1]),
                                        (index_dimensions[2][0], index_dimensions[2][1]))
        material_dimensions = grain_volume_group["Material Dimensions"][()]
        material_dimensions_proper_type = ((material_dimensions[0][0], material_dimensions[0][1]),
                                           (material_dimensions[1][0], material_dimensions[1][1]),
                                           (material_dimensions[2][0], material_dimensions[2][1]))
        offset_origin = grain_volume_group["Offset Origin"][:]

        # Which volumes contributed to this grain volume object?
        contrib_volume_strings_array = grain_volume_group["Contributory Volumes"][:]

        contrib_volumes_list = [load_step_object.get_grain_volume(contrib_volume_string.decode("utf-8")) for
                                contrib_volume_string in
                                contrib_volume_strings_array]

        grain_volume_object = StitchedGrainVolume(index_dimensions=index_dimensions_proper_type,
                                                  material_dimensions=material_dimensions_proper_type,
                                                  offset_origin=offset_origin,
                                                  contrib_vols_list=contrib_volumes_list)
        phase_names = list(grain_volume_group["Phases"].keys())
        grain_map_object_list = []
        # iterate over the phases
        for phase_name in phase_names:
            log.info(f"Importing phase {phase_name}")

            map_group = grain_volume_group["Stitched Grain Maps"][phase_name]
            # What are the clean grain maps that made up the stitched grain map for this phase?
            # We can get this from the contributory volumes

            clean_grain_maps_for_this_phase = []
            for contrib_volume in contrib_volumes_list:
                try:
                    clean_map = contrib_volume.get_clean_map_from_phase_name(phase_name)
                    clean_grain_maps_for_this_phase.append(clean_map)
                except KeyError:  # missing a clean map for this phase in this volume
                    continue

            # clean_grain_maps_for_this_phase = [contrib_volume.get_clean_map_from_phase_name(phase_name) for
            #                                    contrib_volume in contrib_volumes_list]
            # StitchedGrainsMaps get their phases from the first clean grain map in clean_maps_list
            map_object = StitchedGrainsMap.import_from_hdf5_group(map_group=map_group,
                                                                  grain_volume_object=grain_volume_object,
                                                                  clean_maps_list=clean_grain_maps_for_this_phase)

            grain_map_object_list.append(map_object)

        grain_volume_object.add_maps(grain_map_object_list)

        return grain_volume_object

    def export_to_hdf5_group(self, this_load_step_group: h5py.Group) -> h5py.Group:
        """Export the :class:`~.StitchedGrainVolume` to a :class:`h5py.Group`

        :param this_load_step_group: The :class:`h5py.Group` to export this grain volume to
        :return: The :class:`h5py.Group` with the data filled in
        """

        this_grain_volume_group = this_load_step_group.create_group(self.name)
        this_grain_maps_group = this_grain_volume_group.create_group("Stitched Grain Maps")
        this_phases_group = this_grain_volume_group.create_group("Phases")

        this_grain_volume_group.create_dataset("Index Dimensions", data=self.index_dimensions)
        this_grain_volume_group.create_dataset("Material Dimensions", data=self.material_dimensions)
        this_grain_volume_group.create_dataset("Offset Origin", data=self.offset_origin)

        for phase in self.all_phases:
            # Export the phase itself into the phase group
            if phase.name not in this_phases_group.keys():
                phase.export_to_hdf5_group(this_phases_group)

            # Export the stitched grain map for this phase
            grain_map_for_this_phase = self.get_map_from_phase(phase)
            grain_map_for_this_phase.export_to_hdf5_group(this_grain_maps_group)

        # Export the associated grain volumes
        contrib_volume_strings_array = np.array(self.all_contrib_volume_names, dtype="S256")
        this_grain_volume_group.create_dataset("Contributory Volumes", data=contrib_volume_strings_array, dtype="S256")

        return this_grain_volume_group


class TrackedGrainVolume:
    """Class representing a volume of microstructures that has been tracked over multiple load steps, generated from a list of many :class:`~.StitchedGrainVolume`, containing one or more phases as maps and extra info
    such as volume size. Not attached to a load step, just a wider sample.
    Holds one or more :class:`~py3DXRDProc.grain_map.TrackedGrainsMap`

    :param index_dimensions: Indexing limits used during the grid indexing process in the lab frame in mm
    :param material_dimensions: Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes
    :param offset_origin: The coordinates of the origin of this volume in the :class:`~py3DXRDProc.sample.Sample` reference frame
    :param contrib_vols_list: A list of contributory :class:`~.StitchedGrainVolume` that made up this :class:`~.TrackedGrainVolume`
    :raises TypeError: If `index_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `material_dimensions` isn't a 3-tuple of 2-tuples of floats
    :raises TypeError: If `offset_origin` isn't a numpy array of ``float64``
    :raises TypeError: If `load_step` isn't a :class:`~py3DXRDProc.load_step.LoadStep` instance
    :raises TypeError: If `contrib_vols_list` isn't a ``list``
    :raises TypeError: If anything in `contrib_vols_list` isn't a :class:`~.StitchedGrainVolume` instance
    :raises ValueError: If any :class:`~.GrainVolume` in `contrib_vols_list` has a different :attr:`~.GrainVolume.load_step` value
    """

    def __init__(self, index_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 material_dimensions: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 offset_origin: npt.NDArray[np.float64],
                 contrib_vols_list: List[StitchedGrainVolume]):

        if not isinstance(index_dimensions, tuple):
            raise TypeError("Dimensions should be a tuple!")
        for axis in index_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the index_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the index_dimensions should be a 2-tuple!")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        if not isinstance(offset_origin, np.ndarray):
            raise TypeError("Offset origin should be a Numpy array!")
        if not offset_origin.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")

        if not isinstance(contrib_vols_list, list):
            raise TypeError("contrib_vols_list should be a list instance!")

        first_grainvol_sample = contrib_vols_list[0].sample
        for grainvol in contrib_vols_list:
            if grainvol.sample != first_grainvol_sample:
                raise ValueError(f"Contributory StitchedGrainVolume {grainvol.name} has a different sample!")

        if not isinstance(material_dimensions, tuple):
            raise TypeError("Material index_dimensions should be a tuple!")
        for axis in material_dimensions:
            if not isinstance(axis, tuple):
                raise TypeError("Each axis of the material_dimensions should be a 2-tuple!")
            if not len(axis) == 2:
                raise ValueError("Each axis of the material_dimensions should be a 2-tuple")
            for bound in axis:
                if not isinstance(bound, float):
                    raise TypeError("Each bound must be a float!")

        name = f"{first_grainvol_sample.name}_tracked"
        self._name = name
        self._sample = first_grainvol_sample
        self._index_dimensions = index_dimensions
        self._material_dimensions = material_dimensions
        self.offset_origin: npt.NDArray[np.float64] = offset_origin
        self._maps_dict: Dict[str, TrackedGrainsMap] = {}
        self._contrib_volumes = contrib_vols_list

    @property
    def name(self) -> str:
        """The grain volume name

        :return: The grain volume name
        """

        return self._name

    @property
    def index_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Indexing limits used during the grid indexing process in the lab frame in mm

        :return: The indexing limits as a tuple of 2-tuples"""

        return self._index_dimensions

    @property
    def material_dimensions(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Actual index_dimensions of the material in this grain volume, in the sample frame in mm. Used to scale individual grain volumes

        :return: The material as a tuple of 2-tuples"""

        return self._material_dimensions

    @property
    def material_volume(self) -> float:
        """The actual volume (in cubic mm) of material represented by this grain volume

        :return: The volume of physical material in this GrainVolume, in cubic mm
        """

        material_volume = (self.material_dimensions[0][1] - self.material_dimensions[0][0]) * \
                          (self.material_dimensions[1][1] - self.material_dimensions[1][0]) * \
                          (self.material_dimensions[2][1] - self.material_dimensions[2][0])

        return material_volume

    @property
    def maps(self) -> Dict[str, TrackedGrainsMap]:
        """The dictionary of :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` inside this grain volume, indexed by map name

        :return: :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` dictionary
        """

        return self._maps_dict

    @property
    def map_names(self) -> List[str]:
        """The map names from the :attr:`~.TrackedGrainVolume.maps` dictionary

        :return: List of names of all maps in the volume
        """

        return list(self._maps_dict.keys())

    @property
    def maps_list(self) -> List[TrackedGrainsMap]:
        """The maps from the :attr:`~.TrackedGrainVolume.maps` dictionary

        :return: List of all maps in the volume
        """

        return list(self._maps_dict.values())

    @property
    def all_contrib_volumes(self) -> List[StitchedGrainVolume]:
        """The list of contributory :class:`~.StitchedGrainVolume` that made up this :class:`~.TrackedGrainVolume`

        :return: The list of contributory :class:`~.StitchedGrainVolume` that made up this :class:`~.TrackedGrainVolume`
        """

        return self._contrib_volumes

    @property
    def all_contrib_volume_names(self) -> List[str]:
        """The name of each :class:`~.StitchedGrainVolume` in :attr:`~.TrackedGrainVolume.all_contrib_volumes`

        :return: The name of each :class:`~.StitchedGrainVolume` in :attr:`~.TrackedGrainVolume.all_contrib_volumes`
        """

        return [volume.name for volume in self.all_contrib_volumes]

    @property
    def sample(self) -> Sample:
        """The grain volume :class:`~py3DXRDProc.sample.Sample`, hardcoded

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        return self._sample

    @property
    def all_phases(self) -> List[Phase]:
        """Gets the :class:`~py3DXRDProc.phase.Phase` from all maps in the volume

        :return: List of :class:`~py3DXRDProc.phase.Phase`
        """

        return [a_map.phase for a_map in self.maps_list]

    @property
    def all_grains(self) -> List[TrackedGrain]:
        """Gets all grains in the grain volume

        :return: List of all the grains from all the maps
        """

        grains_list: List[TrackedGrain] = []
        for grain_map in self.maps_list:
            grains_list.extend(grain_map.grains)
        return grains_list

    def get_map_from_phase(self, phase: Phase) -> TrackedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` from the :attr:`~.TrackedGrainVolume.maps` dictionary given its :class:`~py3DXRDProc.phase.Phase`

        :param phase: The :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase` isn't a :class:`~py3DXRDProc.phase.Phase`
        :raises KeyError: If a map couldn't be found with a matching `phase`
        :return: The :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` with a matching `phase`
        """

        for single_map in self.maps_list:
            if single_map.phase == phase:
                return single_map
        raise ValueError("Phase not found in this volume")

    def get_map_from_phase_name(self, phase_name: str) -> TrackedGrainsMap:
        """Get a :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` from the :attr:`~.TrackedGrainVolume.maps` dictionary given its :class:`~py3DXRDProc.phase.Phase` name

        :param phase_name: The name of the :class:`~py3DXRDProc.phase.Phase` to look for
        :raises TypeError: If `phase_name` isn't a ``str``
        :raises KeyError: If a raw map couldn't be found with a matching `phase_name`
        :return: The :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` with a matching `phase_name`
        """

        for single_map in self.maps_list:
            if single_map.phase.name == phase_name:
                return single_map
        raise ValueError("Phase name not found in this volume")

    # Add methods:

    def add_map(self, grain_map: TrackedGrainsMap) -> None:
        """Add a :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` to the :attr:`~.TrackedGrainVolume.maps` dictionary.

        :param grain_map: The :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` to add to the dictionary
        :raises TypeError: If `grain_map` isn't a :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` instance
        :raises ValueError: If `grain_map` already exists in :attr:`~.TrackedGrainVolume.maps`
        """

        if not isinstance(grain_map, TrackedGrainsMap):
            raise TypeError("grain_map should be a TrackedGrainsMap instance!")
        if grain_map.name not in self.maps:
            self.maps[grain_map.name] = grain_map
        else:
            raise ValueError(f"TrackedGrainsMap {grain_map} already exists in this grain volume!")

    def add_maps(self, grain_maps_list: List[TrackedGrainsMap]) -> None:
        """Add multiple :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` to the :attr:`~.TrackedGrainVolume.maps` dictionary.

        :param grain_maps_list: List of :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` to add to the dictionary
        :raises TypeError: If `grain_maps_list` isn't a list instance
        :raises ValueError: If `grain_maps_list` is empty
        """

        if not isinstance(grain_maps_list, list):
            raise TypeError("grain_maps_list should be a list!")
        if len(grain_maps_list) == 0:
            raise ValueError("grain_maps_list cannot be empty!")
        for grain_map in grain_maps_list:
            self.add_map(grain_map)

    @classmethod
    def from_grainvolume_list(cls, list_of_grain_volumes: List[StitchedGrainVolume],
                              filter_before_merge: bool = False,
                              filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                              dist_tol: float = 0.1,
                              angle_tol: float = 1.0) -> TrackedGrainVolume:
        """Generate a :class:`~.TrackedGrainVolume` from a list of :class:`~.StitchedGrainVolume` by tracking the grains over multiple load steps.
        Works out all the :class:`~py3DXRDProc.phase.Phase` present in `list_of_grain_volumes`,
        groups all the :class:`~py3DXRDProc.grain_map.StitchedGrainsMap` in `list_of_grain_volumes` by phase,
        then calls :meth:`~py3DXRDProc.grain_map.TrackedGrainsMap.from_stitch_maps_list` from each group of :class:`~py3DXRDProc.grain_map.StitchedGrainsMap`.

        :param list_of_grain_volumes: A list of :class:`~.StitchedGrainVolume` to stitch together
        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~.StitchedGrainVolume` before stitching, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises TypeError: If `list_of_grain_volumes` isn't a list
        :raises TypeError: If anything in `list_of_grain_volumes` isn't a :class:`~.StitchedGrainVolume`
        :raises ValueError: If any volume in `list_of_grain_volumes` has a different :class:`~py3DXRDProc.sample.Sample`
        :raises ValueError: If any :class:`~.StitchedGrainVolume` in `list_of_grain_volumes` after the first :class:`~.StitchedGrainVolume` is missing a :class:`~py3DXRDProc.phase.Phase` that the first :class:`~.StitchedGrainVolume` has.
        :return: The new :class:`~.TrackedGrainVolume`
        """

        # Check the integrity of the list of grain volumes
        if not isinstance(list_of_grain_volumes, list):
            raise TypeError("list_of_grain_volumes must be a list")
        for vol in list_of_grain_volumes:
            if not isinstance(vol, StitchedGrainVolume):
                raise TypeError("Not all grain volumes are StitchedGrainVolume instances!")

        first_vol_sample = list_of_grain_volumes[0].sample
        for vol in list_of_grain_volumes:
            if vol.sample != first_vol_sample:
                raise ValueError("Not all grain volume samples are the same!")

        # Get all phases from each grain volume
        # Check that all phases exist in all volumes
        phases_for_each_grain_volume = [vol.all_phases for vol in list_of_grain_volumes]
        phases_in_first_volume = phases_for_each_grain_volume[0]
        for phases in phases_for_each_grain_volume:
            for phase_in_first_volume in phases_in_first_volume:
                if phase_in_first_volume not in phases:
                    raise ValueError("Grain volumes in list have different phases!")

        # When you track across multiple load steps, the offset origin can be 0
        # The index_dimensions should be the largest

        all_x_min = [volume.index_dimensions[0][0] for volume in list_of_grain_volumes]
        all_x_max = [volume.index_dimensions[0][1] for volume in list_of_grain_volumes]
        all_y_min = [volume.index_dimensions[1][0] for volume in list_of_grain_volumes]
        all_y_max = [volume.index_dimensions[1][1] for volume in list_of_grain_volumes]
        all_z_min = [volume.index_dimensions[2][0] for volume in list_of_grain_volumes]
        all_z_max = [volume.index_dimensions[2][1] for volume in list_of_grain_volumes]

        min_x_min = sorted(all_x_min)[0]
        max_x_max = sorted(all_x_max)[-1]
        min_y_min = sorted(all_y_min)[0]
        max_y_max = sorted(all_y_max)[-1]
        min_z_min = sorted(all_z_min)[0]
        max_z_max = sorted(all_z_max)[-1]

        all_x_min_mat = [volume.material_dimensions[0][0] for volume in list_of_grain_volumes]
        all_x_max_mat = [volume.material_dimensions[0][1] for volume in list_of_grain_volumes]
        all_y_min_mat = [volume.material_dimensions[1][0] for volume in list_of_grain_volumes]
        all_y_max_mat = [volume.material_dimensions[1][1] for volume in list_of_grain_volumes]
        all_z_min_mat = [volume.material_dimensions[2][0] for volume in list_of_grain_volumes]
        all_z_max_mat = [volume.material_dimensions[2][1] for volume in list_of_grain_volumes]

        min_x_min_mat = sorted(all_x_min_mat)[0]
        max_x_max_mat = sorted(all_x_max_mat)[-1]
        min_y_min_mat = sorted(all_y_min_mat)[0]
        max_y_max_mat = sorted(all_y_max_mat)[-1]
        min_z_min_mat = sorted(all_z_min_mat)[0]
        max_z_max_mat = sorted(all_z_max_mat)[-1]

        tracked_volume_index_dimensions = ((min_x_min, max_x_max),
                                           (min_y_min, max_y_max),
                                           (min_z_min, max_z_max))

        tracked_volume_material_dimensions = ((min_x_min_mat, max_x_max_mat),
                                              (min_y_min_mat, max_y_max_mat),
                                              (min_z_min_mat, max_z_max_mat))

        log.info(f"Tracked volume index_dimensions: {tracked_volume_index_dimensions}")

        tracked_origin = np.array([0., 0., 0.])

        tracked_grain_volume_obj = TrackedGrainVolume(index_dimensions=tracked_volume_index_dimensions,
                                                      material_dimensions=tracked_volume_material_dimensions,
                                                      offset_origin=tracked_origin,
                                                      contrib_vols_list=list_of_grain_volumes)

        # For each phase, get a list of stitched maps
        # Gets the list of phases from the first grain volume (i.e the first load step)
        all_phases = phases_for_each_grain_volume[0]
        tracked_maps_for_each_phase = []
        for phase in all_phases:
            log.info(f"Tracking {phase.name} phase")
            this_phase_maps = []
            for vol in list_of_grain_volumes:
                this_phase_maps.append(vol.get_map_from_phase(phase))
            this_phase_tracked_map = TrackedGrainsMap.from_stitch_maps_list(stitch_maps_list=this_phase_maps,
                                                                            tracked_volume=tracked_grain_volume_obj,
                                                                            filter_before_merge=filter_before_merge,
                                                                            filter_bounds=filter_bounds,
                                                                            dist_tol=dist_tol, angle_tol=angle_tol)

            tracked_maps_for_each_phase.append(this_phase_tracked_map)

        tracked_grain_volume_obj.add_maps(tracked_maps_for_each_phase)

        return tracked_grain_volume_obj

    @classmethod
    def import_from_hdf5_group(cls, grain_volume_group: h5py.Group, sample_object: Sample) -> TrackedGrainVolume:
        """Import a :class:`~.TrackedGrainVolume` from a :class:`h5py.Group`.
        Imports the underlying maps too.

        :param grain_volume_group: :class:`h5py.Group` containing the grain volume data
        :param sample_object: :class:`~py3DXRDProc.sample.Sample` object that this grain volume belongs to
        :raises TypeError: If `grain_volume_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `sample_object` is not a :class:`~py3DXRDProc.sample.Sample` instance

        :return: The :class:`~.TrackedGrainVolume` created from the HDF5 group
        """

        if not isinstance(grain_volume_group, h5py.Group):
            raise TypeError("grain_volume_group should be an h5py.Group instance")
        if not sample_object.__class__.__name__ == "Sample":
            raise TypeError("sample_object should be a Sample instance!")

        grain_volume_name = grain_volume_group.name.split("/")[-1]
        log.info(f"Importing tracked grain volume {grain_volume_name}")
        index_dimensions = grain_volume_group["Index Dimensions"][()]
        index_dimensions_proper_type = ((index_dimensions[0][0], index_dimensions[0][1]),
                                        (index_dimensions[1][0], index_dimensions[1][1]),
                                        (index_dimensions[2][0], index_dimensions[2][1]))
        material_dimensions = grain_volume_group["Material Dimensions"][()]
        material_dimensions_proper_type = ((material_dimensions[0][0], material_dimensions[0][1]),
                                           (material_dimensions[1][0], material_dimensions[1][1]),
                                           (material_dimensions[2][0], material_dimensions[2][1]))
        offset_origin = grain_volume_group["Offset Origin"][:]

        # Which volumes contributed to this grain volume object?
        contrib_volume_strings_array = grain_volume_group["Contributory Volumes"][:]

        contrib_volumes_list = [sample_object.get_stitched_grain_volume(contrib_volume_string.decode("utf-8")) for
                                contrib_volume_string in
                                contrib_volume_strings_array]

        grain_volume_object = TrackedGrainVolume(index_dimensions=index_dimensions_proper_type,
                                                 material_dimensions=material_dimensions_proper_type,
                                                 offset_origin=offset_origin,
                                                 contrib_vols_list=contrib_volumes_list)

        phase_names = list(grain_volume_group["Phases"].keys())
        grain_map_object_list = []
        for phase_name in phase_names:
            log.info(f"Importing phase {phase_name}")

            map_group = grain_volume_group["Tracked Grain Maps"][phase_name]

            grain_maps_for_this_phase = [contrib_volume.get_map_from_phase_name(phase_name) for
                                         contrib_volume in contrib_volumes_list]

            map_object = TrackedGrainsMap.import_from_hdf5_group(map_group=map_group,
                                                                 grain_volume_object=grain_volume_object,
                                                                 stitch_maps_list=grain_maps_for_this_phase)

            grain_map_object_list.append(map_object)

        grain_volume_object.add_maps(grain_map_object_list)

        return grain_volume_object

    def export_to_hdf5_group(self, this_sample_group: h5py.Group) -> h5py.Group:
        """Export the :class:`~.TrackedGrainVolume` to a :class:`h5py.Group`

        :param this_sample_group: The :class:`h5py.Group` to export this grain volume to
        :return: The :class:`h5py.Group` with the data filled in
        """

        this_grain_volume_group = this_sample_group.create_group(self.name)
        this_grain_maps_group = this_grain_volume_group.create_group("Tracked Grain Maps")
        this_phases_group = this_grain_volume_group.create_group("Phases")

        this_grain_volume_group.create_dataset("Index Dimensions", data=self.index_dimensions)
        this_grain_volume_group.create_dataset("Material Dimensions", data=self.material_dimensions)
        this_grain_volume_group.create_dataset("Offset Origin", data=self.offset_origin)

        for phase in self.all_phases:
            # Export the phase itself into the phase group
            if phase.name not in this_phases_group.keys():
                phase.export_to_hdf5_group(this_phases_group)

            # Export the stitched grain map for this phase
            grain_map_for_this_phase = self.get_map_from_phase(phase)
            grain_map_for_this_phase.export_to_hdf5_group(this_grain_maps_group)

        # Export the associated grain volumes
        contrib_volume_strings_array = np.array(self.all_contrib_volume_names, dtype="S256")
        this_grain_volume_group.create_dataset("Contributory Volumes", data=contrib_volume_strings_array, dtype="S256")

        return this_grain_volume_group
