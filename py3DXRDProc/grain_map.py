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

import copy
import os

from typing import TYPE_CHECKING, Generic, List, Tuple, Any, Optional, TypeVar

import logging

log = logging.getLogger(__name__)

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from py3DXRDProc.grain import BaseGrain, RawGrain, CleanGrain, StitchedGrain, filter_grain_list, \
    BaseMapGrain, find_multiple_observations, TrackedGrain, TBaseGrain, TBaseMapGrain, \
    find_multiple_observations_stitching
from py3DXRDProc.io_tools import make_folder
from py3DXRDProc.phase import Phase

if TYPE_CHECKING:
    from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume, TrackedGrainVolume
    from py3DXRDProc.load_step import LoadStep
    from py3DXRDProc.sample import Sample

all_gff_columns = "grain_id grainid grainno grain_no phase_id chisq mean_IA grainvolume grainsize x y z rodx rody rodz phi1 PHI phi2 U11 U12 U13 U21 U22 U23 U31 U32 U33 UBI11 UBI12 UBI13 UBI21 UBI22 UBI23 UBI31 UBI32 UBI33 eps11 eps22 eps33 eps23 eps13 eps12 eps11_s eps22_s eps33_s eps23_s eps13_s eps12_s sig11 sig22 sig33 sig23 sig13 sig12 sig11_s sig22_s sig33_s sig23_s sig13_s sig12_s sig_tth sig_eta".split(
    " ")

# Things we'd like in the gff, with attribute names rather than column names
required_gff_grain_attribute_names = ["gid", "pos", "U"]
optional_gff_grain_attribute_names = ["phase_id", "pos_offset", "volume", "UBI", "size", "eul", "rod", "eps",
                                      "eps_lab", "sig", "sig_lab"]
# The ordering here is important, it needs to match the desired FitAllB column ordering
all_gff_grain_attribute_names = ["gid", "phase_id", "chisq", "mean_IA", "volume", "size", "pos", "pos_offset",
                                 "eul", "rod", "U", "UBI", "eps", "eps_lab", "sig", "sig_lab", "sig_tth", "sig_eta"]

# Different types of attributes:
# int
# bool
# 1-dimensional float (scalar or 1-length array)
# 3x3 array
#

required_obj_grain_attribute_names_int = ["gid"]
required_obj_grain_attribute_names_bool: List[str] = []
required_obj_grain_attribute_names_1d_float = ["pos", "volume", "pos_offset"]
required_obj_grain_attribute_names_3x3_float = ["UBI"]

required_obj_grain_attribute_names = required_obj_grain_attribute_names_int + required_obj_grain_attribute_names_bool + required_obj_grain_attribute_names_1d_float + required_obj_grain_attribute_names_3x3_float

calculated_obj_grain_attribute_names_1d_float = ["rod", "eul", "unitcell", "pos_error", "angle_error",
                                                 "sig_vm", "pos_sample"]
calculated_obj_grain_attribute_names_3x3_float = ["UB", "B", "U", "mt", "rmt",
                                                  "eps", "eps_lab", "eps_hydro", "eps_lab_deviatoric", "sig", "sig_lab", "sig_hydro", "sig_lab_deviatoric",
                                                  "eps_error", "eps_lab_error", "sig_error", "sig_lab_error", "U_error"]

optional_obj_grain_attribute_names_int: List[str] = []
optional_obj_grain_attribute_names_bool: List[str] = []
optional_obj_grain_attribute_names_1d_float = calculated_obj_grain_attribute_names_1d_float
optional_obj_grain_attribute_names_3x3_float = calculated_obj_grain_attribute_names_3x3_float

optional_obj_grain_attribute_names_flat = optional_obj_grain_attribute_names_int + optional_obj_grain_attribute_names_bool + optional_obj_grain_attribute_names_1d_float

optional_obj_grain_attribute_names = optional_obj_grain_attribute_names_int + optional_obj_grain_attribute_names_bool + optional_obj_grain_attribute_names_1d_float + optional_obj_grain_attribute_names_3x3_float

calculated_obj_grain_attribute_names = calculated_obj_grain_attribute_names_1d_float + calculated_obj_grain_attribute_names_3x3_float

all_int_attributes = required_obj_grain_attribute_names_int + optional_obj_grain_attribute_names_int
all_bool_attributes = required_obj_grain_attribute_names_bool + optional_obj_grain_attribute_names_bool
all_1d_float_attributes = required_obj_grain_attribute_names_1d_float + optional_obj_grain_attribute_names_1d_float
all_3x3_float_attributes = required_obj_grain_attribute_names_3x3_float + optional_obj_grain_attribute_names_3x3_float

column_to_attribute_dict = {
    "grain_id": "gid",
    "grainid": "gid",
    "grainno": "gid",
    "grain_no": "gid",
    "phase_id": "phase_id",
    "grainvolume": "volume",
    "grainsize": "size",
    "mean_IA": "mean_IA",
    "chisq": "chisq",
    "rodx": "rod",
    "rody": "rod",
    "rodz": "rod",
    "phi1": "eul",
    "PHI": "eul",
    "phi2": "eul",
    "x": "pos",
    "y": "pos",
    "z": "pos",
    "U11": "U",
    "U12": "U",
    "U13": "U",
    "U21": "U",
    "U22": "U",
    "U23": "U",
    "U31": "U",
    "U32": "U",
    "U33": "U",
    "UBI11": "UBI",
    "UBI12": "UBI",
    "UBI13": "UBI",
    "UBI21": "UBI",
    "UBI22": "UBI",
    "UBI23": "UBI",
    "UBI31": "UBI",
    "UBI32": "UBI",
    "UBI33": "UBI",
    "eps11": "eps",
    "eps22": "eps",
    "eps33": "eps",
    "eps23": "eps",
    "eps13": "eps",
    "eps12": "eps",
    "eps11_s": "eps_lab",
    "eps22_s": "eps_lab",
    "eps33_s": "eps_lab",
    "eps23_s": "eps_lab",
    "eps13_s": "eps_lab",
    "eps12_s": "eps_lab",
    "sig11": "sig",
    "sig22": "sig",
    "sig33": "sig",
    "sig23": "sig",
    "sig13": "sig",
    "sig12": "sig",
    "sig11_s": "sig_lab",
    "sig22_s": "sig_lab",
    "sig33_s": "sig_lab",
    "sig23_s": "sig_lab",
    "sig13_s": "sig_lab",
    "sig12_s": "sig_lab",
    "sig_tth": "sig_tth",
    "sig_eta": "sig_eta"
}

attribute_to_columns_dict = {
    "gid": "grainno",
    "volume": "grainvolume",
    "rod": "rodx rody rodz",
    "eul": "phi1 PHI phi2",
    "pos": "x y z",
    "pos_offset": "x y z",
    "U": "U11 U12 U13 U21 U22 U23 U31 U32 U33",
    "UBI": "UBI11 UBI12 UBI13 UBI21 UBI22 UBI23 UBI31 UBI32 UBI33",
    "eps": "eps11 eps22 eps33 eps23 eps13 eps12",
    "eps_lab": "eps11_s eps22_s eps33_s eps23_s eps13_s eps12_s",
    "sig": "sig11 sig22 sig33 sig23 sig13 sig12",
    "sig_lab": "sig11_s sig22_s sig33_s sig23_s sig13_s sig12_s",
}

single_column_attributes = {key: value for key, value in attribute_to_columns_dict.items() if
                            len(value.split(" ")) == 1}


class GrainsCollection(Generic[TBaseGrain]):
    """Base class to store a collection of grains, with no assumptions about a wider sample or load step

    :param grains_list: Optionally pass a grains list on creation
    """

    def __init__(self, grains_list: Optional[List[TBaseGrain]] = None):
        if grains_list is not None:
            self.add_grains(grains_list)  # Grains list is validated in add_grains()
        else:
            self._grains: List[TBaseGrain] = []

    @property
    def grains(self) -> List[TBaseGrain]:
        """List of grains in the :class:`~.GrainsCollection`

        :return: The list of grains in the collection
        """

        return self._grains

    def add_grain(self, grain: TBaseGrain) -> None:
        """Add a single grain to the :class:`~.GrainsCollection`.
        If the :class:`~.GrainsCollection` is empty, sets the grains property to just this grain.

        :param grain: The grain we want to add
        :raises TypeError: If `grain` is not a subclass of :class:`~py3DXRDProc.grain.BaseGrain`
        :raises ValueError: If `grain` was already found in the collection
        """

        if not isinstance(grain, BaseGrain):
            raise TypeError("Grain you added is not a grain instance!")
        if len(self.grains) == 0:
            self._grains = [grain]
        else:
            if grain not in self.grains:
                self._grains.append(grain)
            else:
                raise ValueError(f"Grain already exists in the list!")

    def add_grains(self, grains_list: List[TBaseGrain]) -> None:
        """Add multiple grains to the :class:`~.GrainsCollection`.
        If the :class:`~.GrainsCollection` is empty, sets the grains property to this grains list.

        :param grains_list: The list of the grains to add
        :raises ValueError: If any grain in `grains_list` is already in :class:`~.GrainsCollection`
        """
        self.validate_grains_list(grains_list)
        # If any grains we're going to add are already in the list, we should error the whole process before adding them
        for grain in grains_list:
            # Try to look for the grain in the list
            if grain in self.grains:
                raise ValueError("Grain already found in collection")
        for grain in grains_list:
            self.add_grain(grain)

    def remove_grain(self, grain: TBaseGrain) -> None:
        """Remove a grain from the :class:`~.GrainsCollection`.

        :param grain: The grain to remove from the map
        :raises TypeError: If `grain` is not a :class:`~py3DXRDProc.grain.BaseGrain` or subclass instance
        :raises KeyError: If `grain` is not found in the :class:`~.GrainsCollection`
        """

        if not isinstance(grain, BaseGrain):
            raise TypeError("Grain should be a BaseGrain instance!")
        if grain not in self.grains:
            raise KeyError("Grain not found!")
        self._grains.remove(grain)

    def remove_grains(self, grains_list: List[TBaseGrain]) -> None:
        """Remove a list of grains from the :class:`~.GrainsCollection`.

        :param grains_list: The list of grains to remove
        """

        self.validate_grains_list(grains_list)
        for grain in grains_list:
            self.remove_grain(grain)

    @staticmethod
    def validate_grains_list(grains_list: List[TBaseGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.BaseGrain` (or subclass) instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.BaseGrain` (or subclass) instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.BaseGrain` (or subclass) instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, BaseGrain):
                raise TypeError("All grains in grains_list must be a BaseGrain!")

    @staticmethod
    def validate_attribute_list(attribute_list: List[Any]) -> None:
        """Validate a list of grain attributes to make sure they're all the same type.

        :param attribute_list: A list of grain attributes, such as a list of :attr:`~py3DXRDProc.grain.BaseGrain.pos` or :attr:`~py3DXRDProc.grain.BaseGrain.UBI` matrices
        :raises ValueError: If the attribute list is empty
        :raises TypeError: If the attribute list has the wrong type
        :raises TypeError: If not all the attributes have the same type
        """

        # Ensure list is not length 0
        if len(attribute_list) == 0:
            raise ValueError("Attribute list is empty!")
        if not isinstance(attribute_list, list):
            raise TypeError("Attribute list is of the wrong type!")
        # Ensure that every value in the list has the same type
        # Check against the first value in the list
        if not all([isinstance(x, type(attribute_list[0])) for x in attribute_list]):
            log.debug(attribute_list)
            raise TypeError("Not all attributes have the same type!")

    @staticmethod
    def validate_attribute_array(attribute_array: np.ndarray) -> None:
        """Validate an array of grain attributes to make sure it's a non-zero array

        :param attribute_array: A numpy array of grain attributes, such as an array of :attr:`~py3DXRDProc.grain.BaseGrain.pos` or :attr:`~py3DXRDProc.grain.BaseGrain.UBI` matrices
        :raises ValueError: If the attribute array is empty
        :raises TypeError: If the attribute array has the wrong type
        """

        # Ensure list is not length 0
        if len(attribute_array) == 0:
            raise ValueError("Attribute array is empty!")
        if not isinstance(attribute_array, np.ndarray):
            raise TypeError("Attribute array is of the wrong type!")

    # Get attributes
    @staticmethod
    def get_attribute_list_from_validated_grain_list(attribute: str,
                                                     grains_list: List[TBaseGrain],
                                                     must_be_complete: bool = False) -> List[Any]:
        """Get a list of grain attributes (e.g a list of grain :attr:`~py3DXRDProc.grain.BaseGrain.pos`) from a grains list and specified attribute string.

        :param attribute: The attribute string you want to call `getattr()` with
        :param grains_list: The list of :class:`~py3DXRDProc.grain.BaseGrain` objects
        :param must_be_complete: Whether the attribute list cannot contain `None` values, defaults to `False`
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If some grains in the list don't have that attribute
        :raises AttributeError: If all grains in the list have None as the attribute
        :raises AttributeError: If `must_be_complete` is `True` and some grains in the list have `None` as the attribute
        :raises TypeError: If `must_be_complete` is `False` and `None` elements cannot be replaced with the same type as the non-`None` elements
        :return: List of grain attributes
        """

        # Check all grains have the attribute at all
        if not all([hasattr(grain, attribute) for grain in grains_list]):
            raise ValueError(f"Some grains in the list have no attribute {attribute}")
        # Checked grain list and attribute list, now get unsafe (mixed) attribute list with them
        attribute_list = copy.deepcopy([getattr(grain, attribute) for grain in grains_list])

        # If all the grains have None as the attribute
        if all([x is None for x in attribute_list]):
            raise AttributeError(f"No grains have attribute {attribute}")

        if any([x is None for x in attribute_list]):
            if must_be_complete:
                raise AttributeError(f"Some grains have None for attribute {attribute}")
            else:
                # We need to make sure to fill with the same type
                # The list typing should be consistent
                # Find out the type of the first non-None entry
                for entry in attribute_list:
                    if entry is not None:
                        if isinstance(entry, int):
                            fill_value: int = -1
                        elif isinstance(entry, float):
                            fill_value: float = np.nan
                        elif isinstance(entry, np.ndarray):
                            fill_value: np.ndarray = np.full_like(entry, np.nan)
                        elif isinstance(entry, str):
                            fill_value: str = "NAN"
                        else:
                            raise TypeError("Attribute list entries have unknown type")
                        break
                attribute_list = [fill_value if x is None else x for x in attribute_list]
        else:
            try:
                GrainsCollection.validate_attribute_list(attribute_list)
            except (ValueError, TypeError) as e:
                log.debug(repr(e))
                raise ValueError(f"Problem getting attribute {attribute}:")
        return attribute_list

    @staticmethod
    def get_attribute_array_from_validated_grain_list(attribute: str,
                                                      grains_list: List[TBaseGrain],
                                                      must_be_complete: bool = False) -> np.ndarray:

        return np.array(GrainsCollection.get_attribute_list_from_validated_grain_list(attribute=attribute,
                                                                                      grains_list=grains_list,
                                                                                      must_be_complete=must_be_complete))

    @staticmethod
    def header_list_from_validated_grain_list(grains_list: List[TBaseMapGrain] | List[TrackedGrain],
                                              use_adjusted_position: bool) -> Tuple[List[str], str]:
        """Work out what GFF headers you can throw away if any grains have non-populated values.

        :param grains_list: List of grains to check
        :param use_adjusted_position: Do we replace the `pos` gff column with :attr:`py3DXRDProc.grain.BaseMapGrain.pos_offset`
        :raises ValueError: If grain attribute name has been thrown away already
        :raises TypeError: If there's a problem with the type of the grain attributes
        :return: List of kept gff grain attribute names, GFF header string
        """

        # By default, write everything
        # If you're missing an attribute, drop that attribute (or attribute group) from the outputs
        all_attribute_groups = copy.deepcopy(all_gff_grain_attribute_names)
        groups_removed = []

        # Iterate through all possible gff attributes
        # If they're not in this grain_volume, remove them from the groups list
        for gff_column in all_gff_columns:
            grain_attribute_name: str = column_to_attribute_dict[gff_column]
            try:
                # This will fail with an AttributeError if there are any empty values
                GrainsCollection.get_attribute_list_from_validated_grain_list(attribute=grain_attribute_name,
                                                                              grains_list=grains_list,
                                                                              must_be_complete=True)
            except (AttributeError,
                    ValueError):  # Some or all values are None or don't have attribute, and we need all to be populated for the gff
                if grain_attribute_name not in groups_removed:
                    try:
                        all_attribute_groups.remove(grain_attribute_name)
                        groups_removed.append(grain_attribute_name)
                    except ValueError:
                        raise ValueError(f"Encounted ValueError at attribute {grain_attribute_name}")
            except TypeError:
                raise TypeError(f"Encounted TypeError at attribute {grain_attribute_name}")

        if use_adjusted_position:
            all_attribute_groups.remove("pos")
        else:
            all_attribute_groups.remove("pos_offset")

        output_header_list = [attribute_to_columns_dict[group] for group in all_attribute_groups]
        output_header_string = " ".join(output_header_list)

        return all_attribute_groups, output_header_string

    @staticmethod
    def export_validated_grain_list_to_gff(grains_list: List[TBaseMapGrain] | List[TrackedGrain],
                                           gff_path: str,
                                           use_adjusted_position: bool) -> None:
        """Export a grain list to a gff file. Note: this does nothing to rectify repeating grain IDs.

        :param grains_list: List of grains to export
        :param gff_path: Path to gff file to create
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        :raises TypeError: If gff path not a string
        """

        if not isinstance(gff_path, str):
            raise TypeError("Gff path must be a string!")
        header_list, header_string = GrainsCollection.header_list_from_validated_grain_list(grains_list=grains_list,
                                                                                            use_adjusted_position=use_adjusted_position)
        header_string_final = "# " + header_string + "\n"

        with open(gff_path, "w") as gff_file:
            grain_line_strings_list = [grain.to_gff_line(header_list) for grain in grains_list]
            # Remove the newline from the last entry
            grain_line_strings_list[-1] = grain_line_strings_list[-1].replace("\n", "").rstrip(" ")

            gff_file.write(header_string_final)
            gff_file.writelines(grain_line_strings_list)

    def get_attribute_list_from_grain_list(self, attribute: str,
                                           grains_list: List[TBaseGrain],
                                           must_be_complete: bool = False) -> List[Any]:
        """Get a list of grain attributes (e.g a list of grain :attr:`~py3DXRDProc.grain.BaseGrain.pos`) from a grains list and specified attribute string.

        :param attribute: The attribute string you want to call `getattr()` with
        :param grains_list: The list of :class:`~py3DXRDProc.grain.BaseGrain` objects
        :param must_be_complete: Whether the attribute list cannot contain `None` values, defaults to `False`
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If some grains in the list don't have that attribute
        :raises AttributeError: If all grains in the list have None as the attribute
        :raises AttributeError: If `must_be_complete` is `True` and some grains in the list have `None` as the attribute
        :raises TypeError: If `must_be_complete` is `False` and `None` elements cannot be replaced with the same type as the non-`None` elements
        :return: List of grain attributes
        """

        # print(f"Attr: {attribute}")
        # Check the attribute
        if not isinstance(attribute, str):
            raise TypeError("attribute must be a string")
        if attribute == "":
            raise ValueError("Attribute string cannot be empty")
        self.validate_grains_list(grains_list)
        return self.get_attribute_list_from_validated_grain_list(attribute=attribute,
                                                                 grains_list=grains_list,
                                                                 must_be_complete=must_be_complete)

    def get_attribute_list_for_all_grains(self, attribute: str,
                                          must_be_complete: bool = False) -> List[Any]:
        """Get a list of grain attributes (e.g a list of grain :attr:`~py3DXRDProc.grain.BaseGrain.pos`) for all grains from a specified attribute string.

        :param attribute: The attribute string you want to call `getattr()` with
        :param must_be_complete: Whether the attribute list cannot contain `None` values, defaults to `False`
        :raises ValueError: If you have no grains to get attributes for
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If some grains in the list don't have that attribute
        :raises AttributeError: If all grains in the list have None as the attribute
        :raises AttributeError: If `must_be_complete` is `True` and some grains in the list have `None` as the attribute
        :raises TypeError: If `must_be_complete` is `False` and `None` elements cannot be replaced with the same type as the non-`None` elements
        :return: List of grain attributes
        """

        if len(self.grains) == 0:
            raise ValueError("Grains list is empty!")
        attribute_list = self.get_attribute_list_from_grain_list(attribute=attribute,
                                                                 grains_list=self.grains,
                                                                 must_be_complete=must_be_complete)
        return attribute_list

    def get_attribute_array_from_grain_list(self, attribute: str,
                                            grains_list: List[TBaseGrain],
                                            must_be_complete: bool = False) -> np.ndarray:
        """Get a Numpy array of grain attributes (e.g a list of grain :attr:`~py3DXRDProc.grain.BaseGrain.pos`) from a grains list and specified attribute string.

        :param attribute: The attribute string you want to call `getattr()` with
        :param grains_list: The list of :class:`~py3DXRDProc.grain.BaseGrain` objects
        :param must_be_complete: Whether the attribute list cannot contain `None` values, defaults to `False`
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If some grains in the list don't have that attribute
        :raises AttributeError: If all grains in the list have None as the attribute
        :raises AttributeError: If `must_be_complete` is `True` and some grains in the list have `None` as the attribute
        :raises TypeError: If `must_be_complete` is `False` and `None` elements cannot be replaced with the same type as the non-`None` elements
        :return: Array of grain attributes
        """

        return np.array(
            self.get_attribute_list_from_grain_list(attribute=attribute,
                                                    grains_list=grains_list,
                                                    must_be_complete=must_be_complete))

    def get_attribute_array_for_all_grains(self, attribute: str, must_be_complete: bool = False) -> np.ndarray:
        """Get Numpy array of grain attributes (e.g a list of grain :attr:`~py3DXRDProc.grain.BaseGrain.pos`) for all grains from a specified attribute string.

        :param attribute: The attribute string you want to call `getattr()` with
        :param must_be_complete: Whether the attribute list cannot contain `None` values, defaults to `False`
        :raises ValueError: If you have no grains to get attributes for
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If some grains in the list don't have that attribute
        :raises AttributeError: If all grains in the list have None as the attribute
        :raises AttributeError: If `must_be_complete` is `True` and some grains in the list have `None` as the attribute
        :raises TypeError: If `must_be_complete` is `False` and `None` elements cannot be replaced with the same type as the non-`None` elements
        :return: Array of grain attributes
        """

        return np.array(self.get_attribute_list_for_all_grains(attribute=attribute, must_be_complete=must_be_complete))

    # Apply attributes

    def apply_attribute_list_to_grain_list(self, grains_list: List[TBaseGrain],
                                           attribute_name: str,
                                           attribute_values: List[Any]) -> None:
        """Apply a list of attributes to a list of grains given an attribute name.

        :param grains_list: The list of :class:`~py3DXRDProc.grain.BaseGrain` objects
        :param attribute_name: The attribute string you want to set
        :param attribute_values: List of grain attributes to set
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If the grains_list and attribute_values list have different lengths
        :raises AttributeError: If an attribute could not be set for a grain
        """

        self.validate_grains_list(grains_list)
        # Check the attribute
        if not isinstance(attribute_name, str):
            raise TypeError("attribute must be a string")
        if attribute_name == "":
            raise ValueError("Attribute name cannot be empty")
        self.validate_attribute_list(attribute_values)
        if not len(grains_list) == len(attribute_values):
            raise ValueError("Grain list and attribute list don't have same length!")
        for grain, attribute_value in zip(grains_list, attribute_values):
            try:
                setattr(grain, attribute_name, attribute_value)
            except AttributeError:
                log.warning(f"Could not set attribute {attribute_name} for grain {grain}")

    def apply_attribute_list_to_all_grains(self, attribute_name: str,
                                           attribute_values: List[Any]) -> None:
        """Apply a list of attributes to all grains given an attribute name.

        :param attribute_name: The attribute string you want to set
        :param attribute_values: List of grain attributes to set
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If the number of grains and `attribute_values` list have different lengths
        :raises AttributeError: If an attribute could not be set for a grain
        """

        self.apply_attribute_list_to_grain_list(grains_list=self.grains,
                                                attribute_name=attribute_name,
                                                attribute_values=attribute_values)

    def apply_attribute_array_to_grain_list(self, grains_list: List[TBaseGrain], attribute_name: str,
                                            attribute_values: np.ndarray) -> None:
        """Apply a Numpy array of attributes to a list of grains given an attribute name.

        :param grains_list: The list of :class:`~py3DXRDProc.grain.BaseGrain` objects
        :param attribute_name: The attribute string you want to set
        :param attribute_values: Numpy array of grain attributes to set
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If the `grains_list` and `attribute_values list` have different lengths
        :raises AttributeError: If an attribute could not be set for a grain
        """

        self.validate_grains_list(grains_list)
        # Check the attribute
        if not isinstance(attribute_name, str):
            raise TypeError("attribute must be a string")
        if attribute_name == "":
            raise ValueError("Attribute name cannot be empty")
        self.validate_attribute_array(attribute_values)
        if not len(grains_list) == len(attribute_values):
            raise ValueError("Grain list and attribute list don't have same length!")
        for grain, attribute_value in zip(grains_list, attribute_values):
            try:
                setattr(grain, attribute_name, attribute_value)
            except AttributeError:
                log.warning(f"Could not set attribute {attribute_name} for grain {grain}")

    def apply_attribute_array_to_all_grains(self, attribute_name: str, attribute_values: np.ndarray) -> None:
        """Apply a Numpy array of attributes to all grains given an attribute name.

        :param attribute_name: The attribute string you want to set
        :param attribute_values: Numpy array of grain attributes to set
        :raises TypeError: If the attribute is not a string
        :raises ValueError: If the attribute string is empty
        :raises ValueError: If the number of grains and attribute_values list have different lengths
        :raises AttributeError: If an attribute could not be set for a grain
        """

        self.apply_attribute_array_to_grain_list(grains_list=self.grains,
                                                 attribute_name=attribute_name,
                                                 attribute_values=attribute_values)

    # Filtration

    def filter_all_grains(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float,
                          use_adjusted_pos: bool = False) -> List[TBaseGrain]:
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

        return filter_grain_list(self.grains, xmin, xmax, ymin, ymax, zmin, zmax, use_adjusted_pos)

    # Export
    def fill_hdf5_table(self, parent_group: h5py.Group) -> None:
        """Populates an HDF5 group with array-storable per-grain data.
        Create arrays for all array-storable parameters

        :param parent_group: Parent HDF5 group that datasets will be made in
        :raises AttributeError: If an essential grain attribute is missing from a grain
        """

        # Do all int attributes:
        for attribute in all_int_attributes:
            # Will still fail if the attribute is missing completely
            try:
                attribute_array = self.get_attribute_array_for_all_grains(attribute=attribute, must_be_complete=True)
                parent_group.create_dataset(attribute, data=attribute_array, dtype=int)
            except AttributeError as e:
                if attribute in required_obj_grain_attribute_names_int:
                    raise AttributeError(f"Missing attribute {attribute}, cannot export")
                else:
                    # print(e)
                    # print(f"Skipping missing attribute {attribute}")
                    continue

        # Do all bool attributes:
        for attribute in all_bool_attributes:
            # Will still fail if the attribute is missing completely
            try:
                attribute_array = self.get_attribute_array_for_all_grains(attribute=attribute, must_be_complete=True)
                parent_group.create_dataset(attribute, data=attribute_array, dtype=bool)
            except AttributeError as e:
                if attribute in required_obj_grain_attribute_names_bool:
                    raise AttributeError(f"Missing attribute {attribute}, cannot export")
                else:
                    # print(e)
                    # print(f"Skipping missing attribute {attribute}")
                    continue

        # Do all 1d attributes:
        for attribute in all_1d_float_attributes:
            # Will still fail if the attribute is missing completely
            try:
                attribute_array = self.get_attribute_array_for_all_grains(attribute=attribute, must_be_complete=True)
                parent_group.create_dataset(attribute, data=attribute_array, dtype=float)
            except AttributeError as e:
                if attribute in required_obj_grain_attribute_names_1d_float:
                    raise AttributeError(f"Missing attribute {attribute}, cannot export")
                else:
                    # print(e)
                    # print(f"Skipping missing attribute {attribute}")
                    continue

        # Do all 3x3 attributes:
        for attribute in all_3x3_float_attributes:
            # Will still fail if the attribute is missing completely
            try:
                attribute_array = self.get_attribute_array_for_all_grains(attribute=attribute, must_be_complete=True)
                attribute_array_flattened = attribute_array.reshape((attribute_array.shape[0], 9))

                parent_group.create_dataset(attribute, data=attribute_array_flattened, dtype=float)
            except AttributeError as e:
                if attribute in required_obj_grain_attribute_names_3x3_float:
                    raise AttributeError(f"Missing attribute {attribute}, cannot export")
                else:
                    # print(e)
                    # print(f"Skipping missing attribute {attribute}")
                    continue

    def __repr__(self) -> str:
        return f"GrainCollection with {len(self.grains)} grains"

    def __len__(self) -> int:
        return len(self.grains)


class BaseGrainsMap(GrainsCollection[TBaseMapGrain], Generic[TBaseMapGrain]):
    """Extends the :class:`~.GrainsCollection` class with
    :class:`~py3DXRDProc.phase.Phase` and :class:`~py3DXRDProc.grain_volume.GrainVolume` properties.
    Should not be used directly, this is used as a common base
    for :class:`~.RawGrainsMap`, :class:`~.CleanedGrainsMap`,
    :class:`~.StitchedGrainsMap` and :class:`~.TrackedGrainsMap` classes to avoid code duplication.

    :param grain_volume: The :class:`~py3DXRDProc.grain_volume.GrainVolume` to associate with this map
    :param phase: The :class:`~py3DXRDProc.phase.Phase` to associate with this map
    :param grains_list: List of grains to initialise the :class:`~.BaseGrainsMap` with, defaults to `None`
    :raises TypeError: If `grain_volume` is not a :class:`~py3DXRDProc.grain_volume.GrainVolume` instance
    :raises TypeError: If `phase` is not a :class:`~py3DXRDProc.phase.Phase` instance
    :raises TypeError: If any grain in `grains_list` is not a :class:`~py3DXRDProc.grain.BaseMapGrain` instance
    """

    def __init__(self, grain_volume: GrainVolume | StitchedGrainVolume,
                 phase: Phase,
                 grains_list: Optional[List[TBaseMapGrain]] = None):
        # Initialise an empty RawGrainsMap object
        if grains_list is not None:
            self.validate_grains_list(grains_list)
        super().__init__(grains_list)

        if not grain_volume.__class__.__name__ in ["GrainVolume", "StitchedGrainVolume"]:
            raise TypeError("GrainVolume should be a GrainVolume or StitchedGrainVolume instance!")

        self._grain_volume = grain_volume

        if not isinstance(phase, Phase):
            raise TypeError("Phase should be a Phase instance!")
        self._phase = phase

    @staticmethod
    def validate_grains_list(grains_list: List[TBaseMapGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.BaseMapGrain` (or subclass) instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.BaseMapGrain` (or subclass) instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.BaseMapGrain` (or subclass) instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, BaseMapGrain):
                raise TypeError("All grains in grains_list must be a BaseMapGrain!")

    def add_grain(self, grain: TBaseMapGrain) -> None:
        """Add a single grain to the :class:`~.GrainsCollection`.
        If the :class:`~.GrainsCollection` is empty, sets the grains property to just this grain.

        :param grain: The grain we want to add
        :raises TypeError: If `grain` is not a subclass of :class:`~py3DXRDProc.grain.BaseGrain`
        :raises ValueError: If `grain` was already found in the collection
        """

        if not isinstance(grain, BaseMapGrain):
            raise TypeError("Grain you added is not a BaseMapGrain instance!")
        super().add_grain(grain)

    def remove_grain(self, grain: TBaseMapGrain) -> None:
        """Remove a grain from the :class:`~.GrainsCollection`.

        :param grain: The grain to remove from the map
        :raises TypeError: If `grain` is not a :class:`~py3DXRDProc.grain.BaseGrain` instance
        :raises KeyError: If `grain` not found in the :class:`~.GrainsCollection`
        """

        if not isinstance(grain, BaseMapGrain):
            raise TypeError("Grain should be a BaseMapGrain instance!")
        super().remove_grain(grain)

    @property
    def phase(self) -> Phase:
        """The grain :class:`~py3DXRDProc.phase.Phase` for all the grains in this map

        :return: The :class:`~py3DXRDProc.phase.Phase` for all the grains in this map
        """

        return self._phase

    @property
    def grain_volume(self) -> GrainVolume | StitchedGrainVolume:
        """The grain :class:`~py3DXRDProc.grain_volume.GrainVolume` this map belongs to

        :return: The grain :class:`~py3DXRDProc.grain_volume.GrainVolume` this map belongs to
        """

        return self._grain_volume

    @property
    def load_step(self) -> LoadStep:
        """The grain :class:`~py3DXRDProc.load_step.LoadStep` from the :attr:`~.BaseGrainsMap.grain_volume`

        :return: The grain :class:`~py3DXRDProc.load_step.LoadStep`
        """

        return self.grain_volume.load_step

    @property
    def sample(self) -> Sample:
        """The grain :class:`~py3DXRDProc.sample.Sample` from the :attr:`~.BaseGrainsMap.load_step`

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        return self.load_step.sample

    @property
    def name(self) -> str:
        """The name of this grain map, automatically generated
        from the :attr:`~.BaseGrainsMap.sample`, :attr:`~.BaseGrainsMap.load_step`, :attr:`~.BaseGrainsMap.grain_volume`,
        :attr:`~.BaseGrainsMap.phase`

        :return: The name of the grain map
        """

        return f"{self.sample.name}:{self.load_step.name}:{self.grain_volume.name}:{self.phase.name}"

    @property
    def all_grains_have_errors(self) -> bool:
        """Returns `True` if all grains in this map have errors, `False` otherwise

        :return: `True` if all grains in this map have errors, `False` otherwise
        """

        return all([grain.has_errors for grain in self.grains])

    @property
    def some_grains_have_errors(self) -> bool:
        """Returns `True` if only some grains in this map have errors, `False` otherwise

        :return: `True` if some in this map have errors, `False` otherwise
        """

        return any([grain.has_errors for grain in self.grains])

    @property
    def all_pos_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = self.get_attribute_array_for_all_grains(attribute="pos_error", must_be_complete=True)
        return pos_error_array

    @property
    def all_eps_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = self.get_attribute_array_for_all_grains(attribute="eps_error", must_be_complete=True)
        return eps_error_array

    @property
    def all_eps_lab_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = self.get_attribute_array_for_all_grains(attribute="eps_lab_error", must_be_complete=True)
        return eps_lab_error_array

    @property
    def all_U_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = self.get_attribute_array_for_all_grains(attribute="U_error", must_be_complete=True)
        return U_error_array

    @property
    def all_angle_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = self.get_attribute_array_for_all_grains(attribute="angle_error", must_be_complete=True)
        return angle_error_array

    @property
    def all_sig_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = self.get_attribute_array_for_all_grains(attribute="sig_error", must_be_complete=True)
        return sig_error_array

    @property
    def all_sig_lab_errors(self) -> npt.NDArray[np.float64]:
        """Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all grains

        :return: Array of :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = self.get_attribute_array_for_all_grains(attribute="sig_lab_error", must_be_complete=True)
        return sig_lab_error_array

    @property
    def average_pos_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.pos_error`
        """

        pos_error_array = self.all_pos_errors
        return np.mean(pos_error_array, axis=0)

    @property
    def average_eps_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_error`
        """

        eps_error_array = self.all_eps_errors
        return np.mean(eps_error_array, axis=0)

    @property
    def average_eps_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.eps_lab_error`
        """

        eps_lab_error_array = self.all_eps_lab_errors
        return np.mean(eps_lab_error_array, axis=0)

    @property
    def average_U_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.U_error`
        """

        U_error_array = self.all_U_errors
        return np.mean(U_error_array, axis=0)

    @property
    def average_angle_error(self) -> float:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.angle_error`
        """

        angle_error_array = self.all_angle_errors
        return float(np.mean(angle_error_array, axis=0))

    @property
    def average_sig_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_error`
        """

        sig_error_array = self.all_sig_errors
        return np.mean(sig_error_array, axis=0)

    @property
    def average_sig_lab_error(self) -> npt.NDArray[np.float64]:
        """Average :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error` of all grains

        :return: Mean :attr:`~py3DXRDProc.grain.BaseMapGrain.sig_lab_error`
        """

        sig_lab_error_array = self.all_sig_lab_errors
        return np.mean(sig_lab_error_array, axis=0)

    def get_grain_from_gid(self, gid: int) -> TBaseMapGrain:
        """Looks for a grain by its :attr:`~py3DXRDProc.grain.BaseMapGrain.gid`

        :param gid: Grain :attr:`~py3DXRDProc.grain.BaseMapGrain.gid` to look for
        :raises TypeError: If the grain :attr:`~py3DXRDProc.grain.BaseMapGrain.gid` is not an `int`
        :raises KeyError: If no grain was found with that :attr:`~py3DXRDProc.grain.BaseMapGrain.gid`
        :return: The grain if it was found
        """

        if not isinstance(gid, int):
            raise TypeError("Wrong type supplied for gid!")
        for grain in self.grains:
            if grain.gid == gid:
                return grain
        raise KeyError("Grain not found")

    def get_grain_from_immutable_string(self, immutable_string: str) -> TBaseMapGrain:
        """Looks for a grain by its immutable string

        :param immutable_string: The immutable string to look for
        :raises TypeError: If the immutable string is not a string
        :raises ValueError: If the immutable string is empty
        :raises KeyError: If the grain was not found
        :return: The grain if it was found
        """

        if not isinstance(immutable_string, str):
            raise TypeError("Immutable string must be a string!")
        if immutable_string == "":
            raise ValueError("Immutable string must not be empty!")
        for grain in self.grains:
            if grain.immutable_string == immutable_string:
                return grain
        raise KeyError("Grain not found")

    @staticmethod
    def get_right_map_from_list_by_name(grain_maps_list: List[TBaseGrainsMap], map_name: str) -> TBaseGrainsMap:
        """Given a list of grain maps and a grain map name, returns a matching grain map if it was found.

        :param grain_maps_list: List of :class:`~py3DXRDProc.grain_map.BaseGrainsMap` to look for a match in
        :param map_name: Name of the map to look for
        :raises TypeError: If the grain maps list isn't a ``list``
        :raises TypeError: If not all grain maps in the list are a :class:`~py3DXRDProc.grain_map.BaseGrainsMap` or subclass
        :raises TypeError: If the map name is not a ``str``
        :raises ValueError: If the map name string is empty
        :raises KeyError: If no map was found with that name
        :return: The grain map if one was found
        """

        if not isinstance(grain_maps_list, list):
            raise TypeError("grain_maps_list must be a list type!")
        for grain_map in grain_maps_list:
            if not isinstance(grain_map, BaseGrainsMap):
                raise TypeError("All maps in grain_maps_list must be a BaseGrainsMap or subclass")
        if not isinstance(map_name, str):
            raise TypeError("Map name should be a string!")
        if map_name == "":
            raise ValueError("Map name cannot be empty!")
        for grain_map in grain_maps_list:
            if grain_map.name == map_name:
                return grain_map
        raise KeyError("Grain map with that name not found in the list!")

    def export_to_neper(self) -> None:
        """Exports the grain positions, volumes, radii, orientations
        to the "neper" subdirectory under the processing directory.
        Prints out the required Neper command to tessellate this grain map.
        """

        output_directory = os.path.join(self.sample.pars.directories.processing, self.sample.name, "neper")
        make_folder(output_directory)

        pos_file_path = os.path.join(output_directory, "positions.txt")
        radii_file_path = os.path.join(output_directory, "radii.txt")
        oriens_file_path = os.path.join(output_directory, "oriens.txt")
        planes_file_path = os.path.join(output_directory, "planes.txt")

        n_grains = len(self.grains)

        pos_array = self.get_attribute_array_for_all_grains("pos_offset", must_be_complete=True)
        volume_array = self.get_attribute_array_for_all_grains("volume", must_be_complete=True)
        radii_array = np.power(volume_array * 3 / (4 * np.pi), 1 / 3)
        oriens_array = self.get_attribute_array_for_all_grains("U", must_be_complete=True).reshape(n_grains, 9)

        np.savetxt(pos_file_path, pos_array, fmt="%.6f")
        np.savetxt(radii_file_path, radii_array, fmt="%.6f")
        np.savetxt(oriens_file_path, oriens_array, fmt="%.6f")

        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.grain_volume.index_dimensions

        # Determine the planes of the bounding box
        xmin_plane = np.array([xmin - 0.1, -1, 0, 0])
        ymin_plane = np.array([ymin - 0.1, 0, -1, 0])
        zmin_plane = np.array([zmin - 0.1, 0, 0, -1])
        xmax_plane = np.array([xmax + 0.1, 1, 0, 0])
        ymax_plane = np.array([ymax + 0.1, 0, 1, 0])
        zmax_plane = np.array([zmax + 0.1, 0, 0, 1])
        planes_array = np.stack([xmin_plane, ymin_plane, zmin_plane, xmax_plane, ymax_plane, zmax_plane])
        np.savetxt(planes_file_path, planes_array, fmt="%.6f")

        x_size = xmax - xmin + 0.2
        y_size = ymax - ymin + 0.2
        z_size = zmax - zmin + 0.2

        with open(planes_file_path, 'r') as original:
            data = original.read()
        with open(planes_file_path, 'w') as modified:
            modified.write("6\n" + data)

        # Find out how many CPUs we have
        if self.sample.pars.processing_facility == "DLS":
            ncpu = int(os.getenv('NSLOTS'))
        else:
            ncpu = int(os.getenv('SLURM_NTASKS'))

        log.info("Run the following Neper command to tessellate:")
        log.info(
            f'OMP_NUM_THREADS={ncpu} neper -T -n {n_grains} -domain "cube({x_size},{y_size},{z_size}):translate({-x_size / 2},{-y_size / 2},{-z_size / 2})" -morphooptiini "coo:file({pos_file_path}),weight:file({radii_file_path})" -orioptiini "file({oriens_file_path}),des=rotmat" -o {os.path.join(output_directory, "tess_output")} --morphooptiobjective tesr:pts(res=50) -format tess,tesr')

    def export_grain_list_to_gff(self, grains_list: List[TBaseMapGrain], gff_path: str,
                                 use_adjusted_position: bool) -> None:
        """Export a grain list to a gff file. Note: this does nothing to rectify repeating grain IDs.

        :param grains_list: List of grains to export
        :param gff_path: Path to gff file to create
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        :raises TypeError: If gff path not a string
        """

        self.validate_grains_list(grains_list)
        self.export_validated_grain_list_to_gff(grains_list=grains_list,
                                                gff_path=gff_path,
                                                use_adjusted_position=use_adjusted_position)

    def export_to_gff(self, gff_path: str, use_adjusted_position: bool) -> None:
        """Exports all grains to a gff file.

        :param gff_path: Path to gff file to create
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        """

        self.export_grain_list_to_gff(grains_list=self.grains,
                                      gff_path=gff_path,
                                      use_adjusted_position=use_adjusted_position)

    def __repr__(self) -> str:
        return f"BaseGrainsMap {self.name} with {len(self.grains)} grains"

    def __len__(self) -> int:
        return len(self.grains)


TBaseGrainsMap = TypeVar('TBaseGrainsMap', bound=BaseGrainsMap)


class RawGrainsMap(BaseGrainsMap[RawGrain]):
    """Class to describe a :class:`~.BaseGrainsMap` that was produced by :mod:`ImageD11`.

    :param grain_volume: The :class:`~py3DXRDProc.grain_volume.GrainVolume` to associate with this map
    :param phase: The :class:`~py3DXRDProc.phase.Phase` to associate with this map
    :param grains_list: List of grains to initialise the :class:`~.RawGrainsMap` with, defaults to `None`
    """

    def __init__(self, grain_volume: GrainVolume,
                 phase: Phase,
                 grains_list: Optional[List[RawGrain]] = None):
        if grains_list is not None:
            self.validate_grains_list(grains_list)
        if not grain_volume.__class__.__name__ == "GrainVolume":
            raise TypeError("grain_volume must be a GrainVolume instance!")
        super().__init__(grain_volume=grain_volume,
                         phase=phase,
                         grains_list=grains_list)

    @property
    def name(self) -> str:
        """The name of this grain map, automatically generated
        from the :attr:`~.BaseGrainsMap.sample`, :attr:`~.BaseGrainsMap.load_step`, :attr:`~.BaseGrainsMap.grain_volume`,
        :attr:`~.BaseGrainsMap.phase` with "raw" added to the end

        :return: The name of the grain map
        """

        return f"{self.sample.name}:{self.load_step.name}:{self.grain_volume.name}:{self.phase.name}:raw"

    # Add grains
    def add_grain(self, grain: RawGrain) -> None:
        """Add a single grain to the :class:`~.RawGrainsMap`.

        :param grain: The grain we want to add
        :raises TypeError: If `grain` is not a subclass of :class:`~py3DXRDProc.grain.RawGrainsMap`
        """

        if not isinstance(grain, RawGrain):
            raise TypeError("Must be a RawGrain!")
        super().add_grain(grain)

    # Remove grains

    def remove_grain(self, grain: RawGrain) -> None:
        """Remove a grain from the :class:`~.RawGrainsMap`.

        :param grain: The grain to remove from the map
        :raises TypeError: If `grain` is not a :class:`~py3DXRDProc.grain.RawGrain` instance
        """

        if not isinstance(grain, RawGrain):
            raise TypeError("Must be a RawGrain!")
        super().remove_grain(grain)

    @staticmethod
    def validate_grains_list(grains_list: List[RawGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.RawGrain` (or subclass) instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.RawGrain` (or subclass) instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.RawGrain` (or subclass) instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, RawGrain):
                raise TypeError("All grains in grains_list must be a RawGrain!")

    # Imports
    @classmethod
    def import_from_map(cls, map_path: str,
                        phase: Phase,
                        grain_volume: GrainVolume,
                        errors_folder: Optional[str] = None) -> Optional[RawGrainsMap]:
        """Read in grains from an :mod:`ImageD11` map file, returns a :class:`~.RawGrainsMap` object.

        :param map_path: Path to the map file
        :param phase: Phase of the grain map
        :param grain_volume: :class:`~py3DXRDProc.grain_volume.GrainVolume` instance
        :param errors_folder: Path to folder containing errors from bootstrap method, defaults to `None`
        :raises TypeError: If map path is not a ``str``
        :raises FileNotFoundError: If map path invalid
        :raises TypeError: If errors folder is not a ``str``
        :raises FileNotFoundError: If `errors_folder` is not `None` but errors folder path not found
        :raises FileNotFoundError: If individual grain errors folder is not found
        :raises FileNotFoundError: If individual grain errors file path is not found
        :return grain_map: :class:`~.RawGrainsMap` instance with grains in it.
        """

        if not isinstance(map_path, str):
            raise TypeError("Map path must be string!")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Invalid map path provided: {map_path}")
        if errors_folder is not None:
            if not isinstance(errors_folder, str):
                raise TypeError("errors_folder should be a string")
            if not os.path.exists(errors_folder):
                raise FileNotFoundError("errors_folder path not found!")

        # grain_volume and phase are checked here:
        grain_map = RawGrainsMap(grain_volume=grain_volume, phase=phase)

        # Import into ImageD11
        from ImageD11.grain import read_grain_file
        grains_list = read_grain_file(map_path)

        if len(grains_list) == 0:
            log.warning(
                f"Found empty map at load step {grain_volume.load_step.name}, volume {grain_volume.name}, phase {phase.name}")
            return None

        # Filter out grains that have no peaks assigned
        grains_list = [grain for grain in grains_list if grain.intensity_info != 'no peaks\n']

        phase_sequence = grain_volume.sample.pars.phases.names
        phase_index = phase_sequence.index(phase.name)

        # Filter out grains that have fewer peaks than nuniq
        grains_list = [grain for grain in grains_list if int(grain.nuniq) >= grain_volume.sample.pars.makemap.minpeaks[phase_index]]

        # Make our essential arrays
        id_array = [int(grain.name.split(":")[0]) for grain in grains_list]
        pos_array = np.array([grain.translation / 1000 for grain in grains_list])  # Convert from um to mm
        UBI_array = np.stack([grain.ubi for grain in grains_list], axis=0)

        mean_peak_intensity_array = np.array(
            [float(grain.intensity_info.split("mean = ")[1].split(" , ")[0]) for grain in grains_list])

        mean_peak_intensity_sum = np.sum(mean_peak_intensity_array)
        fractional_peak_intensity_array = mean_peak_intensity_array / mean_peak_intensity_sum

        grain_map.volume = grain_volume.material_volume  # In cubic mm

        grainvolume_array = fractional_peak_intensity_array * grain_map.volume

        grains_to_add = [RawGrain(gid=gid,
                                  pos=pos_array[index],
                                  UBI=UBI_array[index],
                                  grain_map=grain_map,
                                  volume=float(grainvolume_array[index]),
                                  mean_peak_intensity=mean_peak_intensity_array[index]) for index, gid in
                         enumerate(id_array)]

        grain_map.add_grains(grains_to_add)

        if errors_folder is not None:
            for grain in grain_map.grains:
                grain_errors_folder = os.path.join(errors_folder, f"{grain.gid}")
                if not os.path.exists(grain_errors_folder):
                    raise FileNotFoundError(f"Errors folder for grain {grain.gid} not found")
                else:
                    grain_error_files = os.listdir(grain_errors_folder)
                    grain_error_filenames = [os.path.basename(file_path) for file_path in grain_error_files]
                    for expected_filename in ["pos_errors.txt", "eps_errors.txt", "u_errors.txt",
                                              "angle_error.txt"]:
                        if expected_filename not in grain_error_filenames:
                            raise FileNotFoundError(
                                f"{expected_filename} not found in error folder for grain {grain.gid}")

                    grain_pos_error = np.loadtxt(os.path.join(grain_errors_folder,
                                                              "pos_errors.txt")) / 1000  # pos_error is in microns by default coming out of ImageD11
                    grain_eps_error = np.loadtxt(os.path.join(grain_errors_folder,
                                                              "eps_errors.txt"))
                    try:
                        grain_eps_lab_error = np.loadtxt(os.path.join(grain_errors_folder,
                                                                      "eps_lab_errors.txt"))
                    except FileNotFoundError:
                        # use old filename
                        grain_eps_lab_error = np.loadtxt(os.path.join(grain_errors_folder,
                                                                      "eps_s_errors.txt"))
                    grain_U_error = np.loadtxt(os.path.join(grain_errors_folder,
                                                            "u_errors.txt"))
                    grain_angle_error = float(np.loadtxt(os.path.join(grain_errors_folder,
                                                                      "angle_error.txt")))

                    grain.add_errors(pos_error=grain_pos_error,
                                     eps_error=grain_eps_error,
                                     eps_lab_error=grain_eps_lab_error,
                                     U_error=grain_U_error,
                                     angle_error=grain_angle_error)

        # Check all U matrices
        grains_to_remove = []
        for grain in grain_map.grains:
            if not np.allclose(np.dot(grain.U.T, grain.U), np.eye(3, 3)):
                grains_to_remove.append(grain)

            if not np.allclose(np.linalg.det(grain.U), 1.0):
                grains_to_remove.append(grain)
        if not len(grains_to_remove) == 0:
            log.warning(f"Removing {len(grains_to_remove)} grains due to dodgy U matrices")
            grain_map.remove_grains(grains_to_remove)

        return grain_map

    @classmethod
    def import_from_hdf5_group(cls, map_group: h5py.Group,
                               grain_volume_object: GrainVolume,
                               phase_object: Phase) -> RawGrainsMap:
        """Import a :class:`~.RawGrainsMap` from a :class:`h5py.Group` 
        
        :param map_group: :class:`h5py.Group` containing the grains data
        :param grain_volume_object: :class:`~py3DXRDProc.grain_volume.GrainVolume` object that this map belongs to
        :param phase_object: The :class:`~py3DXRDProc.phase.Phase` to associate with this map
        :raises TypeError: If `map_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `grain_volume_object` is not a :class:`~py3DXRDProc.grain_volume.GrainVolume` instance
        :raises TypeError: If `phase_object` is not a :class:`~py3DXRDProc.phase.Phase` instance
        :raises ValueError: If an essential field is missing from the HDF5 dataset
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the UBI array has the wrong shape
        :raises ValueError: If the volume array has the wrong shape
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the grains have a mixture of load step names
        :raises ValueError: If the grains have a different load step name from the one attached to the `grain_volume_object`
        :return: The :class:`~.RawGrainsMap` created from the HDF5 group
        """

        if not isinstance(map_group, h5py.Group):
            raise TypeError("map_group should be an h5py Group instance")
        if not grain_volume_object.__class__.__name__ == "GrainVolume":
            raise TypeError("grain_volume_object should be a GrainVolume instance!")
        if not isinstance(phase_object, Phase):
            raise TypeError("phase_object should be a Phase instance!")
        # We should have much more input validation here
        map_object = RawGrainsMap(grain_volume=grain_volume_object, phase=phase_object)

        # Ensure required fields are there

        essential_columns = required_obj_grain_attribute_names + ["immutable_string", "mean_peak_intensity"]
        map_keys = map_group.keys()
        for column in essential_columns:
            if column not in map_keys:
                raise ValueError(f"Essential field {column} missing from HDF5 dataset! Cannot continue")

        # Set up barebones grains list
        immutable_string_array = map_group["immutable_string"][:]

        n_grains = len(immutable_string_array)

        pos_array = map_group["pos"][:]
        if np.shape(pos_array) != (n_grains, 3):
            raise ValueError("Pos array has wrong shape!")

        UBI_array = map_group["UBI"][:].reshape((n_grains, 3, 3))
        if np.shape(UBI_array) != (n_grains, 3, 3):
            raise ValueError("UBI array has wrong shape!")

        volume_array = map_group["volume"][:]
        if np.shape(volume_array) != (n_grains,):
            raise ValueError("volume array has wrong shape!")

        mean_peak_intensity_array = map_group["mean_peak_intensity"][:]
        if np.shape(mean_peak_intensity_array) != (n_grains,):
            raise ValueError("mean_peak_intensity_array array has wrong shape!")

        gid_list = map_group["gid"][:]

        for index, immut_string in enumerate(immutable_string_array):
            grain_map_name, gid = immut_string.decode("utf-8").rsplit(":", maxsplit=1)
            sample_name, load_step_name, grain_volume_name, phase_name, map_type = grain_map_name.split(
                ":")
            if index == 0:
                first_load_step_name = load_step_name
            else:
                if load_step_name != first_load_step_name:
                    raise ValueError("Not all load step names are equal!")
            if load_step_name != grain_volume_object.load_step.name:
                raise ValueError("Load step object name and its name in the immutable ID list don't match")

        # Establish a list of grain objects, initialised as barebones as possible
        grains_to_add = [RawGrain(gid=gid,
                                  pos=pos_array[index],
                                  UBI=UBI_array[index],
                                  volume=volume_array[index],
                                  grain_map=map_object,
                                  mean_peak_intensity=mean_peak_intensity_array[index]) for index, gid in
                         enumerate(gid_list)]

        if "pos_error" in map_keys:
            pos_error_array = map_group["pos_error"][:]
            eps_error_array = map_group["eps_error"][:]
            eps_lab_error_array = map_group["eps_lab_error"][:]
            U_error_array = map_group["U_error"][:]
            angle_error_array = map_group["angle_error"][:]
            for grain, pos_error, eps_error, eps_lab_error, U_error, angle_error in zip(grains_to_add, pos_error_array,
                                                                                        eps_error_array,
                                                                                        eps_lab_error_array,
                                                                                        U_error_array,
                                                                                        angle_error_array):
                eps_error_reshaped = eps_error.reshape(3, 3)
                eps_lab_error_reshaped = eps_lab_error.reshape(3, 3)
                U_error_reshaped = U_error.reshape(3, 3)
                grain.add_errors(pos_error=pos_error,
                                 eps_error=eps_error_reshaped,
                                 eps_lab_error=eps_lab_error_reshaped,
                                 U_error=U_error_reshaped,
                                 angle_error=angle_error)

        # Add the grains to the RawGrainsMap object, so we can use grain_volume methods for adding further attributes
        map_object.add_grains(grains_to_add)

        # Import array data
        # 1d attributes (data types should be inferred from the HDF5)
        for attribute in optional_obj_grain_attribute_names_flat:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:]
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        # 3x3 arrays
        for attribute in optional_obj_grain_attribute_names_3x3_float:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:].reshape((n_grains, 3, 3))
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        return map_object

    def export_to_hdf5_group(self, this_grain_maps_group: h5py.Group) -> h5py.Group:
        """Export a :class:`~.RawGrainsMap` to a :class:`h5py.Group`

        :param this_grain_maps_group: The :class:`h5py.Group` to export this map to
        :return: The :class:`h5py.Group` with the grain data filled in
        """

        this_map_group = this_grain_maps_group.create_group(self.phase.name)

        # Create an array of immutable strings
        immutable_string_array = self.get_attribute_array_for_all_grains(attribute="immutable_string",
                                                                         must_be_complete=True).astype("S256")
        this_map_group.create_dataset("immutable_string", data=immutable_string_array, dtype="S256")

        # required only for Raw Grains:

        mean_peak_intensity_array = self.get_attribute_array_for_all_grains(attribute="mean_peak_intensity",
                                                                            must_be_complete=True)
        this_map_group.create_dataset("mean_peak_intensity", data=mean_peak_intensity_array, dtype=float)

        # Create arrays for all other array-storable parameters

        self.fill_hdf5_table(this_map_group)

        return this_map_group

    def __repr__(self) -> str:
        return f"RawGrainsMap {self.name} with {len(self.grains)} grains"


class CleanedGrainsMap(BaseGrainsMap[CleanGrain]):
    """Class to describe a :class:`~.CleanedGrainsMap` that was produced by de-duplicating a :class:`~.RawGrainsMap`

    :param raw_map: The :class:`~.RawGrainsMap` that was de-duplicated to create this map
    :param grains_list: List of grains to initialise the :class:`~.BaseGrainsMap` with, defaults to `None`
    :raises TypeError: If the `raw_map` is not a :class:`~.RawGrainsMap` instance
    :raises ValueError: If the :attr:`.RawGrainsMap.grain_volume` and the `grain_volume` are different
    """

    def __init__(self, raw_map: RawGrainsMap,
                 grains_list: Optional[List[CleanGrain]] = None):

        if not isinstance(raw_map, RawGrainsMap):
            raise TypeError("Raw map should be a RawGrainsMap instance!")

        if grains_list is not None:
            self.validate_grains_list(grains_list)

        grain_volume = raw_map.grain_volume
        super().__init__(grain_volume=grain_volume,
                         phase=raw_map.phase,
                         grains_list=grains_list)

        if not raw_map.grain_volume == self.grain_volume:
            raise ValueError("Raw map must have the same grain volume!")
        self._raw_map = raw_map

    @property
    def raw_map(self) -> RawGrainsMap:
        return self._raw_map

    @property
    def name(self) -> str:
        """The name of this grain map, automatically generated
        from the :attr:`~.BaseGrainsMap.sample`, :attr:`~.BaseGrainsMap.load_step`, :attr:`~.BaseGrainsMap.grain_volume`,
        :attr:`~.BaseGrainsMap.phase` with "cleaned" added to the end

        :return: The name of the grain map
        """

        return f"{self.sample.name}:{self.load_step.name}:{self.grain_volume.name}:{self.phase.name}:cleaned"

    # Add grains
    def add_grain(self, grain: CleanGrain) -> None:
        """Add a single grain to the :class:`~.CleanedGrainsMap`.

        :param grain: The grain we want to add
        :raises TypeError: If `grain` is not a subclass of :class:`~py3DXRDProc.grain.CleanGrain`
        """

        if not isinstance(grain, CleanGrain):
            raise TypeError("Must be a CleanGrain!")
        super().add_grain(grain)

    # Remove grains

    def remove_grain(self, grain: CleanGrain) -> None:
        """Remove a grain from the :class:`~.CleanedGrainsMap`.

        :param grain: The grain to remove from the map
        :raises TypeError: If `grain` is not a :class:`~py3DXRDProc.grain.CleanGrain` instance
        """

        if not isinstance(grain, CleanGrain):
            raise TypeError("Must be a CleanGrain!")
        super().remove_grain(grain)

    @staticmethod
    def validate_grains_list(grains_list: List[CleanGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.CleanGrain` (or subclass) instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.CleanGrain` (or subclass) instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.CleanGrain` (or subclass) instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, CleanGrain):
                raise TypeError("All grains in grains_list must be a CleanGrain!")

    @classmethod
    def from_cleaning_grain_map(cls, input_grain_map: RawGrainsMap,
                                dist_tol: float = 0.1,
                                angle_tol: float = 1.0) -> CleanedGrainsMap:
        """Clean a :class:`~.RawGrainsMap` to create a :class:`~.CleanedGrainsMap` by de-duplicating the grains.
        Uses :func:`~py3DXRDProc.grain.find_all_grain_pair_matches_from_list` to look for duplicates in `input_grain_map`,
        using `dist_tol` and `angle_tol` to control the threshold for what is or isn't a `duplicate` grain.

        :param input_grain_map: The :class:`~.RawGrainsMap` to be de-duplicated
        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises TypeError: If `input_grain_map` is not a :class:`~.RawGrainsMap` instance
        :return: De-duplicated :class:`~.CleanedGrainsMap` in the same :attr:`~.RawGrainsMap.grain_volume` as the :class:`~.RawGrainsMap`
        """

        log.info(f"Cleaning grain map {input_grain_map.name}")
        if not isinstance(input_grain_map, RawGrainsMap):
            raise TypeError("Grain map to clean must be a RawGrainsMap instance")

        clean_grains_map_object = CleanedGrainsMap(raw_map=input_grain_map)

        grain_groups = find_multiple_observations(grains_list=input_grain_map.grains,
                                                  dist_tol=dist_tol,
                                                  angle_tol=angle_tol)
        for index, grain_group in enumerate(grain_groups):
            clean_grain = CleanGrain.from_grains_list(gid=index,
                                                      grains_to_merge=grain_group,
                                                      grain_map=clean_grains_map_object)
            clean_grains_map_object.add_grain(clean_grain)

        return clean_grains_map_object

    @classmethod
    def import_from_hdf5_group(cls, map_group: h5py.Group,
                               grain_volume_object: GrainVolume,
                               raw_map: RawGrainsMap) -> CleanedGrainsMap:
        """Import a :class:`~.CleanedGrainsMap` from a :class:`h5py.Group`

        :param map_group: :class:`h5py.Group` containing the grains data
        :param grain_volume_object: :class:`~py3DXRDProc.grain_volume.GrainVolume` object that this map belongs to
        :param raw_map: The :class:`~.RawGrainsMap` this :class:`~.CleanedGrainsMap` had deduplicated
        :raises TypeError: If `map_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `grain_volume_object` is not a :class:`~py3DXRDProc.grain_volume.GrainVolume` instance
        :raises TypeError: If `phase_object` is not a :class:`~py3DXRDProc.phase.Phase` instance
        :raises ValueError: If an essential field is missing from the HDF5 dataset
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the UBI array has the wrong shape
        :raises ValueError: If the volume array has the wrong shape
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the grains have a mixture of load step names
        :raises ValueError: If the grains have a different load step name from the one attached to the `grain_volume_object`
        :return: The :class:`~.CleanedGrainsMap` created from the HDF5 group
        """

        if not isinstance(map_group, h5py.Group):
            raise TypeError("map_group should be an h5py.Group instance")
        if not grain_volume_object.__class__.__name__ == "GrainVolume":
            raise TypeError("grain_volume_object should be a GrainVolume instance!")

        map_object = CleanedGrainsMap(raw_map=raw_map)

        # Ensure required fields are there

        essential_columns = required_obj_grain_attribute_names
        map_keys = map_group.keys()
        for column in essential_columns:
            if column not in map_keys:
                raise ValueError("Essential field missing from HDF5 dataset! Cannot continue")

        immutable_string_array = map_group["immutable_string"][:]

        # Set up barebones grains list
        gid_list = map_group["gid"][:]

        n_grains = len(gid_list)

        pos_array = map_group["pos"][:]
        if np.shape(pos_array) != (n_grains, 3):
            raise ValueError("Pos array has wrong shape!")

        UBI_array = map_group["UBI"][:].reshape((n_grains, 3, 3))
        if np.shape(UBI_array) != (n_grains, 3, 3):
            raise ValueError("UBI array has wrong shape!")

        volume_array = map_group["volume"][:]
        if np.shape(volume_array) != (n_grains,):
            raise ValueError("volume array has wrong shape!")

        # We have to know what the parent grains are
        parent_grains_for_each_grain = []
        # For each grain in our immutable ID list, get the grain data if it exists:
        per_grain_attributes_group = map_group["Per-Grain Data"]

        # Go through each grain in this map, get the parent map grain

        for index, immut_string in enumerate(immutable_string_array):
            this_grain_group = per_grain_attributes_group[immut_string.decode("utf-8")]

            parent_immut_strings_for_this_grain = this_grain_group["Parent Grain Immutable Strings"][:]
            parent_grains_for_this_grain = [raw_map.get_grain_from_immutable_string(parent_immut_string.decode("utf-8"))
                                            for parent_immut_string in parent_immut_strings_for_this_grain]
            parent_grains_for_each_grain.append(parent_grains_for_this_grain)

        # Establish a list of grain objects, initialised as barebones as possible
        grains_to_add = [CleanGrain(gid=gid,
                                    pos=pos_array[index],
                                    UBI=UBI_array[index],
                                    volume=volume_array[index],
                                    grain_map=map_object,
                                    parent_grains=parent_grains_for_each_grain[index]) for
                         index, gid in enumerate(gid_list)]

        if "pos_error" in map_keys:
            pos_error_array = map_group["pos_error"][:]
            eps_error_array = map_group["eps_error"][:]
            eps_lab_error_array = map_group["eps_lab_error"][:]
            U_error_array = map_group["U_error"][:]
            angle_error_array = map_group["angle_error"][:]
            for grain, pos_error, eps_error, eps_lab_error, U_error, angle_error in zip(grains_to_add, pos_error_array,
                                                                                        eps_error_array,
                                                                                        eps_lab_error_array,
                                                                                        U_error_array,
                                                                                        angle_error_array):
                eps_error_reshaped = eps_error.reshape(3, 3)
                eps_lab_error_reshaped = eps_lab_error.reshape(3, 3)
                U_error_reshaped = U_error.reshape(3, 3)
                grain.add_errors(pos_error=pos_error,
                                 eps_error=eps_error_reshaped,
                                 eps_lab_error=eps_lab_error_reshaped,
                                 U_error=U_error_reshaped,
                                 angle_error=angle_error)

        # Add the grains to the grain_volume object so we can use grain_volume methods for adding further attributes
        map_object.add_grains(grains_to_add)

        # Import array data
        # 1d attributes (data types should be inferred from the HDF5)
        for attribute in optional_obj_grain_attribute_names_flat:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:]
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        # 3x3 arrays
        for attribute in optional_obj_grain_attribute_names_3x3_float:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:].reshape((n_grains, 3, 3))
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        return map_object

    def export_to_hdf5_group(self, this_grain_maps_group: h5py.Group) -> h5py.Group:
        """Export the :class:`~.CleanedGrainsMap` to a :class:`h5py.Group`

        :param this_grain_maps_group: The :class:`h5py.Group` to export this map to
        :return: The :class:`h5py.Group` with the grain data filled in
        """

        # Array-able attributes and non-array-able attributes
        this_map_group = this_grain_maps_group.create_group(self.phase.name)

        # Create an array of immutable strings
        immutable_string_array = self.get_attribute_array_for_all_grains(attribute="immutable_string",
                                                                         must_be_complete=True).astype("S256")
        this_map_group.create_dataset("immutable_string", data=immutable_string_array, dtype="S256")

        # Create arrays for all other array-storable parameters

        self.fill_hdf5_table(this_map_group)

        per_grain_attributes_group = this_map_group.create_group("Per-Grain Data")

        # Every grain is going to have a parent here

        for grain in self.grains:
            # Make a group for each grain with the grain immutable string as the name
            this_grain_group = per_grain_attributes_group.create_group(grain.immutable_string)

            parent_immutable_strings = grain.parent_grain_immutable_strings

            parent_immutable_strings_array = np.array(parent_immutable_strings, dtype="S256")

            this_grain_group.create_dataset("Parent Grain Immutable Strings", data=parent_immutable_strings_array,
                                            dtype="S256")

        return this_map_group

    def __repr__(self) -> str:
        return f"CleanedGrainsMap {self.name} with {len(self.grains)} grains"


class StitchedGrainsMap(BaseGrainsMap[StitchedGrain]):
    """Class to describe a :class:`~.StitchedGrainsMap` that was produced by stitching together a list of :class:`~.CleanedGrainsMap`

    :param grain_volume: The :class:`~py3DXRDProc.grain_volume.GrainVolume` to associate with this map
    :param clean_maps_list: The list of :class:`~.CleanedGrainsMap` used to create this :class:`~.StitchedGrainsMap`
    :param grains_list: List of grains to initialise the :class:`~.BaseGrainsMap` with, defaults to `None`
    :raises TypeError: If not all the grains in `grains_list` are :class:`~py3DXRDProc.grain.StitchedGrain`
    :raises TypeError: If `clean_maps_list` is not a ``list`` instance
    :raises ValueError: If the :attr:`.CleanedGrainsMap.load_step` and the `grain_volume` load_step are different
    :raises TypeError: If any maps in the `clean_maps_list` are not a :class:`~.CleanedGrainsMap` instance
    :raises ValueError: If not all maps in the `clean_maps_list` have the same :attr:`~.CleanedGrainsMap.phase`
    """

    def __init__(self, grain_volume: StitchedGrainVolume,
                 clean_maps_list: List[CleanedGrainsMap],
                 grains_list: Optional[List[StitchedGrain]] = None):
        if grains_list is not None:
            self.validate_grains_list(grains_list)
        if not isinstance(clean_maps_list, list):
            raise TypeError("Clean maps list should be a list instance!")
        for clean_map in clean_maps_list:
            if clean_map.load_step != grain_volume.load_step:
                raise ValueError("Clean map and self have different load steps!")
            if not isinstance(clean_map, CleanedGrainsMap):
                raise TypeError("Each clean map in the list must be a CleanedGrainsMap instance!")
        first_clean_map_phase = clean_maps_list[0].phase
        for clean_map in clean_maps_list:
            if clean_map.phase != first_clean_map_phase:
                raise ValueError("All maps in clean_maps_list must have the same phase!")

        super().__init__(grain_volume=grain_volume,
                         phase=first_clean_map_phase,
                         grains_list=grains_list)

        self._stitched_maps_list = clean_maps_list

    @property
    def name(self) -> str:
        """The name of this grain map, automatically generated
        from the :attr:`~.BaseGrainsMap.sample`, :attr:`~.BaseGrainsMap.load_step`, :attr:`~.BaseGrainsMap.grain_volume`,
        :attr:`~.BaseGrainsMap.phase`

        :return: The name of the grain map
        """

        return f"{self.sample.name}:{self.load_step.name}:{self.grain_volume.name}:{self.phase.name}:stitched"

    @property
    def all_contrib_maps(self):
        return self._stitched_maps_list

    @property
    def all_contrib_map_names(self):
        return [a_map.name for a_map in self.all_contrib_maps]

    @property
    def grains(self) -> List[StitchedGrain]:
        """List of grains in the :class:`~.StitchedGrainsMap`

        :return: The list of grains in the :class:`~.StitchedGrainsMap`
        """

        return self._grains

    # Add grains
    def add_grain(self, grain: StitchedGrain) -> None:
        """Add a single grain to the :class:`~.StitchedGrainsMap`.

        :param grain: The grain we want to add
        :raises TypeError: If `grain` is not a subclass of :class:`~py3DXRDProc.grain.StitchedGrain`
        """

        if not isinstance(grain, StitchedGrain):
            raise TypeError("Must be a StitchedGrain!")
        super().add_grain(grain)

    # Remove grains

    def remove_grain(self, grain: StitchedGrain) -> None:
        """Remove a grain from the :class:`~.StitchedGrainsMap`.

        :param grain: The grain to remove from the map
        :raises TypeError: If `grain` is not a :class:`~py3DXRDProc.grain.StitchedGrain` instance
        """

        if not isinstance(grain, StitchedGrain):
            raise TypeError("Must be a StitchedGrain!")
        super().remove_grain(grain)

    @staticmethod
    def validate_grains_list(grains_list: List[StitchedGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.StitchedGrain` instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.StitchedGrain` instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.StitchedGrain` instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, StitchedGrain):
                raise TypeError("All grains in grains_list must be a StitchedGrain!")

    @classmethod
    def from_clean_maps_list(cls, clean_maps_list: List[CleanedGrainsMap],
                             merged_volume: StitchedGrainVolume,
                             filter_before_merge: bool = False,
                             filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                             dist_tol_xy: float = 0.1,
                             dist_tol_z: float = 0.2,
                             angle_tol: float = 1.0) -> StitchedGrainsMap:
        """Generate a :class:`~.StitchedGrainsMap` from a list of :class:`~.CleanedGrainsMap` via stitching.
        Uses :func:`~py3DXRDProc.grain.find_all_grain_pair_matches_from_list` to look for duplicates in `clean_maps_list`,
        indicating grains common in multiple maps due to an overlap region,
        using `dist_tol` and `angle_tol` to control the threshold for what is or isn't a `duplicate` grain.

        :param clean_maps_list: List of :class:`~.CleanedGrainsMap` to stitch together
        :param merged_volume: :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` object that this map belongs to
        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~.CleanedGrainsMap` before merging together, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol_xy: The tolerance in grain centre-centre XY distance (mm)
        :param dist_tol_z: The tolerance in grain centre-centre Z distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises TypeError: If `clean_maps_list` isn't a ``list`` instance
        :raises TypeError: If `merged_volume` isn't a :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` instance
        :raises TypeError: If any map in `clean_meaps_list` isn't a :class:`~.CleanedGrainsMap` instance
        :raises ValueError: If any map in `clean_maps_list` and `merged_volume` have a different load step
        :raises ValueError: If any map in `clean_maps_list` have a different :class:`~py3DXRDProc.phase.Phase`
        :raises ValueError: If `filter_before_merge` is `True` but `filter_bounds` is `None`
        :return: Stitched :class:`~.StitchedGrainsMap` in the same :attr:`~.RawGrainsMap.load_step` as the :class:`~.CleanedGrainsMap`
        """

        if not isinstance(clean_maps_list, list):
            raise TypeError("Clean maps list should be a list instance!")
        for clean_map in clean_maps_list:
            if not isinstance(clean_map, CleanedGrainsMap):
                raise TypeError("Each clean map in the list must be a CleanedGrainsMap instance!")
            if clean_map.load_step != merged_volume.load_step:
                raise ValueError("Clean map and merged_volume have different load steps!")
        if not merged_volume.__class__.__name__ == "StitchedGrainVolume":
            raise TypeError("merged_volume should be a StitchedGrainVolume!")

        first_clean_map_phase = clean_maps_list[0].phase
        for clean_map in clean_maps_list:
            if clean_map.phase != first_clean_map_phase:
                raise ValueError("All maps in clean_maps_list must have the same phase!")

        # Stitches a bunch of clean maps together to form a list of StitchedGrains
        all_grains = []
        for clean_map in clean_maps_list:
            all_grains.extend(clean_map.grains)
        grains_to_filter = all_grains

        if filter_before_merge:
            if filter_bounds is None:
                raise ValueError("filter_bounds must be supplied to filter!")
            else:
                # Only grains that survive the filter will be merged
                candidate_grains = filter_grain_list(grain_list=grains_to_filter,
                                                     xmin=filter_bounds[0],
                                                     xmax=filter_bounds[1],
                                                     ymin=filter_bounds[2],
                                                     ymax=filter_bounds[3],
                                                     zmin=filter_bounds[4],
                                                     zmax=filter_bounds[5],
                                                     use_adjusted_pos=False)
                log.info(f"Filtered from {len(all_grains)} grains to {len(candidate_grains)} grains")
        else:
            candidate_grains = grains_to_filter

        log.info(
            f"Stitching {len(candidate_grains)} {first_clean_map_phase.name} grains in load step {merged_volume.load_step}")

        if len(clean_maps_list) == 1:
            stitched_map = StitchedGrainsMap(grain_volume=merged_volume,
                                             clean_maps_list=clean_maps_list)
            for index, grain in enumerate(candidate_grains):
                stitched_grain = StitchedGrain.from_single_clean_grain(gid=index,
                                                                       clean_grain=grain,
                                                                       grain_map=stitched_map)
                stitched_map.add_grain(stitched_grain)
        else:
            grain_groups = find_multiple_observations_stitching(grains_list=candidate_grains,
                                                                dist_tol_xy=dist_tol_xy,
                                                                dist_tol_z=dist_tol_z,
                                                                angle_tol=angle_tol)

            stitched_map = StitchedGrainsMap(grain_volume=merged_volume,
                                             clean_maps_list=clean_maps_list)

            for index, grain_group in enumerate(grain_groups):
                stitched_grain = StitchedGrain.from_grains_list(gid=index,
                                                                grains_to_merge=grain_group,
                                                                grain_map=stitched_map)
                stitched_map.add_grain(stitched_grain)

        log.info(f"After stitch: {len(stitched_map.grains)} grains")

        return stitched_map

    @classmethod
    def import_from_hdf5_group(cls, map_group: h5py.Group,
                               grain_volume_object: StitchedGrainVolume,
                               clean_maps_list: List[CleanedGrainsMap]) -> StitchedGrainsMap:
        """Import a :class:`~.StitchedGrainsMap` from a :class:`h5py.Group`

        :param map_group: :class:`h5py.Group` containing the grains data
        :param grain_volume_object: :class:`~py3DXRDProc.grain_volume.GrainVolume` object that this map belongs to
        :param clean_maps_list: List of :class:`~.CleanedGrainsMap` that were used to create this :class:`~.StitchedGrainsMap`
        :raises TypeError: If `map_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `grain_volume_object` is not a :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` instance
        :raises ValueError: If an essential field is missing from the HDF5 dataset
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the UBI array has the wrong shape
        :raises ValueError: If the volume array has the wrong shape
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the grains have a mixture of load step names
        :raises ValueError: If the grains have a different load step name from the one attached to the `grain_volume_object`
        :return: The :class:`~.StitchedGrainsMap` created from the HDF5 group
        """

        if not isinstance(map_group, h5py.Group):
            raise TypeError("map_group should be an h5py Group instance")
        if not grain_volume_object.__class__.__name__ == "StitchedGrainVolume":
            raise TypeError("grain_volume_object should be a StitchedGrainVolume instance!")

        map_object = StitchedGrainsMap(grain_volume=grain_volume_object,
                                       clean_maps_list=clean_maps_list)

        # Ensure required fields are there

        essential_columns = required_obj_grain_attribute_names
        map_keys = map_group.keys()
        for column in essential_columns:
            if column not in map_keys:
                raise ValueError("Essential field missing from HDF5 dataset! Cannot continue")

        immutable_string_array = map_group["immutable_string"][:]

        # Set up barebones grains list
        gid_list = map_group["gid"][:]

        n_grains = len(gid_list)

        pos_array = map_group["pos"][:]
        if np.shape(pos_array) != (n_grains, 3):
            raise ValueError("Pos array has wrong shape!")

        pos_offset_array = map_group["pos_offset"][:]
        if np.shape(pos_offset_array) != (n_grains, 3):
            raise ValueError("Pos offset array has wrong shape!")

        UBI_array = map_group["UBI"][:].reshape((n_grains, 3, 3))
        if np.shape(UBI_array) != (n_grains, 3, 3):
            raise ValueError("UBI array has wrong shape!")

        volume_array = map_group["volume"][:]
        if np.shape(volume_array) != (n_grains,):
            raise ValueError("volume array has wrong shape!")

        # We have to know what the parent grains are
        parent_grains_for_each_grain = []
        # For each grain in our immutable ID list, get the grain data if it exists:
        per_grain_attributes_group = map_group["Per-Grain Data"]

        # Go through each grain in this map, get the parent map grain

        for index, immut_string in enumerate(immutable_string_array):
            this_grain_group = per_grain_attributes_group[immut_string.decode("utf-8")]

            parent_immut_strings_for_this_grain = this_grain_group["Parent Grain Immutable Strings"][:]

            # The parent grains could be in different maps as multiple clean grain maps contribute to each stitched grain map
            # Use the immutable string for the parent to work out what map it has
            parent_grain_maps_for_this_grain = [
                BaseGrainsMap.get_right_map_from_list_by_name(grain_maps_list=clean_maps_list,
                                                              map_name=parent_immut_string.decode("utf-8").rsplit(":",
                                                                                                                  maxsplit=1)[
                                                                  0])
                for parent_immut_string in parent_immut_strings_for_this_grain]

            parent_grains_for_this_grain = []
            for index, parent_immut_string in enumerate(parent_immut_strings_for_this_grain):
                this_parent_grain = parent_grain_maps_for_this_grain[index].get_grain_from_immutable_string(
                    parent_immut_string.decode("utf-8"))
                parent_grains_for_this_grain.append(this_parent_grain)

            parent_grains_for_each_grain.append(parent_grains_for_this_grain)

        # Establish a list of grain objects, initialised as barebones as possible
        grains_to_add = [StitchedGrain(gid=gid,
                                       pos_offset=pos_offset_array[index],
                                       UBI=UBI_array[index],
                                       volume=volume_array[index],
                                       grain_map=map_object,
                                       parent_clean_grains=parent_grains_for_each_grain[index])
                         for index, gid in enumerate(gid_list)]

        if "pos_error" in map_keys:
            pos_error_array = map_group["pos_error"][:]
            eps_error_array = map_group["eps_error"][:]
            eps_lab_error_array = map_group["eps_lab_error"][:]
            U_error_array = map_group["U_error"][:]
            angle_error_array = map_group["angle_error"][:]
            for grain, pos_error, eps_error, eps_lab_error, U_error, angle_error in zip(grains_to_add,
                                                                                        pos_error_array,
                                                                                        eps_error_array,
                                                                                        eps_lab_error_array,
                                                                                        U_error_array,
                                                                                        angle_error_array):
                eps_error_reshaped = eps_error.reshape(3, 3)
                eps_lab_error_reshaped = eps_lab_error.reshape(3, 3)
                U_error_reshaped = U_error.reshape(3, 3)
                grain.add_errors(pos_error=pos_error,
                                 eps_error=eps_error_reshaped,
                                 eps_lab_error=eps_lab_error_reshaped,
                                 U_error=U_error_reshaped,
                                 angle_error=angle_error)

        # Add the grains to the grain_volume object so we can use grain_volume methods for adding further attributes
        map_object.add_grains(grains_to_add)

        # Import array data
        # 1d attributes (data types should be inferred from the HDF5)
        for attribute in optional_obj_grain_attribute_names_flat:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:]
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        # 3x3 arrays
        for attribute in optional_obj_grain_attribute_names_3x3_float:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:].reshape((n_grains, 3, 3))
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        return map_object

    def export_to_hdf5_group(self, this_grain_maps_group: h5py.Group) -> h5py.Group:
        """Export a :class:`~.StitchedGrainsMap` to a :class:`h5py.Group`

        :param this_grain_maps_group: The :class:`h5py.Group` to export this map to
        :return: The :class:`h5py.Group` with the grain data filled in
        """

        # Array-able attributes and non-array-able attributes

        this_map_group = this_grain_maps_group.create_group(self.phase.name)

        # Create an array of immutable strings
        immutable_string_array = self.get_attribute_array_for_all_grains(attribute="immutable_string",
                                                                         must_be_complete=True).astype("S256")
        this_map_group.create_dataset("immutable_string",
                                      data=immutable_string_array,
                                      dtype="S256")

        # Create arrays for all other array-storable parameters

        self.fill_hdf5_table(this_map_group)

        per_grain_attributes_group = this_map_group.create_group("Per-Grain Data")

        # Every grain is going to have a parent here

        for grain in self.grains:
            # Make a group for each grain with the grain immutable string as the name
            this_grain_group = per_grain_attributes_group.create_group(grain.immutable_string)

            parent_immutable_strings = grain.parent_clean_grain_immutable_strings

            parent_immutable_strings_array = np.array(parent_immutable_strings, dtype="S256")

            this_grain_group.create_dataset("Parent Grain Immutable Strings",
                                            data=parent_immutable_strings_array,
                                            dtype="S256")

        return this_map_group


class TrackedGrainsMap(GrainsCollection[TrackedGrain]):
    """Class to describe a :class:`~.TrackedGrainsMap` that was produced by tracking a list of :class:`~.StitchedGrainsMap`

    :param grain_volume: The :class:`~py3DXRDProc.grain_volume.GrainVolume` to associate with this map
    :param stitch_maps_list: The list of :class:`~.StitchedGrainsMap` used to create this :class:`~.TrackedGrainsMap`
    :param grains_list: List of grains to initialise the :class:`~.BaseGrainsMap` with, defaults to `None`
    :raises TypeError: If not all the grains in `grains_list` are :class:`~py3DXRDProc.grain.TrackedGrain`
    :raises TypeError: If `clean_maps_list is not a ``list`` instance
    :raises TypeError: If any maps in the `stitch_maps_list` are not a :class:`~.StitchedGrainsMap` instance
    :raises ValueError: If not all maps in the `stitch_maps_list` have the same :attr:`~.StitchedGrainsMap.phase`
    """

    def __init__(self, grain_volume: TrackedGrainVolume,
                 stitch_maps_list: List[StitchedGrainsMap],
                 grains_list: Optional[List[TrackedGrain]] = None):
        if grains_list is not None:
            self.validate_grains_list(grains_list)

        if not isinstance(stitch_maps_list, list):
            raise TypeError("Stitch maps list should be a list instance!")
        for stitched_map in stitch_maps_list:
            if not isinstance(stitched_map, StitchedGrainsMap):
                raise TypeError("Each stitched map in the list must be a StitchedGrainsMap instance!")
        first_stitched_map_phase = stitch_maps_list[0].phase
        for stitched_map in stitch_maps_list:
            if stitched_map.phase != first_stitched_map_phase:
                raise ValueError("All maps in stitch_maps_list must have the same phase!")

        self._stitched_maps_list = stitch_maps_list

        super().__init__(grains_list)

        if not grain_volume.__class__.__name__ == "TrackedGrainVolume":
            raise TypeError("grain_volume should be a TrackedGrainVolume instance!")
        self._grain_volume = grain_volume

        self._phase = first_stitched_map_phase

    @property
    def name(self) -> str:
        """The name of this grain map, automatically generated
        from the :attr:`~.TrackedGrainsMap.sample` and :attr:`~.BaseGrainsMap.phase`

        :return: The name of the grain map
        """

        return f"{self.sample.name}:{self.phase.name}:tracked"

    @property
    def grain_volume(self) -> TrackedGrainVolume:
        """The grain :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` this map belongs to

        :return: The grain :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` this map belongs to
        """

        return self._grain_volume

    @property
    def sample(self) -> Sample:
        """The grain :class:`~py3DXRDProc.sample.Sample` from the :attr:`~.BaseGrainsMap.grain_volume`

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        return self.grain_volume.sample

    @property
    def phase(self) -> Phase:
        """The grain :class:`~py3DXRDProc.phase.Phase` for all the grains in this map

        :return: The :class:`~py3DXRDProc.phase.Phase` for all the grains in this map
        """

        return self._phase

    @property
    def fully_tracked_grains(self) -> List[TrackedGrain]:
        """All the grains in the map that have a paret grain at each load step in the :attr:`~.TrackedGrainsMap.sample`.

        :return: List of fully-tracked :class:`~py3DXRDProc.grain.TrackedGrain`
        """

        return [grain for grain in self.grains if
                set(grain.parent_stitch_grains_load_step_names) == set(self.sample.load_step_names)]

    @property
    def stitched_maps_list(self) -> List[StitchedGrainsMap]:
        """The list of :class:`~.StitchedGrainsMap` used to create this :class:`~.TrackedGrainsMap`

        :return: List of :class:`~.StitchedGrainsMap`
        """
        return self._stitched_maps_list

    def get_grain_from_gid(self, gid: int) -> TrackedGrain:
        """Looks for a grain by its :attr:`~py3DXRDProc.grain.BaseMapGrain.gid`

        :param gid: Grain :attr:`~py3DXRDProc.grain.BaseMapGrain.gid` to look for
        :raises TypeError: If the grain :attr:`~py3DXRDProc.grain.BaseMapGrain.gid` is not an `int`
        :raises KeyError: If no grain was found with that :attr:`~py3DXRDProc.grain.BaseMapGrain.gid`
        :return: The grain if it was found
        """

        if not isinstance(gid, int):
            raise TypeError("Wrong type supplied for gid!")
        for grain in self.grains:
            if grain.gid == gid:
                return grain
        raise KeyError("Grain not found")

    def get_grain_from_immutable_string(self, immutable_string: str) -> TrackedGrain:
        """Looks for a grain by its immutable string

        :param immutable_string: The immutable string to look for
        :raises TypeError: If the immutable string is not a string
        :raises ValueError: If the immutable string is empty
        :raises KeyError: If the grain was not found
        :return: The grain if it was found
        """

        if not isinstance(immutable_string, str):
            raise TypeError("Immutable string must be a string!")
        if immutable_string == "":
            raise ValueError("Immutable string must not be empty!")
        for grain in self.grains:
            if grain.immutable_string == immutable_string:
                return grain
        raise KeyError("Grain not found")

    def export_grain_list_to_gff(self, grains_list: List[TrackedGrain],
                                 gff_path: str, use_adjusted_position: bool) -> None:
        """Export a grain list to a gff file. Note: this does nothing to rectify repeating grain IDs.

        :param grains_list: List of grains to export
        :param gff_path: Path to gff file to create
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        :raises TypeError: If gff path not a string
        """

        self.validate_grains_list(grains_list)
        GrainsCollection.export_validated_grain_list_to_gff(grains_list=grains_list,
                                                            gff_path=gff_path,
                                                            use_adjusted_position=use_adjusted_position)

    def export_to_gff(self, gff_path: str, use_adjusted_position: bool) -> None:
        """Exports all grains to a gff file.

        :param gff_path: Path to gff file to create
        :param use_adjusted_position: Whether to use the translated vertical position for the grains in this gff. Useful for merged letterboxes.
        """

        self.export_grain_list_to_gff(grains_list=self.grains,
                                      gff_path=gff_path,
                                      use_adjusted_position=use_adjusted_position)

    @staticmethod
    def validate_grains_list(grains_list: List[TrackedGrain]) -> None:
        """Validate a list of :class:`~py3DXRDProc.grain.TrackedGrain` instances

        :param grains_list: A list of :class:`~py3DXRDProc.grain.TrackedGrain` instances
        :raises TypeError: If `grains_list` isn't a ``list`` type
        :raises ValueError: If `grains_list` is empty
        :raises TypeError: If any grain in `grains_list` isn't a :class:`~py3DXRDProc.grain.TrackedGrain` instance
        """

        if not isinstance(grains_list, list):
            raise TypeError("grains_list must be a list type!")
        if len(grains_list) == 0:
            raise ValueError("grains_list must not be empty!")
        for grain in grains_list:
            if not isinstance(grain, TrackedGrain):
                raise TypeError("All grains in grains_list must be a TrackedGrain!")

    @classmethod
    def from_stitch_maps_list(cls, stitch_maps_list: List[StitchedGrainsMap],
                              tracked_volume: TrackedGrainVolume,
                              filter_before_merge: bool = False,
                              filter_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                              dist_tol: float = 0.1,
                              angle_tol: float = 1.0) -> TrackedGrainsMap:
        """Generate a :class:`~.TrackedGrainsMap` from a list of :class:`~.StitchedGrainsMap` via tracking grains across multiple load steps.
        Uses :func:`~py3DXRDProc.grain.find_all_grain_pair_matches_from_list` to look for duplicates in `stitch_maps_list`,
        indicating the same grain observed in multiple different load steps,
        using `dist_tol` and `angle_tol` to control the threshold for what is or isn't a `duplicate` grain.

        :param stitch_maps_list: List of :class:`~.StitchedGrainsMap` to stitch together
        :param tracked_volume: :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` object that this map belongs to
        :param filter_before_merge: Whether to filter the grains geometrically in each :class:`~.StitchedGrainsMap` before merging together, defaults to `False`
        :param filter_bounds: Geometric bounds to filter the grains to, in mm, in the format `[xmin, xmax, ymin, ymax, zmin, zmax]`, defaults to `None`
        :param dist_tol: The tolerance in grain centre-centre distance (mm)
        :param angle_tol: The tolerance in grain pair misorientation (degrees)
        :raises TypeError: If `stitch_maps_list` isn't a ``list`` instance
        :raises TypeError: If `tracked_volume` isn't a :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` instance
        :raises TypeError: If any map in `stitch_maps_list` isn't a :class:`~.StitchedGrainsMap` instance
        :raises ValueError: If any map in `stitch_maps_list` and `tracked_volume` have a different sample
        :raises ValueError: If any map in `stitch_maps_list` have a different :class:`~py3DXRDProc.phase.Phase`
        :raises ValueError: If any map in `stitch_maps_list` has the same :class:`~py3DXRDProc.load_step.LoadStep`
        :return: Tracked :class:`~.TrackedGrainsMap` in the same :attr:`~.TrackedGrainsMap.sample` as the :class:`~.StitchedGrainsMap`
        """

        if not isinstance(stitch_maps_list, list):
            raise TypeError("Clean stitch_maps_list list should be a list instance!")
        if not tracked_volume.__class__.__name__ == "TrackedGrainVolume":
            raise TypeError("tracked_volume should be a TrackedGrainVolume!")
        for stitch_map in stitch_maps_list:
            if not isinstance(stitch_map, StitchedGrainsMap):
                raise TypeError("Each stitch map in the list must be a StitchedGrainsMap instance!")
            if stitch_map.sample != tracked_volume.sample:
                raise ValueError("Stitch map and tracked_volume have different samples!")

        first_stitch_map_phase = stitch_maps_list[0].phase
        for stitch_map in stitch_maps_list:
            if stitch_map.phase != first_stitch_map_phase:
                raise ValueError("All maps in stitch_maps_list must have the same phase!")

        load_steps_seen = []
        for stitch_map in stitch_maps_list:
            if stitch_map.load_step in load_steps_seen:
                raise ValueError("Stitch maps have degenerate load steps!")
            load_steps_seen.append(stitch_map.load_step)

        # Stitches a bunch of clean maps together to form a list of StitchedGrains
        all_grains = []
        for stitch_map in stitch_maps_list:
            all_grains.extend(stitch_map.grains)
        grains_to_filter = all_grains

        if filter_before_merge:
            if filter_bounds is None:
                raise ValueError("filter_bounds must be supplied to filter!")
            else:
                # Only grains that survive the filter will be merged
                candidate_grains = filter_grain_list(grain_list=grains_to_filter,
                                                     xmin=filter_bounds[0],
                                                     xmax=filter_bounds[1],
                                                     ymin=filter_bounds[2],
                                                     ymax=filter_bounds[3],
                                                     zmin=filter_bounds[4],
                                                     zmax=filter_bounds[5],
                                                     use_adjusted_pos=False)
                log.info(f"Filtered from {len(all_grains)} grains to {len(candidate_grains)} grains")
        else:
            candidate_grains = grains_to_filter

        log.info(f"Tracking {len(candidate_grains)} grains")

        grain_groups = find_multiple_observations(grains_list=candidate_grains,
                                                  dist_tol=dist_tol,
                                                  angle_tol=angle_tol)

        log.info("Removing single grains and duplicates from groups")
        grain_groups_no_solo = [group for group in grain_groups if len(group) > 1]

        grain_groups_no_duplicates = []
        for group in tqdm(grain_groups_no_solo):
            grains_by_load_step = {}
            group_no_duplicates = []
            for grain in group:
                # If this load step doesn't appear in the dict yet, make a new entry with the grain in a list
                if grain.load_step.name not in grains_by_load_step:
                    grains_by_load_step[grain.load_step.name] = [grain]
                # If the load step already appears in the dict, grow the list for that entry with the grain
                else:
                    grains_by_load_step[grain.load_step.name].append(grain)
            for load_step_name, grains_at_load_step in grains_by_load_step.items():
                if len(grains_at_load_step) > 1:
                    # Sort the grains by volume, take the largest
                    grains_at_load_step_sorted = sorted(grains_at_load_step, key=lambda x: x.volume)
                    largest_grain_at_load_step = grains_at_load_step_sorted[-1]
                    group_no_duplicates.append(largest_grain_at_load_step)
                else:
                    # Only one grain at this load step, so just append it
                    group_no_duplicates.append(grains_at_load_step[0])
            grain_groups_no_duplicates.append(group_no_duplicates)

        log.info(f"Group deduplication went from {len(grain_groups)} to {len(grain_groups_no_duplicates)}")

        completely_tracked_groups_only = [group for group in grain_groups_no_duplicates if
                                          len(group) == len(tracked_volume.sample.load_steps_list)]
        n_completely_tracked = len(completely_tracked_groups_only)

        log.info(f"{n_completely_tracked} grains completely tracked")

        if n_completely_tracked > 0:
            log.info(f"First completely tracked group")
            log.info(completely_tracked_groups_only[0])
            log.info([grain.load_step.name for grain in completely_tracked_groups_only[0]])

        tracked_map = TrackedGrainsMap(grain_volume=tracked_volume,
                                       stitch_maps_list=stitch_maps_list)

        for index, grain_group in enumerate(grain_groups_no_duplicates):
            tracked_grain = TrackedGrain.from_grains_list(gid=index,
                                                          grains_to_merge=grain_group,
                                                          grain_map=tracked_map)

            tracked_map.add_grain(tracked_grain)

        log.info(f"After track: {len(tracked_map.grains)} grains")

        return tracked_map

    @classmethod
    def import_from_hdf5_group(cls, map_group: h5py.Group,
                               grain_volume_object: TrackedGrainVolume,
                               stitch_maps_list: List[StitchedGrainsMap]) -> TrackedGrainsMap:
        """Import a :class:`~.TrackedGrainsMap` from a :class:`h5py.Group`

        :param map_group: :class:`h5py.Group` containing the grains data
        :param grain_volume_object: :class:`~py3DXRDProc.grain_volume.GrainVolume` object that this map belongs to
        :param stitch_maps_list: List of :class:`~.StitchedGrainsMap` that were used to create this :class:`~.TrackedGrainsMap`
        :raises TypeError: If `map_group` is not an :class:`h5py.Group` instance
        :raises TypeError: If `grain_volume_object` is not a :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` instance
        :raises ValueError: If an essential field is missing from the HDF5 dataset
        :raises ValueError: If the position array has the wrong shape
        :raises ValueError: If the UBI array has the wrong shape
        :raises ValueError: If the volume array has the wrong shape
        :raises ValueError: If the position array has the wrong shape
        :return: The :class:`~.TrackedGrainsMap` created from the HDF5 group
        """

        if not isinstance(map_group, h5py.Group):
            raise TypeError("map_group should be an h5py Group instance")
        if not grain_volume_object.__class__.__name__ == "TrackedGrainVolume":
            raise TypeError("grain_volume_object should be a TrackedGrainVolume instance!")

        # We should have much more input validation here
        map_object = TrackedGrainsMap(grain_volume=grain_volume_object,
                                      stitch_maps_list=stitch_maps_list)

        # Ensure required fields are there

        essential_columns = required_obj_grain_attribute_names + ["pos_sample"]
        map_keys = map_group.keys()
        for column in essential_columns:
            if column not in map_keys:
                raise ValueError("Essential field missing from HDF5 dataset! Cannot continue")

        immutable_string_array = map_group["immutable_string"][:]

        # Set up barebones grains list
        gid_list = map_group["gid"][:]

        n_grains = len(gid_list)

        pos_array = map_group["pos"][:]
        if np.shape(pos_array) != (n_grains, 3):
            raise ValueError("Pos array has wrong shape!")

        pos_offset_array = map_group["pos_offset"][:]
        if np.shape(pos_offset_array) != (n_grains, 3):
            raise ValueError("Pos offset array has wrong shape!")

        pos_sample_array = map_group["pos_sample"][:]
        if np.shape(pos_sample_array) != (n_grains, 3):
            raise ValueError("Pos sample array has wrong shape!")

        UBI_array = map_group["UBI"][:].reshape((n_grains, 3, 3))
        if np.shape(UBI_array) != (n_grains, 3, 3):
            raise ValueError("UBI array has wrong shape!")

        volume_array = map_group["volume"][:]
        if np.shape(volume_array) != (n_grains,):
            raise ValueError("volume array has wrong shape!")

        # We have to know what the parent grains are
        parent_grains_for_each_grain = []
        # For each grain in our immutable ID list, get the grain data if it exists:
        per_grain_attributes_group = map_group["Per-Grain Data"]

        # Go through each grain in this map, get the parent map grain

        for immut_string in immutable_string_array:
            this_grain_group = per_grain_attributes_group[immut_string.decode("utf-8")]

            parent_immut_strings_for_this_grain = this_grain_group["Parent Grain Immutable Strings"][:]

            # The parent grains could be in different maps as multiple clean grain maps contribute to each stitched grain map
            # Use the immutable string for the parent to work out what map it has
            parent_grain_maps_for_this_grain = [
                BaseGrainsMap.get_right_map_from_list_by_name(grain_maps_list=stitch_maps_list,
                                                              map_name=parent_immut_string.decode("utf-8").rsplit(":",
                                                                                                                  maxsplit=1)[
                                                                  0])
                for parent_immut_string in parent_immut_strings_for_this_grain]

            parent_grains_for_this_grain = []
            for index, parent_immut_string in enumerate(parent_immut_strings_for_this_grain):
                this_parent_grain = parent_grain_maps_for_this_grain[index].get_grain_from_immutable_string(
                    parent_immut_string.decode("utf-8"))
                parent_grains_for_this_grain.append(this_parent_grain)

            parent_grains_for_each_grain.append(parent_grains_for_this_grain)

        # Establish a list of grain objects, initialised as barebones as possible
        grains_to_add = [TrackedGrain(gid=gid,
                                      pos_offset=pos_offset_array[index],
                                      pos_sample=pos_sample_array[index],
                                      UBI=UBI_array[index],
                                      volume=volume_array[index],
                                      grain_map=map_object,
                                      parent_stitch_grains=parent_grains_for_each_grain[index])
                         for index, gid in enumerate(gid_list)]

        if "pos_error" in map_keys:
            pos_error_array = map_group["pos_error"][:]
            eps_error_array = map_group["eps_error"][:]
            eps_lab_error_array = map_group["eps_lab_error"][:]
            U_error_array = map_group["U_error"][:]
            angle_error_array = map_group["angle_error"][:]
            for grain, pos_error, eps_error, eps_lab_error, U_error, angle_error in zip(grains_to_add,
                                                                                        pos_error_array,
                                                                                        eps_error_array,
                                                                                        eps_lab_error_array,
                                                                                        U_error_array,
                                                                                        angle_error_array):
                eps_error_reshaped = eps_error.reshape(3, 3)
                eps_lab_error_reshaped = eps_lab_error.reshape(3, 3)
                U_error_reshaped = U_error.reshape(3, 3)
                grain.add_errors(pos_error=pos_error,
                                 eps_error=eps_error_reshaped,
                                 eps_lab_error=eps_lab_error_reshaped,
                                 U_error=U_error_reshaped,
                                 angle_error=angle_error)

        # Add the grains to the grain_volume object so we can use grain_volume methods for adding further attributes
        map_object.add_grains(grains_to_add)

        # Import array data
        # 1d attributes (data types should be inferred from the HDF5)
        for attribute in optional_obj_grain_attribute_names_flat:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:]
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        # 3x3 arrays
        for attribute in optional_obj_grain_attribute_names_3x3_float:
            if attribute not in calculated_obj_grain_attribute_names:
                try:
                    attribute_array = map_group[attribute][:].reshape((n_grains, 3, 3))
                except KeyError:
                    continue
                map_object.apply_attribute_array_to_all_grains(attribute_name=attribute,
                                                               attribute_values=attribute_array)

        return map_object

    def export_to_hdf5_group(self, this_grain_maps_group: h5py.Group) -> h5py.Group:
        """Export a :class:`~.TrackedGrainsMap` to a :class:`h5py.Group`

        :param this_grain_maps_group: The :class:`h5py.Group` to export this map to
        :return: The :class:`h5py.Group` with the grain data filled in
        """

        # Array-able attributes and non-array-able attributes

        this_map_group = this_grain_maps_group.create_group(self.phase.name)

        # Create an array of immutable strings
        immutable_string_array = self.get_attribute_array_for_all_grains(attribute="immutable_string",
                                                                         must_be_complete=True).astype("S256")
        this_map_group.create_dataset("immutable_string",
                                      data=immutable_string_array,
                                      dtype="S256")

        # Create arrays for all other array-storable parameters

        self.fill_hdf5_table(this_map_group)

        per_grain_attributes_group = this_map_group.create_group("Per-Grain Data")

        # Every grain is going to have a parent here

        for grain in self.grains:
            # Make a group for each grain with the grain immutable string as the name
            this_grain_group = per_grain_attributes_group.create_group(grain.immutable_string)

            parent_immutable_strings = grain.parent_stitch_grain_immutable_strings

            parent_immutable_strings_array = np.array(parent_immutable_strings, dtype="S256")

            this_grain_group.create_dataset("Parent Grain Immutable Strings",
                                            data=parent_immutable_strings_array,
                                            dtype="S256")

        stitched_maps_array = np.array([stitched_map.name for stitched_map in self.stitched_maps_list], dtype="S256")
        this_map_group.create_dataset("Stitched Maps List",
                                      data=stitched_maps_array,
                                      dtype="S256")

        return this_map_group
