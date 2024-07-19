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

#  This file incorporates work covered by the following copyright and
#  permission notice:

# Copyright(c) 2013 - 2019 Henry Proudhon.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from copy import copy

from typing import List, Tuple, Optional, TYPE_CHECKING, TypeVar, Dict

import logging

from numba import set_num_threads
from scipy import spatial

log = logging.getLogger(__name__)

import ImageD11
import networkx as nx
import numpy as np
import numpy.typing as npt
import xfab
from ImageD11 import finite_strain
from ImageD11.grain import grain as id11_grain
from ImageD11.unitcell import unitcell as id11_unitcell
from pymicro.crystal.lattice import SlipSystem
from pymicro.crystal.microstructure import Orientation
from scipy.spatial import Delaunay, distance
from xfab import tools

from py3DXRDProc.cluster import get_number_of_cores
from py3DXRDProc.conversions import symmetric_to_upper_triangular, custom_array_to_string, \
    disorientation_single_numba, strain2stress, rotate_tensor_to_lab_frame, \
    eps_error_to_sig_error, sig_error_to_sig_lab_error, rotate_tensor, are_grains_duplicate_array_numba_wrapper, \
    are_grains_duplicate_stitching_array_numba_wrapper
from py3DXRDProc.phase import Phase

if TYPE_CHECKING:
    from py3DXRDProc.grain_map import BaseGrainsMap, RawGrainsMap, CleanedGrainsMap, StitchedGrainsMap, TrackedGrainsMap
    from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume, TrackedGrainVolume
    from py3DXRDProc.load_step import LoadStep
    from py3DXRDProc.sample import Sample


# xfab.CHECKS.activated = False


class BaseGrain:
    """Class to hold the basic properties of a 3DXRD grain.
    All properties here can be calculated from the position, UBI matrix, and/or grain physical volume.

    :param pos: The position of the grain in the :class:`~py3DXRDProc.grain_volume.GrainVolume` frame, in mm
    :param UBI: The 3x3 UBI matrix of the grain
    :param volume: The volume of the grain in cubic mm
    :raises TypeError: If `pos` is not a numpy array
    :raises ValueError: If `pos` is not of shape `(3,)`
    :raises TypeError: If `pos` is not a ``float64`` dtype
    :raises TypeError: If `UBI` is not a numpy array
    :raises ValueError: If `UBI` is not of shape `(3,3)`
    :raises TypeError: If `UBI` is not a ``float64`` dtype
    :raises TypeError: If `volume` is not a ``float``
    :raises ValueError: If `volume` is negative or zero
    """

    def __init__(self, pos: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float):
        # Check position type and shape:
        if not isinstance(pos, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")
        self._pos = pos

        # Check UBI type and shape:
        if not isinstance(UBI, np.ndarray):
            raise TypeError("Matrix attribute should be a numpy matrix!")
        if not np.shape(UBI) == (3, 3):
            raise ValueError("Matrix attribute should have shape (3,3)")
        if not UBI.dtype == np.dtype("float64"):
            raise TypeError("UBI array should be an array of floats!")
        self._UBI = UBI

        # Check volume type and positive:
        if not isinstance(volume, float):
            raise TypeError("Volume of the grain should be a float!")
        if volume <= 0:
            raise ValueError("Volume of the grain must be > 0")
        self._volume = volume

    @property
    def pos(self) -> npt.NDArray[np.float64]:
        """The grain position in the lab frame, in mm, as determined by ImageD11

        :return: The grain position
        """

        return self._pos.copy()

    @property
    def UBI(self) -> npt.NDArray[np.float64]:
        """The grain UBI matrix

        :return: The grain UBI matrix (3x3)
        """

        return self._UBI.copy()

    @property
    def volume(self) -> float:
        """The grain volume in cubic mm

        :return: The grain volume in cubic mm
        """

        return copy(self._volume)

    @property
    def UB(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain UB matrix from Busing and Levy.
        Columns are the reciprocal space lattice vectors.
        Calculated from :attr:`~.BaseGrain.UBI`

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain UB matrix (3x3)
        """

        return np.linalg.inv(self.UBI).copy()

    @property
    def mt(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain Metric tensor.
        Calculated from :attr:`~.BaseGrain.UBI`

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain metric tensor (3x3)
        """

        return np.dot(self.UBI, self.UBI.T).copy()

    @property
    def rmt(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain reciprocal metric tensor.
        Calculated by inverting :attr:`~.BaseGrain.mt`.

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain reciprocal metric tensor (3x3)
        """

        return np.linalg.inv(self.mt).copy()

    @property
    def unitcell(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain unit cell as a numpy array `[a,b,c,alpha,beta,gamma]`.
        Calculated from :attr:`~.BaseGrain.rmt`.

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain unit cell (6x1) in the format `[a, b, c, alpha, beta, gamma]`
        """

        G = self.mt
        a, b, c = np.sqrt(np.diag(G))
        al = np.degrees(np.arccos(G[1, 2] / b / c))
        be = np.degrees(np.arccos(G[0, 2] / a / c))
        ga = np.degrees(np.arccos(G[0, 1] / a / b))
        return np.array((a, b, c, al, be, ga)).copy()

    @property
    def B(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain B matrix from Busing and Levy.
        Calculated from :attr:`~.BaseGrain.unitcell`.

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain B matrix (3x3)
        """

        return ImageD11.unitcell.unitcell(self.unitcell).B.copy()

    @property
    def U(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain orientation matrix (U) from Busing and Levy.
        Calculated from :attr:`~.BaseGrain.B` and :attr:`~.BaseGrain.UBI`.

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain orientation (U) matrix (3x3)
        """

        return np.dot(self.B, self.UBI).T.copy()

    @property
    def rod(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain Rodriguez vector.
        Length proportional to angle, direction is axis.
        Calculated from :attr:`~.BaseGrain.U`.

        Copyright (C) 2005-2019  Jon Wright

        :return: The grain Rodrigues vector (3x1)
        """

        return xfab.tools.u_to_rod(self.U).copy()

    @property
    def eul(self) -> npt.NDArray[np.float64]:
        """The grain Euler vector as per :mod:`xfab`.
        Calculated from :attr:`~.BaseGrain.U`.

        :return: The grain Euler vector (3x1)
        """

        return xfab.tools.u_to_euler(self.U).copy()

    def hkl_vec_as_cryst_vec(self, hkl_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a vector in the hkl reciprocal frame to the cartesian grain frame.
        :param hkl_vec: The input length-3 vector to be transformed

        :return: The vector in the cartesian grain frame"""

        return self.B @ hkl_vec

    def cryst_vec_as_lab_vec(self, cryst_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a vector in the cartesian grain frame to the cartesian lab frame.
        :param cryst_vec: The input length-3 vector to be transformed

        :return: The vector in the cartesian lab frame"""

        return self.U @ cryst_vec

    def hkl_vec_as_lab_vec(self, hkl_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a vector in the hkl reciprocal frame to the cartesian lab frame.
        :param hkl_vec: The input length-3 vector to be transformed

        :return: The vector in the cartesian lab frame"""

        return self.UB @ hkl_vec

    @property
    def as_imaged11_grain(self) -> id11_grain:
        """Get the grain as an :mod:`ImageD11` grain object
        (just using :attr:`~.BaseGrain.UBI` and :attr:`~.BaseGrain.pos`)

        :return: Grain as an :class:`ImageD11.grain.grain` object
        """

        imaged11_grain_obj = id11_grain(ubi=self.UBI, translation=self.pos)
        return imaged11_grain_obj

    def attribute_name_to_string(self, attribute: str) -> str:
        """Get a grain attribute formatted as a string

        :param attribute: The name of the grain attribute, e.g "pos" or "UBI"
        :raises TypeError: If `attribute` is not a ``str``
        :raises TypeError: If `attribute` couldn't be found for this grain
        :raises ValueError: If an invalid `attribute` was passed
        :return: The grain attribute formatted as a ``str``
        """

        from py3DXRDProc.grain_map import single_column_attributes
        if not isinstance(attribute, str):
            raise TypeError("Attribute must be string!")
        flattenable_attributes = list(single_column_attributes.keys()) + ["eul", "rod", "pos",
                                                                          "pos_offset", "U", "UBI"]
        upper_tri_attributes = ["eps", "sig", "eps_lab", "sig_lab"]
        if attribute in flattenable_attributes:
            try:
                grain_attr_value = np.array(getattr(self, attribute))
                return custom_array_to_string(grain_attr_value.flatten())
            except TypeError:
                raise TypeError(f"Attribute {attribute} broke")
        elif attribute in upper_tri_attributes:
            if attribute in ["sig", "sig_lab"]:
                # Convert to MPa
                return custom_array_to_string(symmetric_to_upper_triangular(getattr(self, attribute) / 1.0e6))
            else:
                return custom_array_to_string(symmetric_to_upper_triangular(getattr(self, attribute)))
        else:
            raise ValueError(f"Invalid attribute {attribute} passed!")

    def __repr__(self) -> str:
        return f"Grain with position {self.pos}"
        # return f"{'Grain: ' + str(self.gid):<11} {'Position: ' + str(self.pos):<51} Rodrigues: {self.rod}"

    def __eq__(self, other) -> bool:
        return self is other


TBaseGrain = TypeVar('TBaseGrain', bound=BaseGrain)


class VirtualGrain(BaseGrain):
    """Virtual grain class that has a hardcoded :attr:`.VirtualGrain.pos_offset`. Useful for merging operations.

    :param pos: The position of the grain in the lab frame, in mm
    :param UBI: The 3x3 UBI matrix of the grain
    :param volume: The volume of the grain in cubic mm
    :param pos_offset: The position of the grain in the lab frame, including offset by vertical shift
    :param phase: The :class:`~py3DXRDProc.phase.Phase` of the grain, defaults to `None`
    :raises TypeError: If `pos_offset` is not a numpy array
    :raises ValueError: If `pos_offset` is not of shape `(3,)`
    :raises TypeError: If `pos_offset` is not a ``float64`` dtype
    :raises TypeError: If `phase` is not a :class:`~py3DXRDProc.phase.Phase` instance
    """

    def __init__(self, pos: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 pos_offset: npt.NDArray[np.float64],
                 pos_sample: npt.NDArray[np.float64],
                 phase: Optional[Phase] = None):
        super().__init__(pos=pos, UBI=UBI, volume=volume)

        # Check pos_offset type and shape:
        if not isinstance(pos_offset, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos_offset) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos_offset.dtype == np.dtype("float64"):
            raise TypeError("pos_offset array should be an array of floats!")
        self.pos_offset = pos_offset

        # Check pos_sample type and shape:
        if not isinstance(pos_sample, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos_sample) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos_sample.dtype == np.dtype("float64"):
            raise TypeError("pos_sample array should be an array of floats!")
        self.pos_sample = pos_sample

        # Check volume type and positive:

        self._volume = volume

        if phase is not None:
            if not isinstance(phase, Phase):
                raise TypeError("Phase should be a Phase instance!")
            self._phase = phase

        self.pos_error: Optional[npt.NDArray[np.float64]] = None
        self.eps_error: Optional[npt.NDArray[np.float64]] = None
        self.eps_lab_error: Optional[npt.NDArray[np.float64]] = None
        self.U_error: Optional[npt.NDArray[np.float64]] = None
        self.angle_error: Optional[float] = None

    @property
    def phase(self) -> Phase:
        """The grain :class:`~py3DXRDProc.phase.Phase`, hardcoded

        :return: The :class:`~py3DXRDProc.phase.Phase` of the grain
        """

        return self._phase

    @property
    def as_imaged11_grain(self) -> id11_grain:
        """Get the grain as an :mod:`ImageD11` grain object (just using :attr:`~.BaseGrain.UBI` and :attr:`.VirtualGrain.pos_offset`)

        :return: Grain as an :class:`ImageD11.grain.grain` object
        """

        imaged11_grain_obj = id11_grain(ubi=self.UBI, translation=self.pos_offset)
        return imaged11_grain_obj

    @classmethod
    def from_imaged11_grain(cls, imaged11_grain: id11_grain,
                            phase: Optional[Phase] = None) -> VirtualGrain:
        """Get an :class:`ImageD11.grain.grain` object and create a :class:`.VirtualGrain` from it

        :param imaged11_grain: The :mod:`ImageD11` grain object
        :param phase: Optional :class:`~py3DXRDProc.phase.Phase` to specify for the :class:`.VirtualGrain` object
        :return: The new :class:`.VirtualGrain` object
        """

        new_grain_obj = VirtualGrain(pos=imaged11_grain.translation,
                                     UBI=imaged11_grain.ubi,
                                     volume=1.0,
                                     pos_offset=imaged11_grain.translation,
                                     phase=phase)

        return new_grain_obj


class BaseMapGrain(BaseGrain):
    """Base class that describes a grain that is the part of a larger :class:`~py3DXRDProc.sample.Sample` and has a GID.

    :param pos: The position of the grain in the :class:`~py3DXRDProc.grain_volume.GrainVolume` frame, in mm
    :param UBI: The 3x3 UBI matrix of the grain
    :param volume: The volume of the grain in cubic mm
    :param gid: The grain ID of the grain
    :param grain_map: The :class:`~py3DXRDProc.grain_map.BaseGrainsMap` that the grain belongs to
    :raises TypeError: If `gid` is not an ``int``
    :raises ValueError: If `gid` is negative
    :raises TypeError: If `grain_map` is not a :class:`~py3DXRDProc.grain_map.BaseGrainsMap` subclass
    """

    def __init__(self, pos: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 gid: int,
                 grain_map: BaseGrainsMap):
        super().__init__(pos=pos, UBI=UBI, volume=volume)

        if not isinstance(gid, (int, np.int64)):
            raise TypeError(f"Grain ID should be an int not {type(gid)}")
        if gid < 0:
            raise ValueError("Grain ID should be a positive integer")
        self._gid = gid

        # Getter-only pattern is preferred
        # So we need to validate the grain map here
        if grain_map.__class__.__name__ not in ["BaseGrainsMap", "RawGrainsMap", "CleanedGrainsMap",
                                                "StitchedGrainsMap", "TrackedGrainsMap"]:
            raise TypeError("grain_map should be a BaseGrainsMap or subclass instance!")
        self._grain_map = grain_map

        self._pos_error: Optional[npt.NDArray[np.float64]] = None
        self._eps_error: Optional[npt.NDArray[np.float64]] = None
        self._eps_lab_error: Optional[npt.NDArray[np.float64]] = None
        self._U_error: Optional[npt.NDArray[np.float64]] = None
        self._angle_error: Optional[float] = None

    @property
    def gid(self) -> int:
        """The grain ID

        :return: The grain ID
        """

        return self._gid

    @property
    def pos_offset(self) -> npt.NDArray[np.float64]:
        """The grain position in the lab reference frame, offset by the shift of motor position (mm).
        Calculated by adding the :attr:`py3DXRDProc.grain_volume.FloatingGrainVolume.offset_origin` to :attr:`~.BaseGrain.pos`

        :return: The position of the grain in the :attr:`~.BaseMapGrain.sample` frame, in mm
        """

        return self.pos + self.grain_volume.offset_origin

    @property
    def pos_sample(self) -> npt.NDArray[np.float64]:
        """The grain position in the sample reference frame.
        Grain cleaning is performed with the pos attribute
        Grain stitching is performed with the pos_offset attribute
        Grain tracking is performed with the pos_sample attribute, to remove effects of realigning between load steps.
        Taking the 4x4 affine matrix D and decomposing it into a translation and rotation:

        pos_sample = (pos + trans) @ rot.T

        :return: The position of the grain in the sample frame, in mm
        """

        rotation = self.load_step.D[0:3, 0:3]
        translation = self.load_step.D[0:3, 3]

        return (self.pos_offset + translation) @ rotation.T

    @property
    def U_sample(self) -> npt.NDArray[np.float64]:
        """The grain lattice orientation in the sample reference frame.
        U_s = D @ U

        :return: The lattice orientation in the sample reference frame
        """

        return self.load_step.D[0:3, 0:3] @ self.U

    @property
    def UB_sample(self) -> npt.NDArray[np.float64]:
        """The grain UB matrix using U_sample.
        UB_sample = U_sample @ B

        :return: The grain UB matrix using U_sample
        """

        return self.U_sample @ self.B

    def cryst_vec_as_sample_vec(self, cryst_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a vector in the cartesian grain frame to the cartesian sample frame.
        :param cryst_vec: The input length-3 vector to be transformed

        :return: The vector in the cartesian sample frame"""

        return self.U_sample @ cryst_vec

    def hkl_vec_as_sample_vec(self, hkl_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a vector in the hkl reciprocal frame to the cartesian sample frame.
        :param hkl_vec: The input length-3 vector to be transformed

        :return: The vector in the cartesian sample frame"""

        return self.UB_sample @ hkl_vec

    @property
    def immutable_string(self) -> str:
        """An immutable string describing the grain in the form grain_map.name:gid

        :return: The grain immutable string
        """

        return f"{self.grain_map.name}:{self.gid}"

    @property
    def grain_map(self) -> BaseGrainsMap:
        """The :class:`~py3DXRDProc.grain_map.BaseGrainsMap` that this grain belongs to

        :return: The grain :class:`~py3DXRDProc.grain_map.BaseGrainsMap`
        """

        return self._grain_map

    @property
    def phase(self) -> Phase:
        """The grain :class:`~py3DXRDProc.phase.Phase` from the :attr:`~.BaseMapGrain.grain_map`

        :return: The grain :class:`~py3DXRDProc.phase.Phase`
        """

        return self.grain_map.phase

    @property
    def volume_scaled(self):
        """The grain volume in cubic mm, scaled by the volume fraction of this phase

        :return: The grain volume in cubic mm
        """
        # this must be load-step dependent

        return self.volume * self.load_step.phase_volume_fractions[self.phase.name]

    @property
    def radius(self) -> float:
        """The grain radius in mm, assuming spherical grain

        :return: The grain radius in mm
        """
        return np.cbrt((3. / 4) * (1 / np.pi) * self.volume_scaled)

    @property
    def diameter(self) -> float:
        """The grain diameter in mm, assuming spherical grain

        :return: The grain diameter in mm
        """

        return 2 * self.radius

    @property
    def grain_volume(self) -> GrainVolume | StitchedGrainVolume:
        """The grain :class:`~py3DXRDProc.grain_volume.GrainVolume` or :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume` from the :attr:`~.BaseMapGrain.grain_map`

        :return: The grain :class:`~py3DXRDProc.grain_volume.GrainVolume` or :class:`~py3DXRDProc.grain_volume.StitchedGrainVolume`
        """

        return self.grain_map.grain_volume

    @property
    def load_step(self) -> LoadStep:
        """The grain :class:`~py3DXRDProc.load_step.LoadStep` from the :attr:`~.BaseMapGrain.grain_map`

        :return: The grain :class:`~py3DXRDProc.load_step.LoadStep`
        """
        return self.grain_map.load_step

    @property
    def sample(self) -> Sample:
        """The grain :class:`~py3DXRDProc.sample.Sample` from the :attr:`~.BaseMapGrain.load_step`

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        return self.load_step.sample

    @property
    def reference_unit_cell(self) -> npt.NDArray[np.float64]:
        """The grain reference unit cell from the :attr:`~.BaseMapGrain.phase`

        :return: The grain reference unit cell in the format `[a, b, c, alpha, beta, gamma]`
        """

        return self.phase.reference_unit_cell

    @property
    def eps(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain strain as a symmetric matrix in the grain reference system.
        Calculated from :attr:`~.BaseGrain.UBI`, :attr:`~.BaseMapGrain.reference_unit_cell`, and :attr:`~.BaseGrain.B`

        Copyright (C) 2005-2019  Jon Wright

        :return: The symmetric grain strain tensor (3x3) in the grain reference system
        """

        # Calculate a B matrix from the reference strain-free unit cell
        B = id11_unitcell(self.reference_unit_cell).B
        # Use the B matrix to determine the finite strain tensor
        F = finite_strain.DeformationGradientTensor(ubi=self.UBI, ub0=B)
        return F.finite_strain_ref(0.5)

    @property
    def eps_lab(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain strain as a symmetric matrix in the lab reference frame.
        Calculated from :attr:`~.BaseGrain.UBI`, :attr:`~.BaseMapGrain.reference_unit_cell`, and :attr:`~.BaseGrain.B`

        Copyright (C) 2005-2019  Jon Wright

        :return: The symmetric grain strain tensor (3x3) in the lab reference frame
        """

        B = id11_unitcell(self.reference_unit_cell).B
        F = finite_strain.DeformationGradientTensor(ubi=self.UBI, ub0=B)
        return F.finite_strain_lab(0.5)

    @property
    def eps_sample(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain strain as a symmetric matrix in the sample reference frame.
        Calculated by rotating :attr:`~.BaseMapGrain.eps_lab` by D

        Copyright (C) 2005-2019  Jon Wright

        :return: The symmetric grain strain tensor (3x3) in the sample reference frame
        """

        return rotate_tensor(self.eps_lab, self.load_step.D[0:3, 0:3])

    @property
    def eps_hydro(self) -> npt.NDArray[np.float64]:
        """The grain hydrostatic strain (frame invariant).
        Calculated from :attr:`~.BaseMapGrain.eps`

        :return: The grain hydrostatic strain tensor
        """

        return ((self.eps[0, 0] + self.eps[1, 1] + self.eps[2, 2]) / 3) * np.eye(3)

    @property
    def eps_lab_deviatoric(self) -> npt.NDArray[np.float64]:
        """The grain deviatoric strain in the lab reference frame.
        Calculated from :attr:`~.BaseMapGrain.eps_lab`

        :return: The grain deviatoric strain in the lab reference frame
        """

        return self.eps_lab - self.eps_hydro

    @property
    def eps_sample_deviatoric(self) -> npt.NDArray[np.float64]:
        """The grain deviatoric strain in the sample reference frame.
        Calculated from :attr:`~.BaseMapGrain.eps_sample`

        :return: The grain deviatoric strain in the sample reference frame
        """

        return self.eps_sample - self.eps_hydro

    @property
    def sig(self) -> npt.NDArray[np.float64]:
        """The grain stress (in Pa) as a symmetric matrix in the grain reference frame.
        Calculated from :attr:`~.BaseMapGrain.eps` and the :attr:`~.BaseMapGrain.phase` :attr:`~py3DXRDProc.phase.Phase.stiffnessMV`

        :return: The symmetric grain stress tensor (3x3) in the grain reference frame, in Pa
        """

        return strain2stress(epsilon=self.eps, C=self.phase.stiffnessMV)

    @property
    def sig_lab(self) -> npt.NDArray[np.float64]:
        """The grain stress (in Pa) as a symmetric matrix in the lab reference frame.
        Calculated from :attr:`~.BaseMapGrain.sig` and :attr:`~.BaseGrain.U`

        :return: The symmetric grain stress tensor (3x3) in the lab reference frame, in Pa
        """

        return rotate_tensor_to_lab_frame(grain=self.sig, U=self.U)

    @property
    def sig_sample(self) -> npt.NDArray[np.float64]:
        """The grain stress (in Pa) as a symmetric matrix in the sample reference frame.
        Calculated from :attr:`~.BaseMapGrain.sig`, :attr:`~.BaseGrain.U` and D

        :return: The symmetric grain stress tensor (3x3) in the sample reference frame, in Pa
        """

        return rotate_tensor(self.sig, self.load_step.D[0:3, 0:3] @ self.U)

    @property
    def sig_hydro(self) -> float:
        """The grain hydrostatic stress (frame invariant).
        Calculated from :attr:`~.BaseMapGrain.sig`

        :return: The grain hydrostatic stress
        """

        return ((self.sig[0, 0] + self.sig[1, 1] + self.sig[2, 2]) / 3) * np.eye(3)

    @property
    def sig_lab_deviatoric(self) -> float:
        """The grain deviatoric stress in the lab reference frame.
        Calculated from :attr:`~.BaseMapGrain.sig_lab`

        :return: The grain deviatoric stress in the lab reference frame
        """

        return self.sig_lab - self.sig_hydro

    @property
    def sig_sample_deviatoric(self) -> float:
        """The grain deviatoric stress in the sample reference frame.
        Calculated from :attr:`~.BaseMapGrain.sig_sample`

        :return: The grain deviatoric stress in the sample reference frame
        """

        return self.sig_sample - self.sig_hydro

    @property
    def sig_vm(self) -> float:
        """The grain Von Mises stress in Pa.
        Calculated from :attr:`~.BaseMapGrain.sig`

        :return: The grain Von Mises stress in Pa
        """

        sig11 = self.sig[0, 0]
        sig22 = self.sig[1, 1]
        sig33 = self.sig[2, 2]
        sig12 = self.sig[0, 1]
        sig23 = self.sig[1, 2]
        sig31 = self.sig[2, 0]
        return np.sqrt(((sig11 - sig22) ** 2 + (sig22 - sig33) ** 2 + (sig33 - sig11) ** 2 + 6 * (
                sig12 ** 2 + sig23 ** 2 + sig31 ** 2)) / 2.)

    @property
    def sig_vm_error(self) -> float:
        """The error in grain Von Mises stress in Pa.
        Propagated through :attr:`~.BaseMapGrain.sig` -> :attr:`~.BaseMapGrain.sig_vm` calculation

        :return: The error in grain Von Mises stress in Pa
        """

        sig11 = self.sig[0, 0]
        sig22 = self.sig[1, 1]
        sig33 = self.sig[2, 2]
        sig12 = self.sig[0, 1]
        sig23 = self.sig[1, 2]
        sig31 = self.sig[2, 0]

        dsig11 = self.sig_error[0, 0]
        dsig22 = self.sig_error[1, 1]
        dsig33 = self.sig_error[2, 2]
        dsig12 = self.sig_error[0, 1]
        dsig23 = self.sig_error[1, 2]
        dsig31 = self.sig_error[2, 0]

        # sig_vm = sqrt((A+B+C+D)/2) = sqrt(E/2)
        # E = A + B + C + D
        # A = (sig11-sig22)^2
        # B = (sig22-sig33)^2
        # C = (sig33-sig11)^2
        # D = 6(sig12^2 + sig23^2 + sig31^2)

        A = (sig11 - sig22) ** 2
        B = (sig22 - sig33) ** 2
        C = (sig33 - sig11) ** 2
        D = 6 * (sig12 ** 2 + sig23 ** 2 + sig31 ** 2)
        E = A + B + C + D

        dA = 2 * (sig11 - sig22) * np.sqrt(dsig11 ** 2 + dsig22 ** 2)
        dB = 2 * (sig22 - sig33) * np.sqrt(dsig22 ** 2 + dsig33 ** 2)
        dC = 2 * (sig33 - sig11) * np.sqrt(dsig33 ** 2 + dsig11 ** 2)
        dD = 6 * np.sqrt((2 * sig12 * dsig12) ** 2 + (2 * sig23 * dsig23) ** 2 + (2 * sig31 * dsig31) ** 2)
        dE = np.sqrt(dA ** 2 + dB ** 2 + dC ** 2 + dD ** 2)

        dsig_vm = (1 / 2) * self.sig_vm * dE / E
        return dsig_vm

    @property
    def pos_error(self) -> npt.NDArray[np.float64]:
        """The grain position error, one standard deviation, in mm

        :raises AttributeError: If position error for this grain is `None`
        :return: The error in :attr:`~.BaseGrain.pos`
        """

        if self._pos_error is None:
            raise AttributeError("Grain doesn't have position errors!")
        else:
            return self._pos_error

    @property
    def U_error(self) -> npt.NDArray[np.float64]:
        """The grain U matrix error, one standard deviation

        :raises AttributeError: If U matrix error for this grain is `None`
        :return: The error in :attr:`~.BaseGrain.U`
        """

        if self._U_error is None:
            raise AttributeError("Grain doesn't have U matrix errors!")
        else:
            return self._U_error

    @property
    def eps_error(self) -> npt.NDArray[np.float64]:
        """The grain strain error as a symmetric tensor in the grain reference system, one standard deviation

        :raises AttributeError: If eps error for this grain is `None`
        :return: The error in :attr:`~.BaseMapGrain.eps`
        """

        if self._eps_error is None:
            raise AttributeError("Grain doesn't have strain errors!")
        else:
            return self._eps_error

    @property
    def eps_lab_error(self) -> npt.NDArray[np.float64]:
        """The grain strain error as a symmetric tensor in the lab reference frame, one standard deviation.

        :raises AttributeError: If eps_lab error for this grain is `None`
        :return: The error in lab frame
        """

        if self._eps_lab_error is None:
            raise AttributeError("Grain doesn't have strain errors!")
        else:
            return self._eps_lab_error

    @property
    def angle_error(self) -> float:
        """The grain angle error, misorientation between :attr:`~.BaseGrain.U` and (:attr:`~.BaseGrain.U` + :attr:`~.BaseMapGrain.U_error`) in degrees

        :raises AttributeError: If angle error for this grain is `None`
        :return: The grain angle error in degrees
        """

        if self._angle_error is None:
            raise AttributeError("Grain doesn't have angle errors!")
        else:
            return self._angle_error

    @property
    def sig_error(self) -> npt.NDArray[np.float64]:
        """The grain stress error (in Pa) as a symmetric tensor in the grain reference frame, one standard deviation.
        Propagated through :attr:`~.BaseMapGrain.eps` -> :attr:`~.BaseMapGrain.sig` conversion

        :return: The error in :attr:`~.BaseMapGrain.sig`
        """

        return eps_error_to_sig_error(eps_error=self.eps_error, stiffnessMV=self.phase.stiffnessMV)

    @property
    def sig_lab_error(self) -> npt.NDArray[np.float64]:
        """The stress strain error (in Pa) as a symmetric tensor in the lab refrence frame, one standard deviation.
        Propagated through :attr:`~.BaseMapGrain.sig` -> :attr:`~.BaseMapGrain.sig_lab` tensor rotation

        :return: The error in :attr:`~.BaseMapGrain.sig_lab`
        """

        return sig_error_to_sig_lab_error(U=self.U,
                                          U_error=self.U_error,
                                          sig=self.sig,
                                          sig_error=self.sig_error)

    @property
    def has_errors(self) -> bool:
        """Whether the grain has all errors filled in

        :return: `True` if the grain has all errors, `False` otherwise
        """

        try:
            pos_error = self.pos_error
            eps_error = self.eps_error
            eps_lab_error = self.eps_lab_error
            U_error = self.U_error
            angle_error = self.angle_error
            return True
        except AttributeError:
            return False

    @property
    def load_direction(self) -> npt.NDArray[np.float64]:
        """The grain load direction from the grain :attr:`~.BaseMapGrain.load_step` in the :attr:`~.BaseMapGrain.sample` reference frame

        :return: Vector of load axis in the :attr:`~.BaseMapGrain.load_step` reference frame
        """

        return self.load_step.load_direction

    @property
    def all_slip_systems(self) -> List[SlipSystem]:
        """Get all this grain's slip systems from the grain :attr:`~.BaseMapGrain.phase`

        :return: List of all possible grain slip systems

        """

        return self.phase.all_slip_systems

    @property
    def all_schmid_factors(self) -> List[Tuple[float, SlipSystem]]:
        # Copyright(c) 2013 - 2019 Henry Proudhon
        # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro

        """Get this grain's Schmid factors for all its slip systems.

        Copyright(c) 2013 - 2019 Henry Proudhon

        :return: A list of the Schmid factors for all slip systems for this grain.
        """

        schmid_factor_list = []
        for ss in self.all_slip_systems:
            sf = self.schmid_factor(ss)
            schmid_factor_list.append((sf, ss))
        return schmid_factor_list

    @property
    def highest_slip_system(self) -> SlipSystem:
        """Get this grain's slip system with the highest Schmid factor from all its slip systems

        :return: The slip system with the highest Schmid factor
        """

        schmid_factor_list = self.get_schmid_factors(self.all_slip_systems)
        schmid_factor_list_sorted = sorted(schmid_factor_list, key=lambda x: x[0], reverse=True)
        highest_schmid_factor, highest_slip_system = schmid_factor_list_sorted[0]
        return highest_slip_system

    @property
    def highest_schmid_factor(self) -> float:
        """Get the Schmid factor of the highest slip system of this grain

        :return: The Schmid factor
        """

        schmid_factor = self.schmid_factor(slip_system=self.highest_slip_system)

        return schmid_factor

    def add_errors(self, pos_error: npt.NDArray[np.float64],
                   eps_error: npt.NDArray[np.float64],
                   eps_lab_error: npt.NDArray[np.float64],
                   U_error: npt.NDArray[np.float64],
                   angle_error: float) -> None:
        """Add errors to the grain

        :param pos_error: The error in grain position (:attr:`~.BaseGrain.pos`)
        :param eps_error: The error in grain strain in the grain frame (:attr:`~.BaseMapGrain.eps`)
        :param eps_lab_error: The error in grain strain in the lab frame (:attr:`~.BaseMapGrain.eps_lab`)
        :param U_error: The error in the grain U matrix (:attr:`~.BaseGrain.U`)
        :param angle_error: The angular error of the grain
        :raises ValueError: If :attr:`~.BaseMapGrain.pos_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.eps_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.eps_lab_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.U_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.angle_error` is not `None`
        :raises TypeError: If `pos_error` is not a numpy array
        :raises TypeError: If `pos_error` is not shape (3,)
        :raises TypeError: If `pos_error` is not an array of ``float64``
        :raises TypeError: If `pos_error` is `None`
        :raises TypeError: If `eps_error` is not a numpy array
        :raises TypeError: If `eps_error` is not shape (3,3)
        :raises TypeError: If `eps_error` is not an array of ``float64``
        :raises TypeError: If `eps_error` is `None`
        :raises TypeError: If `eps_lab_error` is not a numpy array
        :raises TypeError: If `eps_lab_error` is not shape (3,3)
        :raises TypeError: If `eps_lab_error` is not an array of ``float64``
        :raises TypeError: If `eps_lab_error` is `None`
        :raises TypeError: If `U_error` is not a numpy array
        :raises TypeError: If `U_error` is not shape (3,3)
        :raises TypeError: If `U_error` is not an array of ``float64``
        :raises TypeError: If `U_error` is `None`
        :raises TypeError: If `angle_error` is not a ``float``
        :raises TypeError: If `angle_error` is `None`
        """

        if self._pos_error is not None:
            raise ValueError("Pos error already set!")
        if self._eps_error is not None:
            raise ValueError("eps error already set!")
        if self._eps_lab_error is not None:
            raise ValueError("eps_lab error already set!")
        if self._U_error is not None:
            raise ValueError("U error already set!")
        if self._angle_error is not None:
            raise ValueError("Angle error already set!")

        # Check pos error type and shape:
        if not isinstance(pos_error, np.ndarray):
            raise TypeError("pos_error attribute should be a numpy array!")
        if not np.shape(pos_error) == (3,):
            raise ValueError("pos_error attribute should be an array of shape (3,)")
        if not pos_error.dtype == np.dtype("float64"):
            raise TypeError("pos_error array should be an array of floats!")
        if pos_error is None:
            raise ValueError("Cannot set pos_error to None!")

        # Check eps error type and shape:
        if not isinstance(eps_error, np.ndarray):
            raise TypeError("eps_error attribute should be a numpy array!")
        if not np.shape(eps_error) == (3, 3):
            raise ValueError("eps_error attribute should be an array of shape (3,3)")
        if not eps_error.dtype == np.dtype("float64"):
            raise TypeError("eps_error array should be an array of floats!")
        if eps_error is None:
            raise ValueError("Cannot set eps_error to None!")

        # Check eps_lab error type and shape:
        if not isinstance(eps_lab_error, np.ndarray):
            raise TypeError("eps_lab_error attribute should be a numpy array!")
        if not np.shape(eps_lab_error) == (3, 3):
            raise ValueError("eps_lab_error attribute should be an array of shape (3,3)")
        if not eps_lab_error.dtype == np.dtype("float64"):
            raise TypeError("eps_lab_error array should be an array of floats!")
        if eps_lab_error is None:
            raise ValueError("Cannot set eps_lab_error to None!")

        # Check U error type and shape:
        if not isinstance(U_error, np.ndarray):
            raise TypeError("U_error attribute should be a numpy array!")
        if not np.shape(U_error) == (3, 3):
            raise ValueError("U_error attribute should be an array of shape (3,3)")
        if not U_error.dtype == np.dtype("float64"):
            raise TypeError("U_error array should be an array of floats!")
        if U_error is None:
            raise ValueError("Cannot set U_error to None!")

        # Check angle error type and shape:
        if not isinstance(angle_error, float):
            raise TypeError("angle_error attribute should be a float!")
        if angle_error is None:
            raise ValueError("Cannot set angle_error to None!")

        self._pos_error = pos_error
        self._eps_error = eps_error
        self._eps_lab_error = eps_lab_error
        self._U_error = U_error
        self._angle_error = angle_error

    def to_gff_line(self, header_list: List[str]) -> str:
        """Gets this grain as a row of a GFF file

        :param header_list: List of headers of the GFF file, in order
        :raises TypeError: If `header_list` is not a ``list``
        :return: The grain properties as a string, formatted for a GFF file
        """

        if not isinstance(header_list, list):
            raise TypeError("Header list should be a list!")
        # No conversion here, use eps and eps_lab only
        line_string = ""
        for header in header_list:
            line_string += self.attribute_name_to_string(header) + " "

        string_to_return = line_string + "\n"

        return string_to_return

    def schmid_factor(self, slip_system: SlipSystem) -> float:
        # Copyright(c) 2013 - 2019 Henry Proudhon
        # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro

        """Compute the Schmid factor for this grain and the
        given :class:`~pymicro.crystal.microstructure.SlipSystem`.

        Copyright(c) 2013 - 2019 Henry Proudhon

        :param slip_system: a :class:`~pymicro.crystal.microstructure.SlipSystem` instance.
        :return: The Schmid factor, a ``float`` between 0 and 0.5.
        """

        plane = slip_system.get_slip_plane()
        g = self.U_sample.T
        gt = g.T
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)
        schmid_factor = np.abs(np.dot(n_rot, self.load_direction) *
                               np.dot(slip_rot, self.load_direction))

        if schmid_factor > 0.5:
            raise ValueError(f"Schmid factor greater than 0.5! Slip system {slip_system} might be dodgy")

        return schmid_factor

    def schmid_factor_slip_plane_coeff(self, slip_system: SlipSystem) -> float:
        # Copyright(c) 2013 - 2019 Henry Proudhon
        # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro

        """Compute the Schmid factor slip plane coefficient for this grain and the
        given :class:`~pymicro.crystal.microstructure.SlipSystem`.

        Copyright(c) 2013 - 2019 Henry Proudhon

        :param slip_system: a :class:`~pymicro.crystal.microstructure.SlipSystem` instance.
        :return: The slip plane coefficient, a ``float``
        """

        plane = slip_system.get_slip_plane()
        g = self.U_sample.T
        gt = g.T
        n_rot = np.dot(gt, plane.normal())  # plane.normal() is a unit vector

        return np.abs(np.dot(n_rot, self.load_direction))

    def schmid_factor_slip_direc_coeff(self, slip_system: SlipSystem) -> float:
        # Copyright(c) 2013 - 2019 Henry Proudhon
        # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro

        """Compute the Schmid factor slip direction coefficient for this grain and the
        given :class:`~pymicro.crystal.microstructure.SlipSystem`.

        Copyright(c) 2013 - 2019 Henry Proudhon

        :param slip_system: a :class:`~pymicro.crystal.microstructure.SlipSystem` instance.
        :return: The slip direction coefficient, a ``float``
        """

        g = self.U_sample.T
        gt = g.T
        slip = slip_system.get_slip_direction().direction()
        slip_rot = np.dot(gt, slip)

        return np.abs(np.dot(slip_rot, self.load_direction))

    def get_schmid_factors(self, slip_systems: List[SlipSystem]) -> List[Tuple[float, SlipSystem]]:
        # Copyright(c) 2013 - 2019 Henry Proudhon
        # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro

        """Get all Schmid factors for this grain from a given list of slip systems.

        Copyright(c) 2013 - 2019 Henry Proudhon

        :param slip_systems: A list of the :class:`~pymicro.crystal.microstructure.SlipSystem` from which to compute the Schmid factor values.
        :return: A list of the Schmid factors.
        """

        schmid_factor_list = []
        for ss in slip_systems:
            sf = self.schmid_factor(ss)
            schmid_factor_list.append((sf, ss))
        return schmid_factor_list

    def highest_slip_system_from_list(self, slip_systems: List[SlipSystem]) -> SlipSystem:
        """Get this grain's slip system with the highest Schmid factor from a given list of slip systems

        :param slip_systems: a list of the slip systems from which to compute the Schmid factor values.
        :return: The :class:`~pymicro.crystal.microstructure.SlipSystem` from the given list with the highest Schmid factor
        """

        schmid_factor_list = self.get_schmid_factors(slip_systems)
        schmid_factor_list_sorted = sorted(schmid_factor_list, key=lambda x: x[0], reverse=True)
        highest_schmid_factor, highest_slip_system = schmid_factor_list_sorted[0]
        return highest_slip_system

    # def get_nearest_neighbours_from_grain_list(self, grains_list: List[BaseMapGrain], max_distance: float = 0.2) -> List[
    #     BaseMapGrain]:
    #     """Calculates all the nearest neighbour grains from this grain given a grain list and the maximum neighbour distance
    #
    #     :param grains_list: A list of :class:`~.BaseMapGrain` objects to look for neighbours in
    #     :param max_distance: The maximum neighbour distance in mm
    #     :raises TypeError: If `grains_list` is not a ``list`` of :class:`~.BaseMapGrain` objects
    #     :raises TypeError: If `max_distance` isn't a ``float``
    #     :raises ValueError: If no neighbours were found in that distance
    #     :returns: A list (of max length `n_neighbours`) of all nearest neighbours found within `max_distance`, excluding `self`
    #     """
    #     # Input validation
    #     validate_grains_list(grains_list)
    #     for grain in grains_list:
    #         if not isinstance(grain, BaseMapGrain):
    #             raise TypeError("All grains in grains_list must be a BaseMapGrain instance!")
    #
    #     if not isinstance(max_distance, float):
    #         raise TypeError("max_distance must be a float!")
    #
    #     pos_sample_ref_array = np.array([a_grain.pos_offset for a_grain in grains_list])
    #     n_workers = get_number_of_cores() - 1
    #     tree = KDTree(data=pos_sample_ref_array)
    #
    #     indices = tree.query_ball_point(x=self.pos_offset,
    #                                     workers=n_workers,
    #                                     r=max_distance)
    #
    #     neighbour_grains = [grains_list[index] for index in indices if grains_list[index] is not self]
    #     if len(neighbour_grains) == 0:
    #         raise ValueError("No neighbours found!")
    #
    #     return neighbour_grains


TBaseMapGrain = TypeVar('TBaseMapGrain', bound=BaseMapGrain)


class RawGrain(BaseMapGrain):
    """Class that describes a grain that came from a .map file output by the ImageD11 indexing routine.

    :param pos: The 3D position of the grain in mm as a numpy array
    :param UBI: The UBI matrix of the grain as a numpy array
    :param volume: The volume of the grain in cubic mm
    :param gid: The grain ID of the grain
    :param grain_map: The :attr:`~.BaseMapGrain.grain_map` that the grain belongs to
    :raises TypeError: If `grain_map` is not a :class:`~py3DXRDProc.grain_map.RawGrainsMap` class
    """

    def __init__(self, pos: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 gid: int,
                 grain_map: RawGrainsMap,
                 mean_peak_intensity: float):
        if grain_map.__class__.__name__ != "RawGrainsMap":
            raise TypeError("grain_map should be a RawGrainsMap class!")

        super().__init__(pos=pos, UBI=UBI, volume=volume, gid=gid, grain_map=grain_map)

        self.mean_peak_intensity = mean_peak_intensity


class CleanGrain(BaseMapGrain):
    """Class that describes a grain that has been created from a cleaning operation.

    :param pos: The 3D position of the new :class:`~.CleanGrain` in mm as a numpy array
    :param UBI: The UBI matrix of the new :class:`~.CleanGrain` as a Numpy array
    :param volume: The volume of the new :class:`~.CleanGrain` in cubic mm
    :param gid: The grain ID of the new :class:`~.CleanGrain`
    :param grain_map: The :attr:`~.BaseMapGrain.grain_map` that the new :class:`~.CleanGrain` belongs to
    :param parent_grains: List of :class:`~.RawGrain` that were combined to make this grain
    :raises TypeError: If `grain_map` not a :class:`~py3DXRDProc.grain_map.CleanedGrainsMap` class
    :raises TypeError: If `parent_grains` is not a ``list``
    :raises TypeError: If any parent grain is not a :class:`~.RawGrain`
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.sample` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.load_step` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.grain_volume` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.grain_map` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parent grains have different :attr:`~.BaseMapGrain.load_step` parameters
    :raises ValueError: If grain and parents have different :attr:`~.BaseMapGrain.grain_volume` parameters
    :raises ValueError: If grain and parents have the same :attr:`~.BaseMapGrain.grain_map` parameter
    """

    def __init__(self, gid: int,
                 pos: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 grain_map: CleanedGrainsMap,
                 parent_grains: List[RawGrain]):

        validate_grains_list(parent_grains)

        if grain_map.__class__.__name__ != "CleanedGrainsMap":
            raise TypeError("grain_map should be a CleanedGrainsMap class!")

        super().__init__(pos=pos, UBI=UBI, volume=volume, gid=gid, grain_map=grain_map)

        if not isinstance(parent_grains, list):
            raise TypeError("Parent grains must be a list!")
        for a_grain in parent_grains:
            if not isinstance(a_grain, RawGrain):
                raise TypeError("One of the grains isn't a proper grain!")
        # Ensure that all parent grains have the same sample, load step, phase, volume and map
        all_samples = [parent_grain.sample for parent_grain in parent_grains]
        if len(set(all_samples)) != 1:
            raise ValueError("Parent grains don't all have the same sample")

        all_load_steps = [parent_grain.load_step for parent_grain in parent_grains]
        if len(set(all_load_steps)) != 1:
            raise ValueError("Parent grains don't all have the same load step")

        # All the parent grains must have the same grain volume, map and phase
        all_volumes = [parent_grain.grain_volume for parent_grain in parent_grains]
        if len(set(all_volumes)) != 1:
            raise ValueError("Parent grains don't all have the same grain volume")

        all_maps = [parent_grain.grain_map for parent_grain in parent_grains]
        if len(set(all_maps)) != 1:
            raise ValueError("Parent grains don't all have the same grain map")

        all_phases = [parent_grain.phase for parent_grain in parent_grains]
        if len(set(all_phases)) != 1:
            raise ValueError("Parent grains don't all have the same phase")

        # Ensure that the grain has the same phase as the parent grains
        if not all([parent_grain.phase == self.phase for parent_grain in parent_grains]):
            raise ValueError("Grain and its parents should have the same phase")

        # Ensure that the grain has the same load step as the parent grains
        if not all([parent_grain.load_step == self.load_step for parent_grain in parent_grains]):
            raise ValueError("Grain and its parents should have the same load step")

        # Ensure that the grain has the same grain volume as the parent grains
        if not all([parent_grain.grain_volume == self.grain_volume for parent_grain in parent_grains]):
            raise ValueError("Grain and its parents should have the same grain volume")

        # Ensure that the grain has different grain maps from its parents
        if not all([parent_grain.grain_map != self.grain_map for parent_grain in parent_grains]):
            raise ValueError("Grain and its parents should have different grain maps!")

        self._parent_grains = parent_grains

    @property
    def parent_grains(self) -> List[RawGrain]:
        """List of :class:`~.RawGrain` that combined to create this :class:`~.CleanGrain`

        :return: List of parent grains for this grain
        """

        return self._parent_grains

    @property
    def parent_grain_immutable_strings(self) -> List[str]:
        """List of immutable strings for all this grain's parent grains

        :return: List of immutable strings for each parent grain
        """

        return [grain.immutable_string for grain in self.parent_grains]

    @classmethod
    def from_grains_list(cls, gid: int,
                         grains_to_merge: List[RawGrain],
                         grain_map: CleanedGrainsMap) -> CleanGrain:
        """Generate a :class:`~.CleanGrain` from a list of :class:`~.RawGrain`, used during a cleaning operation

        :param gid: The grain ID you'd like the new :class:`~.CleanGrain` to have
        :param grains_to_merge: The list of :class:`~.RawGrain` that will be merged together
        :param grain_map: The grain map of the new :class:`~.CleanGrain`
        :raises AssertionError: If the `pos_error` of the new grain created isn't `None`, but any other error is `None`
        :return: The new :class:`~.CleanGrain` object that was created
        """

        # gid and grain_map get checked when cleangrain is init
        # grains_to_merge get checked in merge_grains

        new_simple_grain_obj = merge_grains(grains_to_merge)

        # Make a new grain object from the averaged measurements
        new_grain_obj = CleanGrain(gid=gid,
                                   pos=new_simple_grain_obj.pos,
                                   UBI=new_simple_grain_obj.UBI,
                                   volume=float(new_simple_grain_obj.volume),  # just in case
                                   grain_map=grain_map,
                                   parent_grains=grains_to_merge)

        if new_simple_grain_obj.pos_error is not None:
            assert new_simple_grain_obj.eps_error is not None
            assert new_simple_grain_obj.eps_lab_error is not None
            assert new_simple_grain_obj.U_error is not None
            assert new_simple_grain_obj.angle_error is not None
            new_grain_obj.add_errors(pos_error=new_simple_grain_obj.pos_error,
                                     eps_error=new_simple_grain_obj.eps_error,
                                     eps_lab_error=new_simple_grain_obj.eps_lab_error,
                                     U_error=new_simple_grain_obj.U_error,
                                     angle_error=new_simple_grain_obj.angle_error)

        return new_grain_obj

    def __repr__(self) -> str:
        return f"CleanGrain with position {self.pos}"
        # return f"{'Grain: ' + str(self.gid):<11} {'Position: ' + str(self.pos):<51} Rodrigues: {self.rod}"


class StitchedGrain(BaseMapGrain):
    """Class that describes a grain that has been created by stitching multiple grain volumes together

    :param gid: The grain ID of the new :class:`~.StitchedGrain`
    :param pos_offset: The 3D position of the new :class:`~.StitchedGrain` in mm as a numpy array in the sample reference frame
    :param UBI: The UBI matrix of the new :class:`~.StitchedGrain` as a numpy array
    :param volume: The volume of the new :class:`~.StitchedGrain` in cubic mm
    :param grain_map: The :attr:`~.BaseMapGrain.grain_map` that the new :class:`~.StitchedGrain` belongs to
    :param parent_clean_grains: List of :class:`~.CleanGrain` that were combined to make this grain
    :raises TypeError: If `pos_offset` is not a numpy array
    :raises ValueError: If `pos_offset` is not an array of shape (3,)
    :raises TypeError: If `pos_offset` is not an array of type ``float64``
    :raises TypeError: If `parent_clean_grains` is not a ``list``
    :raises TypeError: If any of the parent grains isn't a :class:`~.CleanGrain`
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.sample` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.load_step` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parent grains have different :attr:`~.BaseMapGrain.load_step` parameters
    :raises ValueError: If grain and parents have the same :attr:`~.BaseMapGrain.grain_map`
    """

    def __init__(self, gid: int,
                 pos_offset: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 grain_map: StitchedGrainsMap,
                 parent_clean_grains: List[CleanGrain]):

        validate_grains_list(parent_clean_grains)

        if grain_map.__class__.__name__ != "StitchedGrainsMap":
            raise TypeError("grain_map should be a StitchedGrainsMap or subclass instance!")

        # Check position type and shape:
        if not isinstance(pos_offset, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos_offset) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos_offset.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")

        pos = pos_offset - grain_map.grain_volume.offset_origin

        super().__init__(pos=pos, UBI=UBI, volume=volume, gid=gid, grain_map=grain_map)

        # Check the underlying value first
        if not isinstance(parent_clean_grains, list):
            raise TypeError("Parent grains must be a list!")
        for a_grain in parent_clean_grains:
            if not isinstance(a_grain, CleanGrain):
                raise TypeError("One of the grains isn't a CleanGrain!")
        # Ensure that all parent grains have the same sample, load step, phase, volume and map
        all_samples = [parent_grain.sample for parent_grain in parent_clean_grains]
        if len(set(all_samples)) != 1:
            raise ValueError("Parent grains don't all have the same sample")

        all_load_steps = [parent_grain.load_step for parent_grain in parent_clean_grains]
        if len(set(all_load_steps)) != 1:
            raise ValueError("Parent grains don't all have the same load step")

        all_phases = [parent_grain.phase for parent_grain in parent_clean_grains]
        if len(set(all_phases)) != 1:
            raise ValueError("Parent grains don't all have the same phase")

        # Ensure that the grain has the same phase as the parent grains
        if not all([parent_grain.phase == self.phase for parent_grain in parent_clean_grains]):
            raise ValueError("Grain and its parents should have the same phase")

        # Ensure that the grain has the same load step as the parent grains
        if not all([parent_grain.load_step == self.load_step for parent_grain in parent_clean_grains]):
            raise ValueError("Grain and its parents should have the same load step")

        # Ensure that the grain has different grain maps from its parents
        if not all([parent_grain.grain_map != self.grain_map for parent_grain in parent_clean_grains]):
            raise ValueError("Grain and its parents should have different grain maps!")

        self._parent_clean_grains = parent_clean_grains

    @property
    def parent_clean_grains(self) -> List[CleanGrain]:
        """List of :class:`~.CleanGrain` that combined to create this :class:`~.StitchedGrain`

        :return: List of parent clean grains
        """

        return self._parent_clean_grains

    @property
    def parent_clean_grain_immutable_strings(self) -> List[str]:
        """List of immutable strings for all this grain's parent grains

        :return: List of parent clean grain immutable strings
        """

        return [grain.immutable_string for grain in self.parent_clean_grains]

    @classmethod
    def from_single_clean_grain(cls,
                                gid: int,
                                clean_grain: CleanGrain,
                                grain_map: StitchedGrainsMap) -> StitchedGrain:
        """Generate a :class:`~.StitchedGrain` from a single :class:`~.CleanGrain`, used during a stitching operation

        :param gid: The grain ID you'd like the new :class:`~.StitchedGrain` to have
        :param clean_grain: The :class:`~.CleanGrain` that will be used to create the new :class:`~.StitchedGrain`
        :param grain_map: The grain map of the new :class:`~.StitchedGrain`
        :return: The new :class:`~.StitchedGrain` object that was created
        """

        new_grain_obj = StitchedGrain(gid=gid,
                                      pos_offset=clean_grain.pos_offset,
                                      UBI=clean_grain.UBI,
                                      volume=clean_grain.volume,
                                      grain_map=grain_map,
                                      parent_clean_grains=[clean_grain])

        if clean_grain.has_errors:
            new_grain_obj.add_errors(pos_error=clean_grain.pos_error,
                                     eps_error=clean_grain.eps_error,
                                     eps_lab_error=clean_grain.eps_lab_error,
                                     U_error=clean_grain.U_error,
                                     angle_error=clean_grain.angle_error)

        return new_grain_obj

    @classmethod
    def from_grains_list(cls, gid: int,
                         grains_to_merge: List[CleanGrain],
                         grain_map: StitchedGrainsMap) -> StitchedGrain:
        """Generate a :class:`~.StitchedGrain` from a list of :class:`~.CleanGrain`, used during a stitching operation

        :param gid: The grain ID you'd like the new :class:`~.StitchedGrain` to have
        :param grains_to_merge: The list of :class:`~.CleanGrain` that will be merged together
        :param grain_map: The :attr:`~.BaseMapGrain.grain_map` of the new :class:`~.StitchedGrain`
        :raises AssertionError: If the `pos_error` of the new grain created isn't `None`, but any other error is `None`
        :return: The new :class:`~.StitchedGrain` object that was created
        """

        new_simple_grain_obj = merge_grains(grains_to_merge)

        # StitchedGrain objects still have positions in the sample reference frame
        # We check for duplicates based on pos_offset
        # But how do we merge grains?
        # new_base_grain is a VirtualGrain with separate pos and pos_offset
        # But the grains in a StitchedGrain come from different grain volumes
        # So the only position that matters is the sample ref position
        # pos_offset = pos + offset_origin
        # pos = pos_offset - offset_origin

        # Make a new grain object from the averaged measurements
        new_grain_obj = StitchedGrain(gid=gid,
                                      pos_offset=new_simple_grain_obj.pos_offset,
                                      UBI=new_simple_grain_obj.UBI,
                                      volume=new_simple_grain_obj.volume,
                                      grain_map=grain_map,
                                      parent_clean_grains=grains_to_merge)

        if new_simple_grain_obj.pos_error is not None:
            assert new_simple_grain_obj.eps_error is not None
            assert new_simple_grain_obj.eps_lab_error is not None
            assert new_simple_grain_obj.U_error is not None
            assert new_simple_grain_obj.angle_error is not None
            new_grain_obj.add_errors(pos_error=new_simple_grain_obj.pos_error,
                                     eps_error=new_simple_grain_obj.eps_error,
                                     eps_lab_error=new_simple_grain_obj.eps_lab_error,
                                     U_error=new_simple_grain_obj.U_error,
                                     angle_error=new_simple_grain_obj.angle_error)

        return new_grain_obj


class TrackedGrain(BaseGrain):
    """Class that describes a grain that has been created by tracking grains across multiple load steps

    :param gid: The grain ID of the new :class:`~.TrackedGrain`
    :param pos_offset: The 3D position of the new :class:`~.TrackedGrain` in mm as a numpy array in the sample reference frame
    :param UBI: The UBI matrix of the new :class:`~.TrackedGrain` as a numpy array
    :param volume: The volume of the new :class:`~.TrackedGrain` in cubic mm
    :param grain_map: The :attr:`~.TrackedGrain.grain_map` that the new :class:`~.TrackedGrain` belongs to
    :param parent_stitch_grains: List of :class:`~.StitchedGrain` that were combined to make this grain
    :raises TypeError: If `grain_map` is not a :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` instance
    :raises TypeError: If `pos_offset` is not a numpy array
    :raises ValueError: If `pos_offset` is not an array of shape (3,)
    :raises TypeError: If `pos_offset` is not an array of type ``float64``
    :raises TypeError: If `parent_stitch_grains_list` not a ``list``
    :raises TypeError: If any of the parent grains isn't a :class:`~.StitchedGrain`
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.sample` parameters
    :raises ValueError: If parent grains have degenerate :attr:`~.BaseMapGrain.load_step` parameters
    :raises ValueError: If parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parent grains have different :attr:`~.BaseMapGrain.phase` parameters
    :raises ValueError: If grain and parents have the same :attr:`~.BaseMapGrain.grain_map`
    """

    def __init__(self, gid: int,
                 pos_offset: npt.NDArray[np.float64],
                 pos_sample: npt.NDArray[np.float64],
                 UBI: npt.NDArray[np.float64],
                 volume: float,
                 grain_map: TrackedGrainsMap,
                 parent_stitch_grains: List[StitchedGrain]):

        validate_grains_list(parent_stitch_grains)

        # Check position type and shape:
        if not isinstance(pos_offset, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos_offset) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos_offset.dtype == np.dtype("float64"):
            raise TypeError("Pos array should be an array of floats!")

        # Check pos_sample type and shape:
        if not isinstance(pos_sample, np.ndarray):
            raise TypeError("Vector attribute should be a numpy array!")
        if not np.shape(pos_sample) == (3,):
            raise ValueError("Vector attribute should be an array of length (3,)")
        if not pos_sample.dtype == np.dtype("float64"):
            raise TypeError("pos_sample should be an array of floats!")

        self._pos_sample = pos_sample

        if grain_map.__class__.__name__ != "TrackedGrainsMap":
            raise TypeError("grain_map should be a TrackedGrainsMap instance!")
        self._grain_map = grain_map

        pos = pos_offset - grain_map.grain_volume.offset_origin

        super().__init__(pos=pos, UBI=UBI, volume=volume)

        if not isinstance(gid, (int, np.int64)):
            raise TypeError(f"Grain ID should be an int not {type(gid)}")
        if gid < 0:
            raise ValueError("Grain ID should be a positive integer")
        self._gid = gid

        # Check the underlying value first
        if not isinstance(parent_stitch_grains, list):
            raise TypeError("Parent grains must be a list!")
        for a_grain in parent_stitch_grains:
            if not isinstance(a_grain, StitchedGrain):
                raise TypeError("One of the grains isn't a proper grain!")

        parent_grain_load_steps = [parent_grain.load_step for parent_grain in parent_stitch_grains]
        unique_load_steps = set(parent_grain_load_steps)
        if len(parent_grain_load_steps) != len(unique_load_steps):
            raise ValueError("Parent grains have degenerate load steps!")

        # Ensure that all parent grains have the same sample and phase
        all_samples = [parent_grain.sample for parent_grain in parent_stitch_grains]
        if len(set(all_samples)) != 1:
            raise ValueError("Parent grains don't all have the same sample")

        all_phases = [parent_grain.phase for parent_grain in parent_stitch_grains]
        if len(set(all_phases)) != 1:
            raise ValueError("Parent grains don't all have the same phase")

        # Ensure that the grain has the same phase as the parent grains
        if not all([parent_grain.phase == self.phase for parent_grain in parent_stitch_grains]):
            raise ValueError("Grain and its parents should have the same phase")

        # Ensure that the grain has different grain maps from its parents
        if not all([parent_grain.grain_map != self.grain_map for parent_grain in parent_stitch_grains]):
            raise ValueError("Grain and its parents should have different grain maps!")

        # Make a dictionary of the parent stitch grains
        parent_stitch_grains_dict = {}

        for load_step in self.sample.load_steps_list:
            for parent_grain in parent_stitch_grains:
                if parent_grain.load_step == load_step:
                    parent_stitch_grains_dict[load_step.name] = parent_grain

        self._parent_stitch_grains_dict = parent_stitch_grains_dict

        self._pos_error: Optional[npt.NDArray[np.float64]] = None
        self._eps_error: Optional[npt.NDArray[np.float64]] = None
        self._eps_lab_error: Optional[npt.NDArray[np.float64]] = None
        self._U_error: Optional[npt.NDArray[np.float64]] = None
        self._angle_error: Optional[float] = None

    @property
    def gid(self) -> int:
        """The grain ID

        :return: The grain ID
        """

        return self._gid

    @property
    def pos_offset(self) -> npt.NDArray[np.float64]:
        """The grain position in the :attr:`~.TrackedGrain.sample` reference frame (mm)

        :return: The position of the grain in the :attr:`~.TrackedGrain.sample` frame, in mm
        """

        return self.pos + self.grain_volume.offset_origin

    @property
    def pos_sample(self) -> npt.NDArray[np.float64]:
        return self._pos_sample

    @property
    def immutable_string(self) -> str:
        """An immutable string describing the grain in the form grain_map.name:gid

        :return: The grain immutable string
        """

        return f"{self.grain_map.name}:{self.gid}"

    @property
    def grain_map(self) -> TrackedGrainsMap:
        """The :class:`~py3DXRDProc.grain_map.TrackedGrainsMap` that this grain belongs to

        :return: The grain :class:`~py3DXRDProc.grain_map.TrackedGrainsMap`
        """

        return self._grain_map

    @property
    def phase(self) -> Phase:
        """The grain :class:`~py3DXRDProc.phase.Phase` from the :attr:`~.TrackedGrain.grain_map`

        :return: The grain :class:`~py3DXRDProc.phase.Phase`
        """

        return self.grain_map.phase

    @property
    def grain_volume(self) -> TrackedGrainVolume:
        """The grain :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume` from the :attr:`~.TrackedGrain.grain_map`

        :return: The grain :class:`~py3DXRDProc.grain_volume.TrackedGrainVolume`
        """

        return self.grain_map.grain_volume

    @property
    def sample(self) -> Sample:
        """The grain :class:`~py3DXRDProc.sample.Sample` from its :attr:`~.TrackedGrain.grain_volume`

        :return: The grain :class:`~py3DXRDProc.sample.Sample`
        """

        # Tracked Grains don't have a load step we can use
        return self.grain_volume.sample

    @property
    def parent_stitch_grains(self) -> Dict[str, StitchedGrain]:
        """Dict of :class:`~.StitchedGrain` that combined to create this :class:`~.TrackedGrain`.
        Entered by their load step name

        :return: ``dict`` of parent :class:`~.StitchedGrain` for this :class:`~.TrackedGrain`
        """

        return self._parent_stitch_grains_dict

    @property
    def parent_stitch_grains_list(self) -> List[StitchedGrain]:
        """List of :class:`~.StitchedGrain` that combined to create this :class:`~.TrackedGrain`

        :return: ``list`` of parent :class:`~.StitchedGrain` for this :class:`~.TrackedGrain`
        """

        return list(self.parent_stitch_grains.values())

    @property
    def parent_stitch_grains_load_step_names(self) -> List[str]:
        """List of names of load steps of all :class:`~.StitchedGrain` that combined to create this :class:`~.TrackedGrain`

        :return: ``list`` of names of load steps for each :class:`~.StitchedGrain` for this :class:`~.TrackedGrain`
        """

        return list(self.parent_stitch_grains.keys())

    @property
    def parent_stitch_grain_immutable_strings(self) -> List[str]:
        """List of :attr:`~.BaseMapGrain.immutable_string` for all this grain's parent grains

        :return: ``list`` of the :attr:`~.BaseMapGrain.immutable_string` of each :attr:`~.TrackedGrain.parent_stitch_grains_list`
        """

        return [grain.immutable_string for grain in self.parent_stitch_grains_list]

    @property
    def is_fully_tracked(self) -> bool:
        """Check that each load step in the sample has a corresponding parent grain

        :return: `True` if each load step has a corresponding parent grain, `False` otherwise
        """

        return set(self.parent_stitch_grains_load_step_names) == set(self.sample.load_step_names)

    @property
    def reference_unit_cell(self) -> npt.NDArray[np.float64]:
        """The grain reference unit cell from the :attr:`~.TrackedGrain.phase`

        :return: The grain reference unit cell in the format `[a, b, c, alpha, beta, gamma]`
        """

        return self.phase.reference_unit_cell

    @property
    def eps(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain strain as a symmetric matrix in the grain reference system.
        Calculated from :attr:`~.BaseGrain.UBI`, :attr:`~.TrackedGrain.reference_unit_cell`, and :attr:`~.BaseGrain.B`

        Copyright (C) 2005-2019  Jon Wright

        :return: The symmetric grain strain tensor (3x3) in the grain reference system
        """

        # Calculate a B matrix from the reference strain-free unit cell
        B = id11_unitcell(self.reference_unit_cell).B
        # Use the B matrix to determine the finite strain tensor
        F = finite_strain.DeformationGradientTensor(self.UBI, B)
        return F.finite_strain_ref(0.5)

    @property
    def eps_lab(self) -> npt.NDArray[np.float64]:
        # Copyright (C) 2005-2019  Jon Wright
        # Modified from ImageD11/ImageD11/grain.py at https://github.com/FABLE-3DXRD/ImageD11/

        """The grain strain as a symmetric matrix in the lab reference frame.
        Calculated from :attr:`~.BaseGrain.UBI`, :attr:`~.TrackedGrain.reference_unit_cell`, and :attr:`~.BaseGrain.B`

        Copyright (C) 2005-2019  Jon Wright

        :return: The symmetric grain strain tensor (3x3) in the lab reference frame
        """

        B = id11_unitcell(self.reference_unit_cell).B
        F = finite_strain.DeformationGradientTensor(self.UBI, B)
        return F.finite_strain_lab(0.5)

    @property
    def eps_hydro(self) -> npt.NDArray[np.float64]:
        """The grain hydrostatic strain (frame invariant).
        Calculated from :attr:`~.TrackedGrain.eps`

        :return: The grain hydrostatic strain
        """

        return ((self.eps[0, 0] + self.eps[1, 1] + self.eps[2, 2]) / 3) * np.eye(3)

    @property
    def eps_lab_deviatoric(self) -> npt.NDArray[np.float64]:
        """The grain deviatoric strain in the lab reference frame.
        Calculated from :attr:`~.TrackedGrain.eps_lab`

        :return: The grain deviatoric strain in the lab reference frame
        """

        return self.eps_lab - self.eps_hydro

    @property
    def sig(self) -> npt.NDArray[np.float64]:
        """The grain stress (in Pa) as a symmetric matrix in the grain reference system.
        Calculated from :attr:`~.TrackedGrain.eps` and the :attr:`~.TrackedGrain.phase` :attr:`~py3DXRDProc.phase.Phase.stiffnessMV`

        :return: The symmetric grain stress tensor (3x3) in the grain reference system, in Pa
        """

        return strain2stress(epsilon=self.eps, C=self.phase.stiffnessMV)

    @property
    def sig_lab(self) -> npt.NDArray[np.float64]:
        """The grain stress (in Pa) as a symmetric matrix in the lab reference frame.
        Calculated from :attr:`~.TrackedGrain.sig` and :attr:`~.BaseGrain.U`

        :return: The symmetric grain stress tensor (3x3) in the lab reference frame, in Pa
        """

        return rotate_tensor_to_lab_frame(grain=self.sig, U=self.U)

    @property
    def sig_hydro(self) -> float:
        """The grain hydrostatic stress (frame invariant).
        Calculated from :attr:`~.TrackedGrain.sig`

        :return: The grain hydrostatic stress
        """

        return ((self.sig[0, 0] + self.sig[1, 1] + self.sig[2, 2]) / 3) * np.eye(3)

    @property
    def sig_lab_deviatoric(self) -> float:
        """The grain deviatoric stress in the lab reference frame.
        Calculated from :attr:`~.TrackedGrain.sig_lab`

        :return: The grain deviatoric stress in the lab reference frame
        """

        return self.sig_lab - self.sig_hydro

    @property
    def sig_vm(self) -> float:
        """The grain Von Mises stress in Pa.
        Calculated from :attr:`~.TrackedGrain.sig`

        :return: The grain Von Mises stress in Pa
        """

        sig11 = self.sig[0, 0]
        sig22 = self.sig[1, 1]
        sig33 = self.sig[2, 2]
        sig12 = self.sig[0, 1]
        sig23 = self.sig[1, 2]
        sig31 = self.sig[2, 0]
        return np.sqrt(((sig11 - sig22) ** 2 + (sig22 - sig33) ** 2 + (sig33 - sig11) ** 2 + 6 * (
                sig12 ** 2 + sig23 ** 2 + sig31 ** 2)) / 2.)

    @property
    def sig_vm_error(self) -> float:
        """The error in grain Von Mises stress in Pa.
        Propagated through :attr:`~.TrackedGrain.sig` -> :attr:`~.TrackedGrain.sig_vm` calculation

        :return: The error in grain Von Mises stress in Pa
        """

        sig11 = self.sig[0, 0]
        sig22 = self.sig[1, 1]
        sig33 = self.sig[2, 2]
        sig12 = self.sig[0, 1]
        sig23 = self.sig[1, 2]
        sig31 = self.sig[2, 0]

        dsig11 = self.sig_error[0, 0]
        dsig22 = self.sig_error[1, 1]
        dsig33 = self.sig_error[2, 2]
        dsig12 = self.sig_error[0, 1]
        dsig23 = self.sig_error[1, 2]
        dsig31 = self.sig_error[2, 0]

        # sig_vm = sqrt((A+B+C+D)/2) = sqrt(E/2)
        # E = A + B + C + D
        # A = (sig11-sig22)^2
        # B = (sig22-sig33)^2
        # C = (sig33-sig11)^2
        # D = 6(sig12^2 + sig23^2 + sig31^2)

        A = (sig11 - sig22) ** 2
        B = (sig22 - sig33) ** 2
        C = (sig33 - sig11) ** 2
        D = 6 * (sig12 ** 2 + sig23 ** 2 + sig31 ** 2)
        E = A + B + C + D

        dA = 2 * (sig11 - sig22) * np.sqrt(dsig11 ** 2 + dsig22 ** 2)
        dB = 2 * (sig22 - sig33) * np.sqrt(dsig22 ** 2 + dsig33 ** 2)
        dC = 2 * (sig33 - sig11) * np.sqrt(dsig33 ** 2 + dsig11 ** 2)
        dD = 6 * np.sqrt((2 * sig12 * dsig12) ** 2 + (2 * sig23 * dsig23) ** 2 + (2 * sig31 * dsig31) ** 2)
        dE = np.sqrt(dA ** 2 + dB ** 2 + dC ** 2 + dD ** 2)

        dsig_vm = (1 / 2) * self.sig_vm * dE / E
        return dsig_vm

    @property
    def pos_error(self) -> npt.NDArray[np.float64]:
        """The grain position error, one standard deviation, in mm

        :raises AttributeError: If position error for this grain is `None`
        :return: The error in :attr:`~.BaseGrain.pos`
        """

        if self._pos_error is None:
            raise AttributeError("Grain doesn't have position errors!")
        else:
            return self._pos_error

    @property
    def U_error(self) -> npt.NDArray[np.float64]:
        """The grain U matrix error, one standard deviation

        :raises AttributeError: If U matrix error for this grain is `None`
        :return: The error in :attr:`~.BaseGrain.U`
        """

        if self._U_error is None:
            raise AttributeError("Grain doesn't have U matrix errors!")
        else:
            return self._U_error

    @property
    def eps_error(self) -> npt.NDArray[np.float64]:
        """The grain strain error as a symmetric tensor in the grain reference system, one standard deviation

        :raises AttributeError: If eps error for this grain is `None`
        :return: The error in :attr:`~.TrackedGrain.eps`
        """

        if self._eps_error is None:
            raise AttributeError("Grain doesn't have strain errors!")
        else:
            return self._eps_error

    @property
    def eps_lab_error(self) -> npt.NDArray[np.float64]:
        """The grain strain error as a symmetric tensor in the lab reference frame, one standard deviation

        :raises AttributeError: If eps_lab error for this grain is `None`
        :return: The error in :attr:`~.BaseMapGrain.eps_lab`
        """

        if self._eps_lab_error is None:
            raise AttributeError("Grain doesn't have strain errors!")
        else:
            return self._eps_lab_error

    @property
    def angle_error(self) -> float:
        """The grain angle error, misorientation between :attr:`~.BaseGrain.U` and (:attr:`~.BaseGrain.U` + :attr:`~.BaseMapGrain.U_error`) in degrees

        :raises AttributeError: If angle error for this grain is `None`
        :return: The grain angle error in degrees
        """

        if self._angle_error is None:
            raise AttributeError("Grain doesn't have angle errors!")
        else:
            return self._angle_error

    @property
    def sig_error(self) -> npt.NDArray[np.float64]:
        """The grain stress error (in Pa) as a symmetric tensor in the grain reference system, one standard deviation.
        Propagated through :attr:`~.TrackedGrain.eps` -> :attr:`~.TrackedGrain.sig` conversion

        :return: The error in :attr:`~.TrackedGrain.sig`
        """

        return eps_error_to_sig_error(eps_error=self.eps_error, stiffnessMV=self.phase.stiffnessMV)

    @property
    def sig_lab_error(self) -> npt.NDArray[np.float64]:
        """The stress strain error (in Pa) as a symmetric tensor in the lab reference frame, one standard deviation
        Propagated through :attr:`~.TrackedGrain.sig` -> :attr:`~.TrackedGrain.sig_lab` tensor rotation

        :return: The error in :attr:`~.TrackedGrain.sig_lab`
        """
        return sig_error_to_sig_lab_error(U=self.U,
                                          U_error=self.U_error,
                                          sig=self.sig,
                                          sig_error=self.sig_error)

    @property
    def has_errors(self) -> bool:
        """Whether the grain has all errors filled in

        :return: `True` if the grain has all errors, `False` otherwise
        """

        try:
            pos_error = self.pos_error
            eps_error = self.eps_error
            eps_lab_error = self.eps_lab_error
            U_error = self.U_error
            angle_error = self.angle_error
            return True
        except AttributeError:
            return False

    def add_errors(self, pos_error: npt.NDArray[np.float64],
                   eps_error: npt.NDArray[np.float64],
                   eps_lab_error: npt.NDArray[np.float64],
                   U_error: npt.NDArray[np.float64],
                   angle_error: float) -> None:
        """Add errors to the grain

        :param pos_error: The error in grain position (:attr:`~.BaseGrain.pos`)
        :param eps_error: The error in grain strain in the grain frame (:attr:`~.TrackedGrain.eps`)
        :param eps_lab_error: The error in grain strain in the lab frame (:attr:`~.TrackedGrain.eps_lab`)
        :param U_error: The error in the grain U matrix (:attr:`~.BaseGrain.U`)
        :param angle_error: The angular error of the grain
        :raises ValueError: If :attr:`~.BaseMapGrain.pos_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.eps_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.eps_lab_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.U_error` is not `None`
        :raises ValueError: If :attr:`~.BaseMapGrain.angle_error` is not `None`
        :raises TypeError: If `pos_error` is not a numpy array
        :raises TypeError: If `pos_error` is not shape (3,)
        :raises TypeError: If `pos_error` is not an array of ``float64``
        :raises TypeError: If `pos_error` is `None`
        :raises TypeError: If `eps_error` is not a numpy array
        :raises TypeError: If `eps_error` is not shape (3,3)
        :raises TypeError: If `eps_error` is not an array of ``float64``
        :raises TypeError: If `eps_error` is `None`
        :raises TypeError: If `eps_lab_error` is not a numpy array
        :raises TypeError: If `eps_lab_error` is not shape (3,3)
        :raises TypeError: If `eps_lab_error` is not an array of ``float64``
        :raises TypeError: If `eps_lab_error` is `None`
        :raises TypeError: If `U_error` is not a numpy array
        :raises TypeError: If `U_error` is not shape (3,3)
        :raises TypeError: If `U_error` is not an array of ``float64``
        :raises TypeError: If `U_error` is `None`
        :raises TypeError: If `angle_error` is not a ``float``
        :raises TypeError: If `angle_error` is `None`
        """

        if self._pos_error is not None:
            raise ValueError("Pos error already set!")
        if self._eps_error is not None:
            raise ValueError("eps error already set!")
        if self._eps_lab_error is not None:
            raise ValueError("eps_lab error already set!")
        if self._U_error is not None:
            raise ValueError("U error already set!")
        if self._angle_error is not None:
            raise ValueError("Angle error already set!")

        # Check pos error type and shape:
        if not isinstance(pos_error, np.ndarray):
            raise TypeError("pos_error attribute should be a numpy array!")
        if not np.shape(pos_error) == (3,):
            raise ValueError("pos_error attribute should be an array of shape (3,)")
        if not pos_error.dtype == np.dtype("float64"):
            raise TypeError("pos_error array should be an array of floats!")
        if pos_error is None:
            raise ValueError("Cannot set pos_error to None!")

        # Check eps error type and shape:
        if not isinstance(eps_error, np.ndarray):
            raise TypeError("eps_error attribute should be a numpy array!")
        if not np.shape(eps_error) == (3, 3):
            raise ValueError("eps_error attribute should be an array of shape (3,3)")
        if not eps_error.dtype == np.dtype("float64"):
            raise TypeError("eps_error array should be an array of floats!")
        if eps_error is None:
            raise ValueError("Cannot set eps_error to None!")

        # Check eps_lab error type and shape:
        if not isinstance(eps_lab_error, np.ndarray):
            raise TypeError("eps_lab_error attribute should be a numpy array!")
        if not np.shape(eps_lab_error) == (3, 3):
            raise ValueError("eps_lab_error attribute should be an array of shape (3,3)")
        if not eps_lab_error.dtype == np.dtype("float64"):
            raise TypeError("eps_lab_error array should be an array of floats!")
        if eps_lab_error is None:
            raise ValueError("Cannot set eps_lab_error to None!")

        # Check U error type and shape:
        if not isinstance(U_error, np.ndarray):
            raise TypeError("U_error attribute should be a numpy array!")
        if not np.shape(U_error) == (3, 3):
            raise ValueError("U_error attribute should be an array of shape (3,3)")
        if not U_error.dtype == np.dtype("float64"):
            raise TypeError("U_error array should be an array of floats!")
        if U_error is None:
            raise ValueError("Cannot set U_error to None!")

        # Check angle error type and shape:
        if not isinstance(angle_error, float):
            raise TypeError("angle_error attribute should be a float!")
        if angle_error is None:
            raise ValueError("Cannot set angle_error to None!")

        self._pos_error = pos_error
        self._eps_error = eps_error
        self._eps_lab_error = eps_lab_error
        self._U_error = U_error
        self._angle_error = angle_error

    @classmethod
    def from_grains_list(cls, gid: int,
                         grains_to_merge: List[StitchedGrain],
                         grain_map: TrackedGrainsMap) -> TrackedGrain:
        """Generate a :class:`~.TrackedGrain` from a list of :class:`~.StitchedGrain`, used during a stitching operation

        :param gid: The grain ID you'd like the new :class:`~.TrackedGrain` to have
        :param grains_to_merge: The list of :class:`~.StitchedGrain` that will be merged together
        :param grain_map: The grain map of the new :class:`~.TrackedGrain`
        :raises AssertionError: If the `pos_error` of the new grain created isn't `None`, but any other error is `None`
        :return: The new :class:`~.TrackedGrain` object that was created
        """

        new_simple_grain_obj = merge_grains(grains_to_merge)

        # Make a new grain object from the averaged measurements
        new_grain_obj = TrackedGrain(gid=gid,
                                     pos_offset=new_simple_grain_obj.pos_offset,
                                     pos_sample=new_simple_grain_obj.pos_sample,
                                     UBI=new_simple_grain_obj.UBI,
                                     volume=new_simple_grain_obj.volume,
                                     grain_map=grain_map,
                                     parent_stitch_grains=grains_to_merge)

        if new_simple_grain_obj.pos_error is not None:
            assert new_simple_grain_obj.eps_error is not None
            assert new_simple_grain_obj.eps_lab_error is not None
            assert new_simple_grain_obj.U_error is not None
            assert new_simple_grain_obj.angle_error is not None
            new_grain_obj.add_errors(pos_error=new_simple_grain_obj.pos_error,
                                     eps_error=new_simple_grain_obj.eps_error,
                                     eps_lab_error=new_simple_grain_obj.eps_lab_error,
                                     U_error=new_simple_grain_obj.U_error,
                                     angle_error=new_simple_grain_obj.angle_error)

        return new_grain_obj

    def get_parent_grain_from_load_step(self, load_step: LoadStep) -> StitchedGrain:
        """Get the parent grain of this grain with a specific load step

        :param load_step: The LoadStep object we want the parent grain to have
        :raises KeyError: If a grain cannot be found that matches
        :return: A grain that has a matching load step
        """

        try:
            return self.parent_stitch_grains[load_step.name]
        except KeyError:
            raise KeyError(f"No parent grain with load step {load_step.name} found!")

    def to_gff_line(self, header_list: List[str]) -> str:
        """Gets this grain as a row of a GFF file

        :param header_list: List of headers of the GFF file, in order
        :raises TypeError: If `header_list` is not a ``list``
        :return: The grain properties as a string, formatted for a GFF file
        """

        if not isinstance(header_list, list):
            raise TypeError("Header list should be a list!")
        # No conversion here, use eps and eps_lab only
        line_string = ""
        for header in header_list:
            line_string += self.attribute_name_to_string(header) + " "

        string_to_return = line_string + "\n"

        return string_to_return

    # def get_nearest_neighbours_from_grain_list(self, grains_list: List[TrackedGrain], n_neighbours: int = 6,
    #                                            max_distance=0.2) -> List[
    #     TrackedGrain]:
    #     """Calculates all the nearest neighbour grains from this grain given a grain list, the maximum number of neighbours you want to find, and the maximum neighbour distance
    #
    #     :param grains_list: A list of :class:`~.TrackedGrain` objects to look for neighbours in
    #     :param n_neighbours: The maximum number of neighbours to return (excluding `self`)
    #     :param max_distance: The maximum neighbour distance in mm
    #     :raises TypeError: If `grains_list` is not a ``list`` of :class:`~.TrackedGrain` objects
    #     :raises TypeError: If `max_distance` isn't a ``float``
    #     :raises TypeError: If `n_neighbours` isn't an ``int``
    #     :raises ValueError: If no neighbours were found in that distance
    #     :returns: A list (of max length `n_neighbours`) of all nearest neighbours found within `max_distance`, excluding `self`
    #     """
    #     # Input validation
    #     validate_grains_list(grains_list)
    #     for grain in grains_list:
    #         if not isinstance(grain, TrackedGrain):
    #             raise TypeError("All grains in grains_list must be a TrackedGrain instance!")
    #     if not isinstance(n_neighbours, int):
    #         raise TypeError("n_neighbours must be an integer!")
    #     if not isinstance(max_distance, float):
    #         raise TypeError("max_distance must be a float!")
    #     # Make a master list that definitely contains the grain
    #     if self not in grains_list:
    #         all_grains_list = copy(grains_list) + [self]
    #     else:
    #         all_grains_list = copy(grains_list)
    #
    #     pos_sample_ref_array = np.array([a_grain.pos_offset for a_grain in all_grains_list])
    #     n_workers = get_number_of_cores() - 1
    #     tree = KDTree(data=pos_sample_ref_array)
    #     distances, indices = tree.query(x=[self.pos_offset],
    #                                     k=n_neighbours + 1,
    #                                     workers=n_workers,
    #                                     distance_upper_bound=max_distance)
    #     # Add 1 here to k because self is always considered a neighbour
    #     # convert indices to a flat list
    #     indices = [index for index in list(indices.flatten()) if index < len(grains_list)]
    #
    #     neighbour_grains = [all_grains_list[index] for index in indices if all_grains_list[index] is not self]
    #     if len(neighbour_grains) == 0:
    #         raise ValueError("No neighbours found!")
    #
    #     return neighbour_grains


def validate_grains_list(grain_list: List[TBaseGrain]) -> bool:
    """Check a list of grains to make sure it's not empty and contains only grains

    :param grain_list: List of grains to validate
    :raises TypeError: If `grains_list` is not a ``list`` type
    :raises ValueError: If `grains_list` is empty
    :raises TypeError: If `grains_list` is not entirely :class:`~.BaseGrain`
    """

    # Check the grain list
    if not isinstance(grain_list, list):
        raise TypeError("Grain list must be a list!")
    if len(grain_list) == 0:
        raise ValueError("Grain list must not be empty")
    # Grain list must be checked before calling this! To make sure it's a list with non-zero length
    bool_array = [isinstance(grain, BaseGrain) for grain in grain_list]
    if not all(bool_array):
        raise TypeError("Grain list is not entirely grains")
    return True


def filter_grain_list(grain_list: List[TBaseGrain],
                      xmin: float, xmax: float,
                      ymin: float, ymax: float,
                      zmin: float, zmax: float,
                      use_adjusted_pos: bool = False) -> List[TBaseGrain]:
    """Filter a grain list by positional bounds to remove outlying grains

    :param grain_list: List of grains to filter
    :param xmin: Minimum x position of the grain (mm)
    :param xmax: Maximum x position of the grain (mm)
    :param ymin: Minimum y position of the grain (mm)
    :param ymax: Maximum y position of the grain (mm)
    :param zmin: Minimum z position of the grain (mm)
    :param zmax: Maximum z position of the grain (mm)
    :param use_adjusted_pos: Decide whether you filter on the grain position in the map frame (`False`) or :attr:`~.BaseMapGrain.sample` frame (`True`), defaults to `False`
    :raises AssertionError: If the grain is a :class:`~.BaseGrain` but you want to filter with `use_adjusted_pos`
    :raises TypeError: If any of the bounds aren't ``float`` or ``int``
    :raises ValueError: If any of the max bounds are less than the min bounds
    :return: A list of :class:`~.BaseGrain` objects that survived the filter
    """

    # Check the grains list
    validate_grains_list(grain_list)
    # Check the bounds
    for bound in [xmin, xmax, ymin, ymax, zmin, zmax]:
        if not isinstance(bound, (float, int)):
            raise TypeError("Bound has the wrong type!")
    if xmin >= xmax:
        raise ValueError("Xmax must be larger than xmin!")
    if ymin >= ymax:
        raise ValueError("Ymax must be larger than ymin!")
    if zmin >= zmax:
        raise ValueError("Zmax must be larger than zmin!")

    survived_filter_list = []
    for grain in grain_list:
        if (type(grain) == BaseGrain) & use_adjusted_pos:
            raise TypeError("Can't filter a BaseGrain with an adjusted position because it doesn't have one!")
        if isinstance(grain, (RawGrain, CleanGrain, StitchedGrain, TrackedGrain)):
            if use_adjusted_pos:
                pos_to_use = grain.pos_offset
            else:
                pos_to_use = grain.pos
        else:
            pos_to_use = grain.pos
        if (xmin <= pos_to_use[0] <= xmax) & (ymin <= pos_to_use[1] <= ymax) & (zmin <= pos_to_use[2] <= zmax):
            # Grain survived
            survived_filter_list.append(grain)

    return survived_filter_list


def bootstrap_mean_grain_and_errors(grain_list: List[id11_grain], phase: Phase) -> Tuple[
    id11_grain, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[
        np.float64], float]:
    # Copyright (C) 2019-2024  James Ball
    # Copyright (C) 2013-2019  Henry Proudhon
    # Modified from pymicro/crystal/microstructure.py at https://github.com/heprom/pymicro
    """Merge a list of similar :mod:`ImageD11` grains.
    Used for generating mean grains and grain errors during the bootstrap approach.

    `pos error`: the standard deviation of all grain positions in mm.

    `eps error`: the standard deviation of grain strains (grain frame) from the grains_list as a symmetric tensor.

    `eps_lab error`: the standard deviation of grain strains (sample frame) from the grains_list as a symmetric tensor.

    `u_error`: the standard deviation of all grain U matrices accounting for fundamental zone rotations

    `angle_error`: the misorientation between the (average U matrix) and (average U matrix + u_stdev) in degrees

    Grain misorientation portion is Copyright (C) 2013-2019  Henry Proudhon

    :param grain_list: List of :mod:`ImageD11` grain objects to merge
    :param phase: :class:`~py3DXRDProc.phase.Phase` object of the grains, used to calculate strain
    :return: (merged grain, pos error, eps error, eps_lab error, u error, angle error)
    """

    rotated_grains = []

    # This is just all the cubic symmetry operators in an array:
    num_symms = phase.symmetry.symmetry_operators().shape[0]

    for gid, grain in enumerate(grain_list):
        if gid == 0:
            # If it's the first grain, make a new grain identical to the first
            new_grain = id11_grain(translation=grain.translation, ubi=grain.ubi)
        else:
            # If it's not the first grain, compare its rotation to the first grain, and rotate if required
            angle_array = np.zeros(shape=num_symms)
            rotated_grain_u_matrices = np.zeros(shape=(num_symms, 3, 3))
            for i, symm_op in enumerate(phase.symmetry.symmetry_operators()):
                # Rotate this grain by the symmetry op
                o_i = np.dot(symm_op, grain.U)
                # Save it to a list for this grain
                rotated_grain_u_matrices[i] = o_i
                # Work out the angle between the first grain U and this rotated U
                delta = np.dot(grain_list[0].U, o_i.T)
                cw = (np.trace(delta) - 1.0) / 2.0
                if cw > 1. and cw - 1. < 10 * np.finfo("float32").eps:
                    cw = 1.
                angle_array[i] = np.arccos(cw)  # returns radians

            # Get the smallest angle
            index = np.argmin(angle_array)
            closest_grain_rotation = rotated_grain_u_matrices[index]
            rotated_UBI = np.linalg.inv(np.matmul(closest_grain_rotation, grain.B))
            new_grain = id11_grain(translation=grain.translation, ubi=rotated_UBI)

        rotated_grains.append(new_grain)

    eps_symms = np.array([a_grain.eps_grain_matrix(dzero_cell=phase.reference_unit_cell) for a_grain in rotated_grains])
    eps_lab_symms = np.array(
        [a_grain.eps_sample_matrix(dzero_cell=phase.reference_unit_cell) for a_grain in rotated_grains])
    ubis_fz = np.array([a_grain.ubi for a_grain in rotated_grains])
    us_fz = np.array([a_grain.U for a_grain in rotated_grains])
    rods_fz = np.array([a_grain.Rod for a_grain in rotated_grains])
    trans_array = np.array([a_grain.translation for a_grain in rotated_grains])

    trans_mean = np.mean(trans_array, axis=0)
    ubi_mean = np.mean(ubis_fz, axis=0)

    mean_grain = id11_grain(ubi=ubi_mean, translation=trans_mean)

    trans_stdev = np.std(trans_array, axis=0)
    u_stdev = np.std(us_fz, axis=0)

    rod_mean = np.mean(rods_fz, axis=0)
    rod_stdev = np.std(rods_fz, axis=0)

    rod_mean_plus_stdev = rod_mean + rod_stdev
    u_mean_plus_stdev = Orientation.Rodrigues2OrientationMatrix(rod_mean_plus_stdev)

    rod_mean_as_matrix = Orientation.Rodrigues2OrientationMatrix(rod_mean)
    angle_error = disorientation_single_numba(matrix_tuple=(rod_mean_as_matrix, u_mean_plus_stdev),
                                              symmetries=phase.symmetry.symmetry_operators())

    eps_stdev = np.std(eps_symms, axis=0)
    eps_lab_stdev = np.std(eps_lab_symms, axis=0)

    return mean_grain, trans_stdev, eps_stdev, eps_lab_stdev, u_stdev, angle_error


def merge_grains(grains_to_merge: List[TBaseMapGrain]) -> VirtualGrain:
    """Merge a list of similar py3DXRDPRoc grains.
    Used for merging together multiple observations of the same grain.
    If only one grain to merge, just copy the grain :attr:`~.BaseGrain.pos`, :attr:`~.BaseGrain.UBI`, :attr:`~.BaseGrain.volume` and errors to the new grain.
    If multiple grains to merge:

    New position is a volume-weighted average of positions

    New position in sample frame is a volume-weighted average of positions in sample frame

    New volume is sum of volumes

    New UBI is UBI of the largest grain

    New pos error is propagated from volume-weighted average

    New eps, eps_lab, U, angle errors are taken from the largest grain

    :param grains_to_merge: List of :class:`~.BaseMapGrain` grain objects to merge
    :raises AttributeError: If some grains in `grains_to_merge` have errors but some don't
    :return: The new :class:`~.VirtualGrain` object
    """

    validate_grains_list(grains_to_merge)

    if len(grains_to_merge) == 1:
        new_grain_obj = VirtualGrain(pos=grains_to_merge[0].pos,
                                     UBI=grains_to_merge[0].UBI,
                                     volume=grains_to_merge[0].volume,  # this is a float
                                     pos_offset=grains_to_merge[0].pos_offset,
                                     pos_sample=grains_to_merge[0].pos_sample,
                                     phase=grains_to_merge[0].phase)
        if grains_to_merge[0].has_errors:
            new_grain_obj.pos_error = grains_to_merge[0].pos_error
            new_grain_obj.eps_error = grains_to_merge[0].eps_error
            new_grain_obj.eps_lab_error = grains_to_merge[0].eps_lab_error
            new_grain_obj.U_error = grains_to_merge[0].U_error
            new_grain_obj.angle_error = grains_to_merge[0].angle_error

        return new_grain_obj
    else:
        pos_array = np.array([grain.pos for grain in grains_to_merge])
        pos_offset_array = np.array([grain.pos_offset for grain in grains_to_merge])
        pos_sample_array = np.array([grain.pos_sample for grain in grains_to_merge])
        vol_array = np.array([grain.volume for grain in grains_to_merge])

        new_pos = np.average(pos_array, axis=0, weights=vol_array)
        new_pos_offset = np.average(pos_offset_array, axis=0, weights=vol_array)
        new_pos_sample = np.average(pos_sample_array, axis=0, weights=vol_array)

        new_volume = float(np.sum(vol_array))  # this is a np.float64

        # Sort grains by volume descending
        grains_sorted_by_volume = sorted(grains_to_merge, key=lambda x: x.volume, reverse=True)
        largest_grain = grains_sorted_by_volume[0]

        # Take largest grain UBI as the new UBI
        new_UBI = largest_grain.UBI

        phase = largest_grain.phase

        new_grain_obj = VirtualGrain(pos=new_pos,
                                     UBI=new_UBI,
                                     volume=new_volume,
                                     pos_offset=new_pos_offset,
                                     pos_sample=new_pos_sample,
                                     phase=phase)

        if all([grain.has_errors for grain in grains_to_merge]):
            # Propagate position error
            # Take largest grain errors for everything else
            pos_error_array = np.array([grain.pos_error for grain in grains_to_merge])
            new_pos_error = (1 / (len(grains_to_merge) * new_volume)) * np.sqrt(
                np.sum((np.power(vol_array, 2) * np.power(pos_error_array, 2).T).T, axis=0))

            new_eps_error = largest_grain.eps_error
            new_eps_lab_error = largest_grain.eps_lab_error
            new_u_error = largest_grain.U_error
            new_angle_error = largest_grain.angle_error

            new_grain_obj.pos_error = new_pos_error
            new_grain_obj.eps_error = new_eps_error
            new_grain_obj.eps_lab_error = new_eps_lab_error
            new_grain_obj.U_error = new_u_error
            new_grain_obj.angle_error = new_angle_error

        elif any([grain.has_errors for grain in grains_to_merge]):
            raise AttributeError("Some grains have errors and some don't!")

        return new_grain_obj


# not currently used
# def are_grains_similar(grain_pair: Tuple[VirtualGrain | TBaseMapGrain, VirtualGrain | TBaseMapGrain],
#                        dist_tol: float,
#                        angle_tol: float) -> bool:
#     """Determine if two different grains are similar.
#     Checks if they are close together in distance and are similarly oriented.
#
#     :param grain_pair: 2-Tuple of grains
#     :param dist_tol: Tolerance of grain centre-of-mass separation distance, mm
#     :param angle_tol: Tolerance of misorientation, degrees
#     :raises TypeError: If `dist_tol` isn't a ``float``
#     :raises TypeError: If `angle_tol` isn't a ``float``
#     :raises TypeError: If either grain in `grain_pair` isn't a :class:`~.BaseGrain` instance
#     :return: `True` if the grains are similar, `False` otherwise
#     """
#
#     if not isinstance(dist_tol, float):
#         raise TypeError("dist_tol should be a float!")
#     if not isinstance(angle_tol, float):
#         raise TypeError("angle_tol should be a float")
#
#     for grain in grain_pair:
#         if not isinstance(grain, (VirtualGrain, BaseMapGrain)):
#             raise TypeError("grain is not a VirtualGrain or BaseMapGrain instance!")
#
#     grain_a, grain_b = grain_pair
#     symmetries = grain_a.phase.symmetry.symmetry_operators()
#
#     if grain_a is grain_b:
#         return False
#
#     # Check distances:
#     separation = jit_linalg(grain_a.pos_offset, grain_b.pos_offset)
#
#     if separation > dist_tol:
#         return False
#
#     # Check misorientations:
#     misorientation = disorientation_single(matrix_tuple=(grain_a.U, grain_b.U),
#                                         symmetries=symmetries,
#                                         floateps=np.finfo('float32').eps)
#
#     if misorientation > angle_tol:
#         return False
#
#     return True


def find_all_grain_pair_matches_from_list(grains_list: List[TBaseMapGrain],
                                          dist_tol: float,
                                          angle_tol: float) -> List[Tuple[TBaseMapGrain, TBaseMapGrain]]:
    """Find matching grain pairs from a list of grains.
    Uses underlying numba function to find matching grain pairs from a list using reference grain duplicate check.

    :param grains_list: List of grains
    :param dist_tol: Tolerance of grain centre-of-mass separation distance, mm
    :param angle_tol: Tolerance of misorientation, degrees

    :raises TypeError: If `dist_tol` isn't a ``float``
    :raises TypeError: If `angle_tol` isn't a ``float``
    :return: List of 2-Tuples of matching grain pairs
    """

    if not isinstance(dist_tol, float):
        raise TypeError("dist_tol should be a float!")
    if not isinstance(angle_tol, float):
        raise TypeError("angle_tol should be a float")

    # grain list A, B, C, D, E, F
    # returns A-B, A-C, D-F
    # no deduplication

    log.info("Finding out which grain pairs match")

    ncpu = get_number_of_cores() - 1
    set_num_threads(ncpu)

    matching_grain_pair_indices = are_grains_duplicate_array_numba_wrapper(grains_list,
                                                                           dist_tol=dist_tol,
                                                                           angle_tol=angle_tol)

    log.info("Associating back to grains")
    matching_grain_pairs = [(grains_list[a], grains_list[b]) for (a, b) in matching_grain_pair_indices]

    return matching_grain_pairs


def find_all_grain_pair_matches_from_list_stitching(grains_list: List[TBaseMapGrain],
                                                    dist_tol_xy: float,
                                                    dist_tol_z: float,
                                                    angle_tol: float) -> List[Tuple[TBaseMapGrain, TBaseMapGrain]]:
    """Find matching grain pairs from a list of grains.
    Uses underlying numba function to find matching grain pairs from a list using reference grain stitching check.

    :param grains_list: List of grains
    :param dist_tol_xy: Tolerance of grain centre-of-mass separation distance in XY plane, mm
    :param dist_tol_z: Tolerance of grain centre-of-mass separation distance in Z plane, mm
    :param angle_tol: Tolerance of misorientation, degrees

    :raises TypeError: If `dist_tol` isn't a ``float``
    :raises TypeError: If `angle_tol` isn't a ``float``
    :return: List of 2-Tuples of matching grain pairs
    """

    # grain list A, B, C, D, E, F
    # returns A-B, A-C, D-F
    # no deduplication

    if not isinstance(dist_tol_xy, float):
        raise TypeError("dist_tol_xy should be a float!")
    if not isinstance(dist_tol_z, float):
        raise TypeError("dist_tol_z should be a float!")
    if not isinstance(angle_tol, float):
        raise TypeError("angle_tol should be a float")

    log.info("Finding out which grain pairs match")

    ncpu = get_number_of_cores() - 1
    set_num_threads(ncpu)

    matching_grain_pair_indices = are_grains_duplicate_stitching_array_numba_wrapper(grains_list,
                                                                                     dist_tol_xy=dist_tol_xy,
                                                                                     dist_tol_z=dist_tol_z,
                                                                                     angle_tol=angle_tol)

    log.info("Associating back to grains")
    matching_grain_pairs = [(grains_list[a], grains_list[b]) for (a, b) in matching_grain_pair_indices]

    return matching_grain_pairs


def combine_matching_grain_pairs_into_groups(grains_list: List[TBaseMapGrain],
                                             matching_grain_pairs: List[Tuple[TBaseMapGrain, TBaseMapGrain]]) -> List[
    List[TBaseMapGrain]]:
    """Takes a list of matching grain pairs and collects them into isolated groups using graph theory.
    Intended to group pairs like A<->B, A<->C into a group ABC because they have a grain in common.

    :param grains_list: List of all the grains
    :param matching_grain_pairs: List of matching grain pairs from find_all_grain_pair_matches_from_list
    :return: List of groups containing matching grains
    """

    log.info("Grouping matching grain pairs together")
    # Make a dictionary that we can use to go from the grain immutable string to the grain object

    grain_string_to_grain_obj_dict = {}
    for grain in grains_list:
        grain_string_to_grain_obj_dict[grain.immutable_string] = grain

    grain_strings_list = [grain.immutable_string for grain in grains_list]

    matching_string_pairs = [(grain_pair[0].immutable_string, grain_pair[1].immutable_string) for grain_pair in
                             matching_grain_pairs]

    # Establish NetworkX Graph object
    G = nx.Graph()

    # Add every grain string as a node from the grains_list
    G.add_nodes_from(grain_strings_list)

    # Add the edges
    G.add_edges_from(matching_string_pairs)

    connected_component_subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    isolated_groups_as_strings = [list(subgraph.nodes) for subgraph in connected_component_subgraphs]
    isolated_groups = [[grain_string_to_grain_obj_dict[grain_string] for grain_string in string_group] for string_group
                       in isolated_groups_as_strings]

    return isolated_groups


def find_multiple_observations(grains_list: List[TBaseMapGrain],
                               dist_tol: float,
                               angle_tol: float) -> List[List[TBaseMapGrain]]:
    """Find groups of similar grains from a list of grains.
    Checks if they are close together in distance and are similarly oriented.
    Do not use for stitching!

    :param grains_list: List of grains
    :param dist_tol: Tolerance of grain centre-of-mass separation distance, mm
    :param angle_tol: Tolerance of misorientation, degrees
    :return: List of groups containing matching grains
    """

    validate_grains_list(grains_list)

    # edge case: grains_list contains only one grain
    # in this case just return that one grain
    if len(grains_list) == 1:
        return [grains_list]

    matching_grain_pairs = find_all_grain_pair_matches_from_list(grains_list=grains_list,
                                                                 dist_tol=dist_tol,
                                                                 angle_tol=angle_tol)

    all_groups = combine_matching_grain_pairs_into_groups(grains_list=grains_list,
                                                          matching_grain_pairs=matching_grain_pairs)

    return all_groups


def find_multiple_observations_stitching(grains_list: List[TBaseMapGrain],
                                         dist_tol_xy: float,
                                         dist_tol_z: float,
                                         angle_tol: float) -> List[List[TBaseMapGrain]]:
    """Find groups of similar grains from a list of grains.
    Checks if they are close together in distance and are similarly oriented.

    :param grains_list: List of grains
    :param dist_tol_xy: Tolerance of grain centre-of-mass separation distance in XY plane, mm
    :param dist_tol_z: Tolerance of grain centre-of-mass separation distance in Z plane, mm
    :param angle_tol: Tolerance of misorientation, degrees
    :return: List of groups containing matching grains
    """

    validate_grains_list(grains_list)

    # edge case: grains_list contains only one grain
    # in this case just return that one grain
    if len(grains_list) == 1:
        return [grains_list]

    matching_grain_pairs = find_all_grain_pair_matches_from_list_stitching(grains_list=grains_list,
                                                                           dist_tol_xy=dist_tol_xy,
                                                                           dist_tol_z=dist_tol_z,
                                                                           angle_tol=angle_tol)

    all_groups = combine_matching_grain_pairs_into_groups(grains_list=grains_list,
                                                          matching_grain_pairs=matching_grain_pairs)

    return all_groups


def inclination_angle(grain_a: TBaseGrain, grain_b: TBaseGrain) -> float:
    """Find the vector between `grain_a` and `grain_b`, then find the angle
    between that vector and the vertical axis.
    Uses :attr:`~.BaseGrain.pos` if either grain supplied is a
    :class:`~.BaseGrain` rather than a :class:`~.BaseMapGrain`.
    Uses :attr:`~.BaseMapGrain.pos_offset` otherwise.

    :param grain_a: The first :class:`~.BaseGrain` or :class:`~.BaseMapGrain`
    :param grain_b: The second :class:`~.BaseGrain` or :class:`~.BaseMapGrain`
    :raises TypeError: If either grain supplied isn't a :class:`~.BaseGrain` or subclass
    :return: The angle to the vertical axis in degrees as a float
    """
    if not isinstance(grain_a, BaseGrain) or not isinstance(grain_b, BaseGrain):
        raise TypeError("Both grains supplied must be a BaseGrain or subclass instance!")

    if isinstance(grain_a, BaseMapGrain) or isinstance(grain_b, BaseMapGrain):
        pos_a = grain_a.pos_offset
        pos_b = grain_b.pos_offset
    else:
        pos_a = grain_a.pos
        pos_b = grain_b.pos

    horizontal_distance = spatial.distance.euclidean(pos_a[0:2], pos_b[0:2])
    vertical_distance = np.abs(pos_b[2] - pos_a[2])

    return np.rad2deg(np.arctan(horizontal_distance / vertical_distance))


# TODO: Update
def find_grain_neighbours(grain_list):
    # https://stackoverflow.com/a/73663511
    # Find the nearest neighbours of every grain
    # Get the COM position of each grain
    points = np.array([grain.pos_offset for grain in grain_list])
    # Perform Delaunay triangulation to get neighbouring data
    indptr_neigh, neighbours = Delaunay(points).vertex_neighbor_vertices
    for i in range(len(points)):
        i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i + 1]]
        centre_grain = grain_list[i]
        neighbour_grains = [grain_list[n_index] for n_index in i_neigh]
        if centre_grain in neighbour_grains:
            raise ValueError("Centre grain in neighbour grains!")
        centre_grain.neighbours = neighbour_grains

    # Iterate over all the grains
    for grain in grain_list:
        # Establish containers
        grain.series_neighbours = []
        grain.parall_neighbours = []

        # Iterate over all the neighbours
        for neighbour in grain.neighbours:
            pos_a = grain.pos_offset
            pos_b = neighbour.pos_offset

            # If grain further away horizontally than vertically, it's parallel
            horizontal_distance = distance.euclidean(pos_a[0:2], pos_b[0:2])
            vertical_distance = np.abs(pos_b[2] - pos_a[2])
            if horizontal_distance > vertical_distance:
                grain.parall_neighbours.append(neighbour)
            else:
                grain.series_neighbours.append(neighbour)

        grain.has_both_neighbour_types = len(grain.series_neighbours) > 0 and len(grain.parall_neighbours) > 0
