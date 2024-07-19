# py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
# University of Birmingham.
#
# Copyright (C) 2019-2024  James Ball
#
# This file is part of py3DXRDProc.
#
# py3DXRDProc is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# py3DXRDProc is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.

###

from typing import Dict

import h5py
import numpy as np
from py3DXRDProc.conversions import formStiffnessMV
from pymicro.crystal.lattice import Symmetry, Lattice, SlipSystem
from xfab import parameters


class Phase:
    """Class to hold information about a single phase"""

    @property
    def reference_unit_cell(self):
        return self._ref_cell.copy()

    @reference_unit_cell.setter
    def reference_unit_cell(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Unit cell attribute should be a numpy array not {type(value)}")
        if not len(value) == 6:
            raise ValueError("Unit cell should have length 6")
        self._ref_cell = value

    @property
    def stiffness_constants(self) -> Dict:
        return self._stiffness_dict

    @stiffness_constants.setter
    def stiffness_constants(self, value: Dict):
        unique_list = 'c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66'
        # Value is a dictionary of stiffness constants
        for key in value.keys():
            if key not in unique_list:
                raise ValueError("Unknown stiffness key!")
        for entry in value.values():
            if not isinstance(entry, float):
                raise TypeError("Stiffness values must all be floats!")
        self._stiffness_dict = value
        self._stiffnessMV = None  # Reset the stiffness Mandel-Voigt tensor, so it can be recalculated with new constants

    @property
    def lattice(self):
        return self._lattice

    @property
    def stiffnessMV(self):
        if self.stiffness_constants is None:
            raise ValueError("No stiffness constants!")
        if self._stiffnessMV is None:
            crystal_system = self.symmetry.to_string()
            if crystal_system == "cubic":
                c11 = self.stiffness_constants["c11"]
                c12 = self.stiffness_constants["c12"]
                c44 = self.stiffness_constants["c44"]
                self._stiffnessMV = formStiffnessMV(crystal_system=crystal_system, c11=c11, c12=c12, c44=c44)
                return self._stiffnessMV
            elif crystal_system == "hexagonal":
                c11 = self.stiffness_constants["c11"]
                c12 = self.stiffness_constants["c12"]
                c13 = self.stiffness_constants["c13"]
                c33 = self.stiffness_constants["c33"]
                c44 = self.stiffness_constants["c44"]
                self._stiffnessMV = formStiffnessMV(crystal_system=crystal_system, c11=c11, c12=c12, c13=c13, c33=c33, c44=c44)
                return self._stiffnessMV
            else:
                raise ValueError("Structure not supported!")
        else:
            return self._stiffnessMV

    def __init__(self, name: str, reference_unit_cell: np.ndarray, symmetry: Symmetry, lattice: Lattice, stiffness_dict: Dict = None):
        """
        :param name: The name of the load step
        :param sample: The Sample instance for this load step
        :param applied_load: Not used yet.
        """
        self.name = name
        self.reference_unit_cell = reference_unit_cell
        self.symmetry = symmetry
        self._stiffnessMV = None
        if stiffness_dict is None:
            self._stiffness_dict = None
        else:
            self.stiffness_constants = stiffness_dict
        self._lattice = lattice

    @property
    def all_slip_systems(self):
        # if self.lattice._centering == "I":
        #     # BCC
        #     systems_123 = []
        #
        #     systems_111 = SlipSystem.get_slip_systems(slip_type='111', lattice=self.lattice)
        #     systems_112 = SlipSystem.get_slip_systems(slip_type='112', lattice=self.lattice)
        #
        #     # https://doi.org/10.1016/j.actamat.2019.11.030
        #
        #     # TODO: Look up CRSS of these
        #     # The below maybe aren't worth considering
        #
        #     systems_123.append(SlipSystem.from_indices((1, 2, -3), (1, 1, 1), self.lattice))  # C1
        #     systems_123.append(SlipSystem.from_indices((1, -3, 2), (1, 1, 1), self.lattice))  # C2
        #     systems_123.append(SlipSystem.from_indices((2, 1, -3), (1, 1, 1), self.lattice))  # C3
        #     systems_123.append(SlipSystem.from_indices((2, -3, 1), (1, 1, 1), self.lattice))  # C4
        #     systems_123.append(SlipSystem.from_indices((-3, 1, 2), (1, 1, 1), self.lattice))  # C5
        #     systems_123.append(SlipSystem.from_indices((-3, 2, 1), (1, 1, 1), self.lattice))  # C6
        #
        #     systems_123.append(SlipSystem.from_indices((1, -2, 3), (-1, 1, 1), self.lattice))  # C7
        #     systems_123.append(SlipSystem.from_indices((1, 3, -2), (-1, 1, 1), self.lattice))  # C8
        #     systems_123.append(SlipSystem.from_indices((2, -1, 3), (-1, 1, 1), self.lattice))  # C9
        #     systems_123.append(SlipSystem.from_indices((2, 3, -1), (-1, 1, 1), self.lattice))  # C10
        #     systems_123.append(SlipSystem.from_indices((3, 1, 2), (-1, 1, 1), self.lattice))  # C11
        #     systems_123.append(SlipSystem.from_indices((3, 2, 1), (-1, 1, 1), self.lattice))  # C12
        #
        #     systems_123.append(SlipSystem.from_indices((1, 2, 3), (-1, -1, 1), self.lattice))  # C13
        #     systems_123.append(SlipSystem.from_indices((-1, 3, 2), (-1, -1, 1), self.lattice))  # C14
        #     systems_123.append(SlipSystem.from_indices((2, 1, 3), (-1, -1, 1), self.lattice))  # C15
        #     systems_123.append(SlipSystem.from_indices((-2, 3, 1), (-1, -1, 1), self.lattice))  # C16
        #     systems_123.append(SlipSystem.from_indices((3, -1, 2), (-1, -1, 1), self.lattice))  # C17
        #     systems_123.append(SlipSystem.from_indices((3, -2, 1), (-1, -1, 1), self.lattice))  # C18
        #
        #     systems_123.append(SlipSystem.from_indices((-1, 2, 3), (1, -1, 1), self.lattice))  # C19
        #     systems_123.append(SlipSystem.from_indices((1, 3, 2), (1, -1, 1), self.lattice))  # C20
        #     systems_123.append(SlipSystem.from_indices((-2, 1, 3), (1, -1, 1), self.lattice))  # C21
        #     systems_123.append(SlipSystem.from_indices((2, 3, 1), (1, -1, 1), self.lattice))  # C22
        #     systems_123.append(SlipSystem.from_indices((-3, -1, 2), (1, -1, 1), self.lattice))  # C23
        #     systems_123.append(SlipSystem.from_indices((3, 2, -1), (1, -1, 1), self.lattice))  # C24
        #
        #     all_systems = systems_111 + systems_112 + systems_123
        #
        #     return all_systems
        if self.lattice._centering == "I":
            # BCC
            # Slip occurs on {110} planes in <111> directions
            # SlipSystem.from_indices(plane, direction)
            systems = [
                # 110
                SlipSystem.from_indices((0,  1, -1), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((1,  0, -1), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((1, -1,  0), (1, 1, 1), self.lattice),

                SlipSystem.from_indices((0, 1, -1), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((1, 0,  1), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((1, 1,  0), (-1, 1, 1), self.lattice),

                SlipSystem.from_indices((0, 1,  1), (1, -1, 1), self.lattice),
                SlipSystem.from_indices((1, 0, -1), (1, -1, 1), self.lattice),
                SlipSystem.from_indices((1, 1,  0), (1, -1, 1), self.lattice),

                SlipSystem.from_indices((0,  1, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices((1,  0, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices((1, -1, 0), (1, 1, -1), self.lattice),

                # 112
                SlipSystem.from_indices((-2,  1,  1), (1, 1, 1), self.lattice),
                SlipSystem.from_indices(( 1, -2,  1), (1, 1, 1), self.lattice),
                SlipSystem.from_indices(( 1,  1, -2), (1, 1, 1), self.lattice),

                SlipSystem.from_indices((2,  1,  1), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((1,  2, -1), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((1, -1,  2), (-1, 1, 1), self.lattice),

                SlipSystem.from_indices(( 2, 1, -1), (1, -1, 1), self.lattice),
                SlipSystem.from_indices(( 1, 2,  1), (1, -1, 1), self.lattice),
                SlipSystem.from_indices((-1, 1,  2), (1, -1, 1), self.lattice),

                SlipSystem.from_indices(( 2, -1, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices((-1,  2, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices(( 1,  1, 2), (1, 1, -1), self.lattice),

                # 123
                SlipSystem.from_indices((1,  2, -3), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((1, -3,  2), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((2,  1, -3), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((2, -3,  1), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((-3, 1,  2), (1, 1, 1), self.lattice),
                SlipSystem.from_indices((-3, 2,  1), (1, 1, 1), self.lattice),

                SlipSystem.from_indices((1, -2,  3), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((1,  3, -2), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((2, -1,  3), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((2,  3, -1), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((3,  1,  2), (-1, 1, 1), self.lattice),
                SlipSystem.from_indices((3,  2,  1), (-1, 1, 1), self.lattice),

                SlipSystem.from_indices(( 1, 3,  2), (1, -1, 1), self.lattice),
                SlipSystem.from_indices((-1, 2,  3), (1, -1, 1), self.lattice),
                SlipSystem.from_indices(( 2, 3,  1), (1, -1, 1), self.lattice),
                SlipSystem.from_indices((-2, 1,  3), (1, -1, 1), self.lattice),
                SlipSystem.from_indices(( 3, 1, -2), (1, -1, 1), self.lattice),
                SlipSystem.from_indices(( 3, 2, -1), (1, -1, 1), self.lattice),

                SlipSystem.from_indices(( 1,  2, 3), (1, 1, -1), self.lattice),
                SlipSystem.from_indices((-1,  3, 2), (1, 1, -1), self.lattice),
                SlipSystem.from_indices(( 2,  1, 3), (1, 1, -1), self.lattice),
                SlipSystem.from_indices((-2,  3, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices(( 3, -2, 1), (1, 1, -1), self.lattice),
                SlipSystem.from_indices(( 3, -1, 2), (1, 1, -1), self.lattice)
            ]

            return systems

        elif self.lattice._centering == "F":
            # FCC
            systems = [
                SlipSystem.from_indices((1, 1, 1), (1, -1, 0), self.lattice),
                SlipSystem.from_indices((1, 1, 1), (0, 1, -1), self.lattice),
                SlipSystem.from_indices((1, 1, 1), (1, 0, -1), self.lattice),

                SlipSystem.from_indices((-1, 1, 1), (1, 1, 0), self.lattice),
                SlipSystem.from_indices((-1, 1, 1), (0, 1, -1), self.lattice),
                SlipSystem.from_indices((-1, 1, 1), (1, 0, 1), self.lattice),

                SlipSystem.from_indices((1, 1, -1), (1, -1, 0), self.lattice),
                SlipSystem.from_indices((1, 1, -1), (0, 1, 1), self.lattice),
                SlipSystem.from_indices((1, 1, -1), (1, 0, 1), self.lattice),

                SlipSystem.from_indices((1, -1, 1), (1, 1, 0), self.lattice),
                SlipSystem.from_indices((1, -1, 1), (0, 1, 1), self.lattice),
                SlipSystem.from_indices((1, -1, 1), (1, 0, -1), self.lattice)
            ]

            return systems

        elif self.lattice._centering == "P" and self.symmetry.to_string() == "hexagonal":
            # HCP
            systems = []

            # http://dx.doi.org/10.1016/j.msea.2015.09.016

            # TODO:
            # Look for the ones with shortest Burgers vector
            # Get BV from direction
            # First 3 most likely
            # Have people done disloc. studies in eps martensite?

            # <a> Basal
            systems.append(SlipSystem.from_indices((0, 0, 0, 1), (-2, 1, 1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((0, 0, 0, 1), (1, -2, 1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((0, 0, 0, 1), (1, 1, -2, 0), self.lattice))

            # <a> Prism
            systems.append(SlipSystem.from_indices((0, 1, -1, 0), (-2, 1, 1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((1, 0, -1, 0), (-1, 2, -1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 1, 0, 0), (-1, -1, 2, 0), self.lattice))

            # <a> Pyram.
            systems.append(SlipSystem.from_indices((1, 0, -1, 1), (-1, 2, -1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((0, 1, -1, 1), (-2, 1, 1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 1, 0, 1), (-1, -1, 2, 0), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 0, 1, 1), (1, -2, 1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((0, -1, 1, 1), (2, -1, -1, 0), self.lattice))
            systems.append(SlipSystem.from_indices((1, -1, 0, 1), (1, 1, -2, 0), self.lattice))

            # <c + a> Pyram.(1st)
            systems.append(SlipSystem.from_indices((1, 0, -1, 1), (-2, 1, 1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((0, 1, -1, 1), (-1, -1, 2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 1, 0, 1), (1, -2, 1, 3), self.lattice))

            systems.append(SlipSystem.from_indices((-1, 0, 1, 1), (2, -1, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((0, -1, 1, 1), (1, 1, -2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((1, -1, 0, 1), (-1, 2, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((1, 0, -1, 1), (-1, -1, 2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((0, 1, -1, 1), (1, -2, 1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 1, 0, 1), (2, -1, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 0, 1, 1), (1, 1, -2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((0, -1, 1, 1), (-1, 2, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((1, -1, 0, 1), (-2, 1, 1, 3), self.lattice))

            # Least likely:

            # <c + a> Pyram.(2nd)
            systems.append(SlipSystem.from_indices((1, 1, -2, 2), (-1, -1, 2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-1, 2, -1, 2), (1, -2, 1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-2, 1, 1, 2), (2, -1, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((-1, -1, 2, 2), (1, 1, -2, 3), self.lattice))
            systems.append(SlipSystem.from_indices((1, -2, 1, 2), (-1, 2, -1, 3), self.lattice))
            systems.append(SlipSystem.from_indices((2, -1, -1, 2), (-2, 1, 1, 3), self.lattice))

            return systems
        else:
            raise ValueError("Can't get slip systems for this unsupported lattice!")


    @classmethod
    def from_id11_pars(cls, id11_path, name):
        # Use ImageD11 to read the parameter file
        id11_parameter_obj = parameters.read_par_file(filename=id11_path)
        cell_a = id11_parameter_obj.get("cell__a")
        cell_b = id11_parameter_obj.get("cell__b")
        cell_c = id11_parameter_obj.get("cell__c")
        alpha = id11_parameter_obj.get("cell_alpha")
        beta = id11_parameter_obj.get("cell_beta")
        gamma = id11_parameter_obj.get("cell_gamma")
        centering = id11_parameter_obj.get("cell_lattice_[P,A,B,C,I,F,R]")
        unit_cell_array = np.array([cell_a, cell_b, cell_c, alpha, beta, gamma])

        if alpha == 90 and beta == 90 and gamma == 90:
            # Assume cubic
            pymicro_symmetry = Symmetry.from_string("cubic")
        elif alpha == 90 and beta == 90 and gamma == 120:
            # Assume hexagonal
            pymicro_symmetry = Symmetry.from_string("hexagonal")
        else:
            raise ValueError("This symmetry is not supported!")

        pymicro_lattice = Lattice.from_parameters(a=cell_a, b=cell_b, c=cell_c, alpha=alpha, beta=beta, gamma=gamma, centering=centering, symmetry=pymicro_symmetry)

        return Phase(name=name, reference_unit_cell=unit_cell_array, symmetry=pymicro_symmetry, lattice=pymicro_lattice)

    def export_to_hdf5_group(self, container_group: h5py.Group) -> h5py.Group:
        this_phase_group = container_group.create_group(self.name)
        this_phase_group.create_dataset("reference_unit_cell", data=self.reference_unit_cell)
        this_phase_group.create_dataset("symmetry_string", data=self.symmetry.to_string(), dtype="S32")
        this_phase_group.create_dataset("centering", data=self.lattice._centering, dtype="S32")
        stiffness_constant_tuples_list = [(const, val) for (const, val) in self.stiffness_constants.items()]
        stiffness_constant_array = np.array(stiffness_constant_tuples_list, dtype=[('constant', 'S32'), ('value', '<f4')])

        this_phase_group.create_dataset("stiffness_constants", data=stiffness_constant_array)

        return this_phase_group

    @classmethod
    def import_from_hdf5_group(self, container_group: h5py.Group):
        phase_name = container_group.name.split("/")[-1]
        phase_ref_cell = container_group["reference_unit_cell"][:]
        phase_symmetry = Symmetry.from_string(container_group["symmetry_string"][()].decode("utf-8"))
        stiffness_constants_array = container_group["stiffness_constants"][:]
        stiffness_constants_dict = {entry[0].decode("utf-8"):float(entry[1]) for entry in stiffness_constants_array}
        lattice_centering = container_group["centering"][()].decode("utf-8")

        pymicro_lattice = Lattice.from_parameters(a=phase_ref_cell[0], b=phase_ref_cell[1], c=phase_ref_cell[2], alpha=phase_ref_cell[3], beta=phase_ref_cell[4], gamma=phase_ref_cell[5],
                                                  centering=lattice_centering, symmetry=phase_symmetry)

        phase_object = Phase(name=phase_name, reference_unit_cell=phase_ref_cell, symmetry=phase_symmetry, stiffness_dict=stiffness_constants_dict, lattice=pymicro_lattice)
        return phase_object

    def __eq__(self, other):
        if self is other:
            return True
        else:
            same_name = self.name == other.name
            same_ref_cell = np.allclose(self.reference_unit_cell, other.reference_unit_cell)
            same_symmetry = self.symmetry == other.symmetry
            same_lattice = self.lattice == other.lattice

            if same_name and same_ref_cell and same_symmetry and same_lattice:
                return True
            else:
                return False

    def __hash__(self):
        return hash((self.name, self.reference_unit_cell.tostring(), self.symmetry))