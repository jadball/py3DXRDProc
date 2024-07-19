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
#  WITHOUT ANY WARRANTY without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np
from pymicro.crystal.lattice import Symmetry, Lattice
from scipy.spatial.transform import Rotation
from xfab import tools

from py3DXRDProc.conversions import upper_triangular_to_symmetric, symmetric_to_upper_triangular, \
    custom_array_to_string, MVCOBMatrix, symmToMVvec, MVvecToSymm, strain2stress, eps_error_to_sig_error, \
    sig_error_to_sig_lab_error, check_S_N_OR, check_K_S_OR, check_N_W_OR, check_N_W_OR_numba, check_K_S_OR_numba, \
    check_N_W_OR_numba_parallel, check_K_S_OR_numba_parallel, check_S_N_OR_numba, check_S_N_OR_numba_parallel, \
    check_G_T_OR, check_Pitsch_OR, check_G_T_OR_numba, check_G_T_OR_numba_parallel, \
    check_Pitsch_OR_numba, check_Pitsch_OR_numba_parallel, get_K_S_OR_angle_numba_parallel, \
    are_grains_duplicate_array_numba_wrapper, disorientation_single_numba, disorientation_single_check_numba, \
    disorientation_single, are_grains_duplicate, are_grains_duplicate_stitching, \
    are_grains_duplicate_stitching_array_numba_wrapper, are_grains_embedded, are_grains_embedded_array_numba_wrapper, \
    get_S_N_variant, get_K_S_variant

from py3DXRDProc.grain import BaseMapGrain, RawGrain
from py3DXRDProc.grain_map import BaseGrainsMap, RawGrainsMap
from py3DXRDProc.grain_volume import GrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample


class TestDisorientationSingle(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

    def test_random_matrices(self):
        # check 100 random matrices
        U_A_array = Rotation.random(100).as_matrix()
        for U_A in U_A_array:
            # pick a random float from 0 to 45 degrees
            angle = np.random.rand() * 45
            U_B = U_A @ Rotation.from_euler('x', angle, degrees=True).as_matrix().T

            for symm_op in self.phase.symmetry.symmetry_operators():
                U_A_symm = U_A @ symm_op.T

                UBI_A = tools.u_to_ubi(U_A_symm, self.phase.reference_unit_cell)
                grain_A = RawGrain(gid=1,
                                   pos=np.array([1., 2., 3.]),
                                   UBI=UBI_A,
                                   volume=250.0,
                                   grain_map=self.raw_grain_map,
                                   mean_peak_intensity=1.0)

                UBI_B = tools.u_to_ubi(U_B, self.phase.reference_unit_cell)
                grain_B = RawGrain(gid=2,
                                   pos=np.array([1., 2., 3.]),
                                   UBI=UBI_B,
                                   volume=250.0,
                                   grain_map=self.raw_grain_map,
                                   mean_peak_intensity=1.0)

                misd = disorientation_single(grain_A, grain_B)
                self.assertAlmostEqual(misd, angle, places=5)

    def test_hardcoded_from_mtex(self):
        U_A = np.array([
            [0.2399, 0.7539, -0.6116],
            [0.8368, -0.4800, -0.2634],
            [-0.4921, -0.4486, -0.7460]
        ])

        U_B = np.array([
            [0.9442, -0.1000, -0.3139],
            [-0.0618, 0.8822, -0.4669],
            [0.3236, 0.4602, 0.8268]
        ])

        import xfab
        xfab.CHECKS.activated = False
        UBI_A = tools.u_to_ubi(U_A, self.phase.reference_unit_cell)
        grain_A = RawGrain(gid=1,
                           pos=np.array([1., 2., 3.]),
                           UBI=UBI_A,
                           volume=250.0,
                           grain_map=self.raw_grain_map,
                           mean_peak_intensity=1.0)

        UBI_B = tools.u_to_ubi(U_B, self.phase.reference_unit_cell)
        grain_B = RawGrain(gid=2,
                           pos=np.array([1., 2., 3.]),
                           UBI=UBI_B,
                           volume=250.0,
                           grain_map=self.raw_grain_map,
                           mean_peak_intensity=1.0)

        desired_misd = 46.3419
        calculated_misd = disorientation_single(grain_A, grain_B)
        self.assertAlmostEqual(calculated_misd, desired_misd, places=2)


class TestDisorientationSingleNumba(unittest.TestCase):
    def test_against_reference(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        # check 100 random matrices
        U_A = Rotation.random(1).as_matrix()[0]
        symmetries = Symmetry.cubic.symmetry_operators()
        # pick a random float from 0 to 45 degrees
        angle = np.random.rand() * 45
        U_B = U_A @ Rotation.from_euler('x', angle, degrees=True).as_matrix().T

        for symm_op in symmetries:
            U_A_symm = U_A @ symm_op.T

            misd_this = disorientation_single_numba((U_A_symm, U_B), symmetries)

            import xfab
            xfab.CHECKS.activated = False
            UBI_A = tools.u_to_ubi(U_A_symm, self.phase.reference_unit_cell)
            grain_A = RawGrain(gid=1,
                               pos=np.array([1., 2., 3.]),
                               UBI=UBI_A,
                               volume=250.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

            UBI_B = tools.u_to_ubi(U_B, self.phase.reference_unit_cell)
            grain_B = RawGrain(gid=2,
                               pos=np.array([1., 2., 3.]),
                               UBI=UBI_B,
                               volume=250.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

            misd_ref = disorientation_single(grain_A, grain_B)

            self.assertAlmostEqual(misd_this, misd_ref, places=5)

    def test_random_matrices(self):
        # check 100 random matrices
        U_A_array = Rotation.random(100).as_matrix()
        symmetries = Symmetry.cubic.symmetry_operators()
        for U_A in U_A_array:
            # pick a random float from 0 to 45 degrees
            angle = np.random.rand() * 45
            U_B = U_A @ Rotation.from_euler('x', angle, degrees=True).as_matrix().T

            for symm_op in symmetries:
                U_A_symm = U_A @ symm_op.T

                misd = disorientation_single_numba((U_A_symm, U_B), symmetries)
                self.assertAlmostEqual(misd, angle, places=5)

    def test_hardcoded_from_mtex(self):
        symmetries = Symmetry.cubic.symmetry_operators()
        U_A = np.array([
            [0.2399, 0.7539, -0.6116],
            [0.8368, -0.4800, -0.2634],
            [-0.4921, -0.4486, -0.7460]
        ])

        U_B = np.array([
            [0.9442, -0.1000, -0.3139],
            [-0.0618, 0.8822, -0.4669],
            [0.3236, 0.4602, 0.8268]
        ])

        desired_misd = 46.3419
        calculated_misd = disorientation_single_numba((U_A, U_B), symmetries)
        self.assertAlmostEqual(calculated_misd, desired_misd, places=2)


class TestDisorientationCheck(unittest.TestCase):
    def test_random_matrices(self):
        # check 100 random matrices
        U_A_array = Rotation.random(100).as_matrix()
        symmetries = Symmetry.cubic.symmetry_operators()
        for U_A in U_A_array:
            # pick a random float from 0 to 45 degrees
            angle = np.random.rand() * 45
            U_B = U_A @ Rotation.from_euler('z', angle, degrees=True).as_matrix().T

            for symm_op in symmetries:
                U_A_symm = U_A @ symm_op.T

                # should return true if our tolerance is 0.5deg larger than the angle
                self.assertTrue(disorientation_single_check_numba((U_A_symm, U_B), angle + 0.50, symmetries))
                # should return false if our tolerance is 0.5deg smaller than the angle
                self.assertFalse(disorientation_single_check_numba((U_A_symm, U_B), angle - 0.50, symmetries))


class TestAreGrainsDuplicate(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = self.raw_orientation_1 @ Rotation.from_euler('x', 1, degrees=True).as_matrix().T
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_5 = self.raw_orientation_1 @ Rotation.from_euler('x', 5, degrees=True).as_matrix().T

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_5 = tools.u_to_ubi(self.raw_orientation_5, self.phase.reference_unit_cell)

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=250.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1., 2., 3.1]),
                                      UBI=raw_ubi_2,
                                      volume=750.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([-1., -2., -3.]),
                                      UBI=raw_ubi_3,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_4 = RawGrain(gid=4,
                                      pos=np.array([-1., -2., -3.]),  # pos too far away
                                      UBI=raw_ubi_1,  # match grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_5 = RawGrain(gid=4,
                                      pos=np.array([1., 2., 3.005]),  # very close to grain 1
                                      UBI=raw_ubi_5,  # 5 degrees away from grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4, self.valid_grain_5])

    def test_matching_grains(self):
        self.assertTrue(are_grains_duplicate(self.valid_grain_1,
                                             self.valid_grain_2,  # 1 degree away from grain 1, 0.1 mm away
                                             dist_tol=0.11, angle_tol=1.1))

        self.assertFalse(are_grains_duplicate(self.valid_grain_1,
                                              self.valid_grain_2,
                                              dist_tol=0.09, angle_tol=0.9))

        self.assertFalse(are_grains_duplicate(self.valid_grain_1,
                                              self.valid_grain_3,  # too far away in position
                                              dist_tol=0.1, angle_tol=0.9))

        self.assertTrue(are_grains_duplicate(self.valid_grain_1,
                                             self.valid_grain_5,  # 5 degrees away
                                             dist_tol=0.01, angle_tol=5.05))

    def test_catches_same_grain_twice(self):
        with self.assertRaises(ValueError):
            are_grains_duplicate(self.valid_grain_1,
                                 self.valid_grain_1,  # same grain twice
                                 dist_tol=0.11, angle_tol=1.1)


class TestAreGrainsDuplicateArrayNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = self.raw_orientation_1 @ Rotation.from_euler('x', 1, degrees=True).as_matrix().T
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_5 = self.raw_orientation_1 @ Rotation.from_euler('x', 5, degrees=True).as_matrix().T

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_5 = tools.u_to_ubi(self.raw_orientation_5, self.phase.reference_unit_cell)

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=250.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1., 2., 3.1]),
                                      UBI=raw_ubi_2,
                                      volume=750.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([-1., -2., -3.]),
                                      UBI=raw_ubi_3,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_4 = RawGrain(gid=4,
                                      pos=np.array([-1., -2., -3.]),  # pos too far away
                                      UBI=raw_ubi_1,  # match grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_5 = RawGrain(gid=4,
                                      pos=np.array([1., 2., 3.005]),  # very close to grain 1
                                      UBI=raw_ubi_5,  # 5 degrees away from grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4, self.valid_grain_5])

    def test_valid_grains_list(self):
        matching_grain_pairs = are_grains_duplicate_array_numba_wrapper(self.raw_grain_map.grains, dist_tol=0.15,
                                                                        angle_tol=2.0)

        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 1]])))

    def test_valid_grains_list_wider(self):
        matching_grain_pairs = are_grains_duplicate_array_numba_wrapper(self.raw_grain_map.grains, dist_tol=0.15,
                                                                        angle_tol=6.5)

        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 1], [0, 4], [1, 4]])))


class TestAreGrainsDuplicateStitching(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = self.raw_orientation_1 @ Rotation.from_euler('x', 1, degrees=True).as_matrix().T
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_5 = self.raw_orientation_1 @ Rotation.from_euler('x', 5, degrees=True).as_matrix().T

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_5 = tools.u_to_ubi(self.raw_orientation_5, self.phase.reference_unit_cell)

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=250.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1.1, 2., 3.25]),
                                      UBI=raw_ubi_2,
                                      volume=750.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([-1., -2., -3.]),
                                      UBI=raw_ubi_3,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_4 = RawGrain(gid=4,
                                      pos=np.array([-1., -2., -3.]),  # pos too far away
                                      UBI=raw_ubi_1,  # match grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_5 = RawGrain(gid=4,
                                      pos=np.array([1., 2., 3.005]),  # very close to grain 1
                                      UBI=raw_ubi_5,  # 5 degrees away from grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4, self.valid_grain_5])

    def test_matching_grains(self):
        self.assertTrue(
            are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_2,  # 1 degree away from grain 1
                                           dist_tol_xy=0.11, dist_tol_z=0.26, angle_tol=1.1))

        self.assertFalse(are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_2,  # too far in xy
                                                        dist_tol_xy=0.09, dist_tol_z=0.26, angle_tol=1.1))

        self.assertFalse(are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_2,  # too far in z
                                                        dist_tol_xy=0.11, dist_tol_z=0.24, angle_tol=1.1))

        self.assertFalse(are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_2,  # too far in angle
                                                        dist_tol_xy=0.11, dist_tol_z=0.26, angle_tol=0.9))

        self.assertFalse(are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_3,  # too far in xy
                                                        dist_tol_xy=0.5, dist_tol_z=7.0, angle_tol=20.0))

        self.assertTrue(
            are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_5,  # 1 degree away from grain 1
                                           dist_tol_xy=0.0005, dist_tol_z=0.006, angle_tol=5.01))

        self.assertFalse(are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_5,  # angle too harsh
                                                        dist_tol_xy=0.0005, dist_tol_z=0.006, angle_tol=4.99))

    def test_catches_same_grain_twice(self):
        with self.assertRaises(ValueError):
            are_grains_duplicate_stitching(self.valid_grain_1, self.valid_grain_1,  # same grain twice
                                           dist_tol_xy=0.0005, dist_tol_z=0.006, angle_tol=4.99)


class TestAreGrainsDuplicateStitchingArrayNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = self.raw_orientation_1 @ Rotation.from_euler('x', 1, degrees=True).as_matrix().T
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_5 = self.raw_orientation_1 @ Rotation.from_euler('x', 5, degrees=True).as_matrix().T

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_5 = tools.u_to_ubi(self.raw_orientation_5, self.phase.reference_unit_cell)

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=250.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1.1, 2., 3.25]),
                                      UBI=raw_ubi_2,
                                      volume=750.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([-1., -2., -3.]),
                                      UBI=raw_ubi_3,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_4 = RawGrain(gid=4,
                                      pos=np.array([-1., -2., -3.]),  # pos too far away
                                      UBI=raw_ubi_1,  # match grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_5 = RawGrain(gid=4,
                                      pos=np.array([1., 2., 3.005]),  # very close to grain 1
                                      UBI=raw_ubi_5,  # 5 degrees away from grain 1
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4, self.valid_grain_5])

    def test_matching_grains(self):
        matching_grain_pairs = are_grains_duplicate_stitching_array_numba_wrapper(self.raw_grain_map.grains,
                                                                                  dist_tol_xy=0.11, dist_tol_z=0.26,
                                                                                  angle_tol=1.1)

        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 1]])))

        matching_grain_pairs = are_grains_duplicate_stitching_array_numba_wrapper(self.raw_grain_map.grains,
                                                                                  dist_tol_xy=0.09, dist_tol_z=0.26,
                                                                                  angle_tol=1.1)

        print(matching_grain_pairs)

        self.assertTrue(np.size(matching_grain_pairs) == 0)

        matching_grain_pairs = are_grains_duplicate_stitching_array_numba_wrapper(self.raw_grain_map.grains,
                                                                                  dist_tol_xy=0.11, dist_tol_z=0.24,
                                                                                  angle_tol=1.1)

        self.assertTrue(np.size(matching_grain_pairs) == 0)

        matching_grain_pairs = are_grains_duplicate_stitching_array_numba_wrapper(self.raw_grain_map.grains,
                                                                                  dist_tol_xy=0.11, dist_tol_z=0.26,
                                                                                  angle_tol=0.9)

        self.assertTrue(np.size(matching_grain_pairs) == 0)

        matching_grain_pairs = are_grains_duplicate_stitching_array_numba_wrapper(self.raw_grain_map.grains,
                                                                                  dist_tol_xy=0.11, dist_tol_z=0.3,
                                                                                  angle_tol=6.0)

        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 1], [0, 4], [1, 4]])))


class TestAreGrainsEmbedded(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        self.load_step.phase_volume_fractions = {"test_phase": 1.0}
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)

        radius_1 = 0.1  # 100 um
        radius_2 = 0.05  # 50 um
        radius_3 = 0.05  # 50 um

        volume_1 = (4 / 3) * np.pi * radius_1 ** 3
        volume_2 = (4 / 3) * np.pi * radius_2 ** 3
        volume_3 = (4 / 3) * np.pi * radius_3 ** 3

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=volume_1,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1.05, 2., 3.]),
                                      UBI=raw_ubi_2,
                                      volume=volume_2,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([1., 2.15, 3.]),
                                      UBI=raw_ubi_3,
                                      volume=volume_3,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3])

    def test_grains_inside(self):
        self.assertTrue(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_2, dist_const=1.00))  # dist_tol 0.10, dist 0.05
        self.assertTrue(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_2, dist_const=0.75))  # dist_tol 0.075, dist 0.05
        self.assertTrue(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_2, dist_const=0.51))  # dist_tol 0.051, dist 0.05
        self.assertFalse(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_2, dist_const=0.49))  # dist_tol 0.049, dist 0.05

    def test_grains_outside(self):
        self.assertTrue(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_3, dist_const=1.51))  # dist_tol 0.151, dist 0.15
        self.assertFalse(
            are_grains_embedded(self.valid_grain_1, self.valid_grain_3, dist_const=1.49))  # dist_tol 0.149, dist 0.15

    def test_catches_same_grain_twice(self):
        with self.assertRaises(ValueError):
            are_grains_embedded(self.valid_grain_1, self.valid_grain_1)


class TestAreGrainsEmbeddedArrayNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        self.load_step.phase_volume_fractions = {"test_phase": 1.0}
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)

        radius_1 = 0.1  # 100 um
        radius_2 = 0.05  # 50 um
        radius_3 = 0.05  # 50 um

        volume_1 = (4 / 3) * np.pi * radius_1 ** 3
        volume_2 = (4 / 3) * np.pi * radius_2 ** 3
        volume_3 = (4 / 3) * np.pi * radius_3 ** 3

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2., 3.]),
                                      UBI=raw_ubi_1,
                                      volume=volume_1,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1.05, 2., 3.]),
                                      UBI=raw_ubi_2,
                                      volume=volume_2,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_3 = RawGrain(gid=3,
                                      pos=np.array([1., 2.15, 3.]),
                                      UBI=raw_ubi_3,
                                      volume=volume_3,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.raw_grain_map.add_grains(
            [self.valid_grain_1, self.valid_grain_2, self.valid_grain_3])

    def test_grains(self):
        matching_grain_pairs = are_grains_embedded_array_numba_wrapper(self.raw_grain_map.grains,
                                                                       self.raw_grain_map.grains, dist_const=1.00)

        # grains will match against themselves, but valid_grain_1 and valid_grain_2 should match too
        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 0], [0, 1], [1, 1], [2, 2]])))

        matching_grain_pairs = are_grains_embedded_array_numba_wrapper(self.raw_grain_map.grains,
                                                                       self.raw_grain_map.grains, dist_const=0.49)

        # grains will match against themselves, tolerance now too tight
        self.assertTrue(np.array_equal(matching_grain_pairs, np.array([[0, 0], [1, 1], [2, 2]])))

        matching_grain_pairs = are_grains_embedded_array_numba_wrapper(self.raw_grain_map.grains,
                                                                       self.raw_grain_map.grains, dist_const=1.51)

        # grains will match against themselves, but
        # 0 and 1 will match: distance is 0.05 mm, distance tolerance is 0.1 * 1.51 = 0.151 mm
        # 0 and 2 will match: distance is 0.15 mm, distance tolerance is 0.1 * 1.51 = 0.151 mm
        # 1 and 0 will match: distance is 0.05 mm, distance tolerance is 0.05 * 1.51 = 0.0755 mm
        self.assertTrue(
            np.array_equal(matching_grain_pairs, np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 2]])))


class TestGetKSORAngleNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = Rotation.from_euler("X", 5.0, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [-0.6667, 0.7416, 0.0749],
            [-0.0749, -0.1667, 0.9832]
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        desired_angle = 5.0
        determined_angle = \
            get_K_S_OR_angle_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                            np.array([self.aprime_grain.U_sample]))[
                0]
        self.assertAlmostEqual(desired_angle, determined_angle, places=1)

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = Rotation.from_euler("X", 91.0, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = Rotation.from_euler("X", -6.0, degrees=True).as_matrix() @ np.array([
            [0.7416, 0.6498, 0.1667],
            [-0.6667, 0.7416, 0.0749],
            [-0.0749, -0.1667, 0.9832]
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        desired_angle = 7.0
        determined_angle = get_K_S_OR_angle_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                                           np.array([self.aprime_grain.U_sample]))[
            0]
        self.assertAlmostEqual(desired_angle, determined_angle, places=1)

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)

        U_gamma = np.array([[0.9285, 0.1363, -0.3455],
                            [-0.3510, 0.6263, -0.6961],
                            [0.1216, 0.7676, 0.6293]])

        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([[0.4480, 0.8369, 0.3146],
                             [-0.7713, 0.1838, 0.6094],
                             [0.4522, -0.5156, 0.7278]])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        desired_angle = 0.0
        determined_angle = get_K_S_OR_angle_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                                           np.array([self.aprime_grain.U_sample]))[
            0]
        self.assertAlmostEqual(desired_angle, determined_angle, places=1)

    def test_valid_OR_4(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)

        U_gamma = Rotation.from_euler("X", -6.0, degrees=True).as_matrix() @ np.array([[0.9285, 0.1363, -0.3455],
                                                                                       [-0.3510, 0.6263, -0.6961],
                                                                                       [0.1216, 0.7676, 0.6293]])

        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = Rotation.from_euler("X", 4.0, degrees=True).as_matrix() @ np.array([[0.4480, 0.8369, 0.3146],
                                                                                       [-0.7713, 0.1838, 0.6094],
                                                                                       [0.4522, -0.5156, 0.7278]])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        desired_angle = 10.0
        determined_angle = get_K_S_OR_angle_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                                           np.array([self.aprime_grain.U_sample]))[
            0]
        self.assertAlmostEqual(desired_angle, determined_angle, places=0)


class TestCheckPitschNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(
            check_Pitsch_OR_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                           np.array([self.aprime_grain.U_sample]),
                                           0.5), [True])

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
                np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
                np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]),
                0.5)), [True, True, False])

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
                np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
                np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]),
                1.0)), [True, True, False])

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.0)),
            [True, True, False])

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample,
                      self.gamma_grain.U_sample @ Rotation.from_euler("X", 90, degrees=True).as_matrix(),
                      self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.0)),
            [True, True, False])

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.0)),
            [False, False, False])

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_Pitsch_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 0.5)),
            [False, False, False])


class TestCheckPitschNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.75))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.5))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2.2))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_Pitsch_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))


class TestCheckPitsch(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.6685, -0.7434, -0.0192],
            [0.7410, 0.6681, -0.0671],
            [0.0627, 0.0306, 0.9976],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.0431, -0.1447, 0.9885],
             [0.9976, 0.0469, 0.0504],
             [-0.0536, 0.9884, 0.1423]],
            [[-0.3972, -0.7360, -0.5482],
             [-0.6506, 0.6471, -0.3974],
             [0.6472, 0.1988, -0.7359]],
            [[0.4592, 0.6622, 0.5922],
             [-0.5143, 0.7417, -0.4305],
             [-0.7243, -0.1069, 0.6811]],
            [[-0.0621, 0.1320, -0.9893],
             [0.9809, 0.1914, -0.0360],
             [0.1846, -0.9726, -0.1414]],
            [[-0.5684, 0.6197, 0.5412],
             [-0.4840, -0.7838, 0.3891],
             [0.6653, -0.0407, 0.7454]],
            [[0.6117, -0.5332, -0.5845],
             [-0.3296, -0.8433, 0.4244],
             [-0.7192, -0.0670, -0.6916]],
            [[-0.4518, -0.7846, 0.4246],
             [0.5068, -0.6174, -0.6016],
             [0.7342, -0.0566, 0.6766]],
            [[-0.3634, -0.8458, 0.3906],
             [-0.5489, 0.5332, 0.6438],
             [-0.7527, 0.0195, -0.6580]],
            [[0.9939, 0.0941, 0.0575],
             [0.0520, 0.0592, -0.9969],
             [-0.0972, 0.9938, 0.0540]],
            [[-0.6190, 0.6556, -0.4324],
             [0.3371, 0.7190, 0.6078],
             [0.7094, 0.2305, -0.6661]],
            [[0.5750, -0.7295, 0.3704],
             [0.4908, 0.6698, 0.5572],
             [-0.6545, -0.1386, 0.7432]],
            [[-0.9931, -0.1068, 0.0477],
             [-0.0664, 0.1791, -0.9816],
             [0.0963, -0.9780, -0.1849]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.6969, 0.7071, -0.1196],
            [-0.1691, 0, 0.9856],
            [0.6969, 0.7071, 0.1196],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 2.2))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_Pitsch_OR(self.gamma_grain, self.aprime_grain, 0.5))


class TestCheckGTORNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, -0.1625, 0.0342],
            [0.1363, 0.6743, -0.7258],
            [0.0948, 0.7204, 0.6871],
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(
            check_G_T_OR_numba_parallel(np.array([self.gamma_grain.U_sample]),
                                        np.array([self.aprime_grain.U_sample]),
                                        0.5), [True])

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
                np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
                np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]),
                0.5)), [True, True, False])

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
                np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
                np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]),
                1.0)), [True, True, False])

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.0)),
            [True, True, False])

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample,
                      self.gamma_grain.U_sample @ Rotation.from_euler("X", 90, degrees=True).as_matrix(),
                      self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.0)),
            [True, True, False])

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 2.4)),
            [False, False, False])

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_G_T_OR_numba_parallel(
            np.array([self.gamma_grain.U_sample, self.gamma_grain.U_sample, self.aprime_grain.U_sample]),
            np.array([self.aprime_grain.U_sample, self.aprime_grain.U_sample, self.aprime_grain.U_sample]), 0.5)),
            [False, False, False])


class TestCheckGTORNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, -0.1625, 0.0342],
            [0.1363, 0.6743, -0.7258],
            [0.0948, 0.7204, 0.6871],
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 0.1))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 2.4))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_G_T_OR_numba(self.gamma_grain.U_sample, self.aprime_grain.U_sample, 0.5))


class TestCheckGTOR(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, -0.1625, 0.0342],
            [0.1363, 0.6743, -0.7258],
            [0.0948, 0.7204, 0.6871],
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_G_T_OR(self.gamma_grain, self.aprime_grain, 0.1))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.8405, -0.4925, -0.2258],
            [-0.5416, -0.7520, -0.3758],
            [0.0153, 0.4381, -0.8988],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9174, 0.3582, -0.1736],
             [-0.3960, 0.8657, -0.3061],
             [0.0406, 0.3496, 0.9360]],
            [[-0.3695, -0.9009, -0.2277],
             [-0.4217, -0.0559, 0.9050],
             [-0.8281, 0.4304, -0.3592]],
            [[0.3839, 0.8849, -0.2639],
             [0.3680, 0.1154, 0.9226],
             [0.8469, -0.4513, -0.2814]],
            [[-0.6217, -0.6494, 0.4380],
             [-0.7034, 0.2167, -0.6770],
             [0.3447, -0.7289, -0.5915]],
            [[0.5962, 0.6777, 0.4305],
             [0.7414, -0.2590, -0.6191],
             [-0.3081, 0.6883, -0.6568]],
            [[-0.9063, -0.3705, -0.2033],
             [0.4116, -0.8831, -0.2254],
             [-0.0960, -0.2880, 0.9528]],
            [[0.2352, 0.2366, 0.9427],
             [0.2167, -0.9583, 0.1865],
             [0.9475, 0.1604, -0.2767]],
            [[-0.7831, 0.3060, -0.5414],
             [0.6009, 0.1484, -0.7854],
             [-0.1600, -0.9404, -0.3001]],
            [[0.8129, -0.3391, -0.4735],
             [-0.5542, -0.2003, -0.8079],
             [0.1791, 0.9192, -0.3508]],
            [[0.3752, -0.8377, -0.3968],
             [0.7417, 0.0146, 0.6706],
             [-0.5560, -0.5459, 0.6268]],
            [[-0.3497, 0.8094, -0.4717],
             [-0.7797, 0.0276, 0.6255],
             [0.5193, 0.5866, 0.6214]],
            [[-0.2905, -0.1753, 0.9407],
             [-0.2254, 0.9680, 0.1107],
             [-0.9300, -0.1799, -0.3207]],
            [[0.3926, -0.4841, 0.7820],
             [0.8510, 0.5137, -0.1092],
             [-0.3489, 0.7083, 0.6137]],
            [[-0.1548, 0.2486, -0.9562],
             [-0.5157, -0.8459, -0.1364],
             [-0.8427, 0.4719, 0.2591]],
            [[0.2101, -0.3100, -0.9272],
             [0.5244, 0.8362, -0.1608],
             [0.8252, -0.4525, 0.3382]],
            [[0.7403, -0.6313, 0.2311],
             [-0.6722, -0.6897, 0.2690],
             [-0.0104, -0.3545, -0.9350]],
            [[-0.7513, 0.6436, 0.1458],
             [0.6566, 0.7071, 0.2625],
             [0.0659, 0.2930, -0.9539]],
            [[-0.4368, 0.5332, 0.7245],
             [-0.8441, -0.5214, -0.1252],
             [0.3110, -0.6662, 0.6778]],
            [[0.8448, 0.0660, 0.5311],
             [-0.5140, 0.3763, 0.7708],
             [-0.1490, -0.9241, 0.3518]],
            [[-0.5346, -0.3732, -0.7582],
             [-0.6390, 0.7657, 0.0737],
             [0.5531, 0.5239, -0.6478]],
            [[0.5788, 0.3241, -0.7483],
             [0.6321, -0.7581, 0.1606],
             [-0.5152, -0.5660, -0.6436]],
            [[0.0614, -0.9582, 0.2793],
             [0.3730, -0.2375, -0.8969],
             [0.9258, 0.1592, 0.3429]],
            [[-0.0758, 0.9743, 0.2123],
             [-0.3194, 0.1780, -0.9308],
             [-0.9446, -0.1384, 0.2977]],
            [[-0.8745, -0.0329, 0.4838],
             [0.4672, -0.3244, 0.8225],
             [0.1299, 0.9454, 0.2991]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_G_T_OR(self.gamma_grain, self.aprime_grain, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR(self.gamma_grain, self.aprime_grain, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_G_T_OR(self.gamma_grain, self.aprime_grain, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.9861, 0.1363, 0.0948],
            [-0.1625, 0.6743, 0.7204],
            [0.0342, -0.7258, 0.6871],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_G_T_OR(self.gamma_grain, self.aprime_grain, 2.4))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_G_T_OR(self.gamma_grain, self.aprime_grain, 0.5))


class TestCheckNWORNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(
            check_N_W_OR_numba_parallel(np.array([self.gamma_grain.UB_sample]), np.array([self.aprime_grain.UB_sample]),
                                        0.5), [True])

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
                np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
                np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]),
                0.75)), [True, True, False])
            # self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
                np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
                np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]),
                1.0)), [True, True, False])

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("Y", -180, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 1.5)),
            [True, True, False])

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample,
                      self.gamma_grain.UB_sample @ Rotation.from_euler("X", 90, degrees=True).as_matrix(),
                      self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 2.0)),
            [True, True, False])

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 2.49)),
            [False, False, False])

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_N_W_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 0.5)),
            [False, False, False])


class TestCheckNWORNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.8))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("Y", -180, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.5))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2.49))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_N_W_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))


class TestCheckNWOR(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_N_W_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9359, 0.3480, 0.0538],
            [-0.3244, 0.9114, -0.2532],
            [-0.1372, 0.2195, 0.9659]
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[-0.4157, 0.2066, 0.8857],
             [0.8738, -0.1793, 0.4520],
             [0.2522, 0.9619, -0.1060]]
            ,
            [[-0.2080, -0.9705, -0.1218],
             [-0.8235, 0.2410, -0.5136],
             [0.5278, -0.0066, -0.8493]]
            ,
            [[0.6238, 0.4614, 0.6309],
             [-0.0503, 0.8292, -0.5566],
             [-0.7800, 0.3154, 0.5405]]
            ,
            [[-0.2841, 0.8873, -0.3633],
             [-0.4655, -0.4589, -0.7568],
             [-0.8382, -0.0459, 0.5434]]
            ,
            [[0.9079, -0.1233, -0.4006],
             [0.4151, 0.3973, 0.8184],
             [0.0582, -0.9094, 0.4119]]
            ,
            [[-0.6238, -0.2247, 0.7486],
             [0.0503, -0.9674, -0.2484],
             [0.7800, -0.1172, 0.6147]]
            ,
            [[-0.9079, -0.0173, -0.4188],
             [-0.4151, -0.1018, 0.9041],
             [-0.0582, 0.9947, 0.0852]]
            ,
            [[0.6999, -0.2375, 0.6736],
             [-0.4084, -0.9068, 0.1045],
             [0.5860, -0.3482, -0.7316]]
            ,
            [[0.2080, -0.8744, 0.4383],
             [0.8235, 0.3984, 0.4039],
             [-0.5278, 0.2769, 0.8030]]
            ,
            [[-0.6999, 0.4485, 0.5559],
             [0.4084, 0.8898, -0.2037],
             [-0.5860, 0.0844, -0.8059]]
            ,
            [[0.2841, -0.9576, -0.0468],
             [0.4655, 0.1804, -0.8665],
             [0.8382, 0.2244, 0.4970]]
            ,
            [[0.4157, 0.1005, 0.9039],
             [-0.8738, 0.3197, 0.3663],
             [-0.2522, -0.9422, 0.2207]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_N_W_OR(self.gamma_grain, self.aprime_grain, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR(self.gamma_grain, self.aprime_grain, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_N_W_OR(self.gamma_grain, self.aprime_grain, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [-0.7071, 0.7071, 0.0000],
            [0.1196, 0.1196, 0.9856],
            [0.6969, 0.6969, -0.1691]
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_N_W_OR(self.gamma_grain, self.aprime_grain, 2.49))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_N_W_OR(self.gamma_grain, self.aprime_grain, 0.5))


class TestCheckKSORNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [-0.0749, -0.1667, 0.9832],
            [0.6667, -0.7416, -0.0749]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(
            check_K_S_OR_numba_parallel(np.array([self.gamma_grain.UB_sample]), np.array([self.aprime_grain.UB_sample]),
                                        0.5), [True])

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
                np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
                np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]),
                0.5)), [True, True, False])

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
                np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
                np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]),
                1.0)), [True, True, False])

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 1.1)),
            [True, True, False])

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample,
                      self.gamma_grain.UB_sample @ Rotation.from_euler("X", 90, degrees=True).as_matrix(),
                      self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 2.0)),
            [True, True, False])

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 2.3)),
            [False, False, False])

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertSequenceEqual(list(check_K_S_OR_numba_parallel(
            np.array([self.gamma_grain.UB_sample, self.gamma_grain.UB_sample, self.aprime_grain.UB_sample]),
            np.array([self.aprime_grain.UB_sample, self.aprime_grain.UB_sample, self.aprime_grain.UB_sample]), 0.5)),
            [False, False, False])


class TestCheckKSORNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [-0.0749, -0.1667, 0.9832],
            [0.6667, -0.7416, -0.0749]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 2.3))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_K_S_OR_numba(self.gamma_grain.UB_sample, self.aprime_grain.UB_sample, 0.5))


class TestCheckKSOR(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [-0.0749, -0.1667, 0.9832],
            [0.6667, -0.7416, -0.0749]
        ]).T
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_K_S_OR(self.gamma_grain, self.aprime_grain, 0.5))

    def test_valid_OR_3(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # no need to invert these
        # test all variants

        for U_aprime in np.array([
            [[0.9050, 0.3565, 0.2323],
             [-0.2985, 0.9209, -0.2505],
             [-0.3032, 0.1574, 0.9398]]
            ,
            [[-0.2188, 0.3296, -0.9184],
             [0.8805, -0.3390, -0.3314],
             [-0.4206, -0.8812, -0.2161]]
            ,
            [[-0.5286, 0.8175, -0.2287],
             [-0.5681, -0.1405, 0.8109],
             [0.6308, 0.5586, 0.5387]]
            ,
            [[0.8737, -0.4724, -0.1164],
             [-0.4449, -0.8725, 0.2021],
             [-0.1970, -0.1248, -0.9724]]
            ,
            [[-0.3764, -0.2909, 0.8796],
             [0.8666, 0.2252, 0.4453],
             [-0.3276, 0.9299, 0.1673]]
            ,
            [[-0.6549, -0.7404, 0.1516],
             [-0.4356, 0.2058, -0.8763],
             [0.6176, -0.6399, -0.4573]]
            ,
            [[-0.2099, 0.1472, 0.9666],
             [0.2705, -0.9413, 0.2021],
             [0.9396, 0.3038, 0.1577]]
            ,
            [[-0.4763, -0.8334, -0.2804],
             [-0.8524, 0.3593, 0.3798],
             [-0.2158, 0.4199, -0.8815]]
            ,
            [[0.6067, 0.6332, 0.4806],
             [0.5650, 0.0819, -0.8211],
             [-0.5593, 0.7696, -0.3081]]
            ,
            [[-0.1005, -0.1270, -0.9868],
             [0.4136, 0.8967, -0.1575],
             [0.9049, -0.4240, -0.0376]]
            ,
            [[-0.3968, 0.8903, 0.2235],
             [-0.8354, -0.2494, -0.4898],
             [-0.3803, -0.3811, 0.8427]]
            ,
            [[0.5767, -0.7103, -0.4035],
             [0.4388, -0.1473, 0.8864],
             [-0.6891, -0.6883, 0.2267]]
            ,
            [[0.3816, -0.9240, -0.0247],
             [0.8920, 0.3752, -0.2523],
             [0.2424, 0.0743, 0.9673]]
            ,
            [[-0.7267, 0.5789, 0.3698],
             [0.1210, 0.6378, -0.7606],
             [-0.6762, -0.5080, -0.5336]]
            ,
            [[0.3599, -0.0158, -0.9329],
             [-0.8302, 0.4508, -0.3279],
             [0.4257, 0.8925, 0.1491]]
            ,
            [[0.4914, 0.8671, 0.0816],
             [0.7959, -0.4851, 0.3623],
             [0.3537, -0.1131, -0.9285]]
            ,
            [[-0.7415, -0.4832, -0.4654],
             [-0.0617, -0.6417, 0.7645],
             [-0.6681, 0.5956, 0.4460]]
            ,
            [[0.2353, -0.0230, 0.9716],
             [-0.9169, -0.3369, 0.2141],
             [0.3225, -0.9412, -0.1004]]
            ,
            [[0.7897, 0.5904, -0.1668],
             [-0.0676, 0.3539, 0.9328],
             [0.6098, -0.7254, 0.3194]]
            ,
            [[0.2415, 0.4408, -0.8645],
             [-0.3634, -0.7849, -0.5018],
             [-0.8998, 0.4354, -0.0294]]
            ,
            [[-0.8585, 0.5061, -0.0825],
             [0.3883, 0.7467, 0.5401],
             [0.3349, 0.4316, -0.8376]]
            ,
            [[0.6786, -0.6861, 0.2624],
             [0.0083, -0.3501, -0.9367],
             [0.7345, 0.6378, -0.2318]]
            ,
            [[0.0688, -0.4611, 0.8847],
             [-0.3207, 0.8295, 0.4573],
             [-0.9447, -0.3152, -0.0908]]
            ,
            [[-0.9201, -0.3902, -0.0334],
             [0.3551, -0.7951, -0.4916],
             [0.1653, -0.4642, 0.8702]]
        ]):
            UB_aprime = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertTrue(check_K_S_OR(self.gamma_grain, self.aprime_grain, 1.0))

    def test_valid_OR_rotated(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make 1 degree off
        U_gamma = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ])
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR(self.gamma_grain, self.aprime_grain, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        # make some complex rotation
        U_gamma = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertTrue(check_K_S_OR(self.gamma_grain, self.aprime_grain, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = np.array([
            [0.7416, 0.6498, 0.1667],
            [0.0749, 0.1667, -0.9832],
            [-0.6667, 0.7416, 0.0749]
        ]) @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_K_S_OR(self.gamma_grain, self.aprime_grain, 2.3))

    def test_invalid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)
        # invert this because
        U_aprime = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_aprime = U_aprime @ B_aprime
        UBI_aprime = np.linalg.inv(UB_aprime)

        self.aprime_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_aprime,
            volume=100.0,
            gid=1,
            grain_map=self.aprime_grain_map)

        self.assertFalse(check_K_S_OR(self.gamma_grain, self.aprime_grain, 0.5))


class TestCheckSNORNumbaParallel(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.cubic_phase = Phase(name="test_cubic_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.hex_phase = Phase(name="test_hex_phase", reference_unit_cell=np.array([1, 1, 3, 90, 90, 120]),
                               symmetry=Symmetry.hexagonal, lattice=Lattice.hexagonal(1, 3))
        self.cubic_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.cubic_phase)
        self.hex_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.hex_phase)

    def test_valid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
            np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
            np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
            1.0)), [True, True, False])

    def test_valid_OR_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.4082, -0.7071, 0.5774],
                [0.4082, 0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
            ]),
            np.array([
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
            ])
        ]:
            UB_hex = U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
                np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
                np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
                0.5)), [True, True, False])

    def test_valid_OR_3(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.1705, -0.8931, 0.4163],
                [0.7975, 0.3733, 0.4741],
                [-0.5788, 0.2512, 0.7758],
            ]),
            np.array([
                [-0.5962, 0.1558, 0.7876],
                [-0.7803, -0.3435, -0.5227],
                [0.1891, -0.9261, 0.3264],
            ]),
            np.array([
                [-0.5982, -0.4384, -0.6708],
                [0.5237, -0.8475, 0.0869],
                [-0.6066, -0.2993, 0.7365],
            ]),
            np.array([
                [0.6056, -0.7372, 0.2995],
                [-0.4138, 0.0298, 0.9099],
                [-0.6797, -0.6750, -0.2871],
            ])
        ]:
            UB_hex = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
                np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
                np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
                0.4)), [True, True, False])

    def test_valid_OR_rotated(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make 1 degree off
        U_cubic = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
            np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
            np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
            1.1)), [True, True, False])

    def test_valid_OR_rotated_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make some complex rotation
        U_cubic = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
            np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
            np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
            2.0)), [True, True, False])

    def test_invalid_OR_rotated_too_far(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
            np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
            np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
            2.4)), [False, False, False])

    def test_invalid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertSequenceEqual(list(check_S_N_OR_numba_parallel(
            np.array([self.cubic_grain.U_sample, self.cubic_grain.U_sample, self.hex_grain.U_sample]),
            np.array([self.hex_grain.U_sample, self.hex_grain.U_sample, self.hex_grain.U_sample]),
            0.5)), [False, False, False])


class TestCheckSNORNumba(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.cubic_phase = Phase(name="test_cubic_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.hex_phase = Phase(name="test_hex_phase", reference_unit_cell=np.array([1, 1, 3, 90, 90, 120]),
                               symmetry=Symmetry.hexagonal, lattice=Lattice.hexagonal(1, 3))
        self.cubic_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.cubic_phase)
        self.hex_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.hex_phase)

    def test_valid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 0.5))

    def test_valid_OR_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.4082, -0.7071, 0.5774],
                [0.4082, 0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
            ]),
            np.array([
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
            ])
        ]:
            UB_hex = U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertTrue(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 0.5))

    def test_valid_OR_3(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.1705, -0.8931, 0.4163],
                [0.7975, 0.3733, 0.4741],
                [-0.5788, 0.2512, 0.7758],
            ]),
            np.array([
                [-0.5962, 0.1558, 0.7876],
                [-0.7803, -0.3435, -0.5227],
                [0.1891, -0.9261, 0.3264],
            ]),
            np.array([
                [-0.5982, -0.4384, -0.6708],
                [0.5237, -0.8475, 0.0869],
                [-0.6066, -0.2993, 0.7365],
            ]),
            np.array([
                [0.6056, -0.7372, 0.2995],
                [-0.4138, 0.0298, 0.9099],
                [-0.6797, -0.6750, -0.2871],
            ])
        ]:
            UB_hex = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertTrue(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 0.4))

    def test_valid_OR_rotated(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make 1 degree off
        U_cubic = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 1.1))

    def test_valid_OR_rotated_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make some complex rotation
        U_cubic = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertFalse(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 2.4))

    def test_invalid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertFalse(check_S_N_OR_numba(self.cubic_grain.U_sample, self.hex_grain.U_sample, 0.5))


class TestCheckSNOR(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.cubic_phase = Phase(name="test_cubic_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.hex_phase = Phase(name="test_hex_phase", reference_unit_cell=np.array([1, 1, 3, 90, 90, 120]),
                               symmetry=Symmetry.hexagonal, lattice=Lattice.hexagonal(1, 3))
        self.cubic_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.cubic_phase)
        self.hex_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.hex_phase)

    def test_valid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR(self.cubic_grain, self.hex_grain, 0.5))

    def test_valid_OR_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.4082, -0.7071, 0.5774],
                [0.4082, 0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
            ]),
            np.array([
                [-0.4082, -0.7071, -0.5774],
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
            ]),
            np.array([
                [0.4082, -0.7071, 0.5774],
                [-0.8165, -0.0000, 0.5774],
                [-0.4082, -0.7071, -0.5774],
            ])
        ]:
            UB_hex = U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertTrue(check_S_N_OR(self.cubic_grain, self.hex_grain, 0.5))

    def test_valid_OR_3(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.array([
            [0.9415, -0.3215, 0.1011],
            [0.3353, 0.8632, -0.3774],
            [0.0341, 0.3892, 0.9205],
        ])
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # no need to invert these
        # test all variants

        for U_hex in [
            np.array([
                [0.1705, -0.8931, 0.4163],
                [0.7975, 0.3733, 0.4741],
                [-0.5788, 0.2512, 0.7758],
            ]),
            np.array([
                [-0.5962, 0.1558, 0.7876],
                [-0.7803, -0.3435, -0.5227],
                [0.1891, -0.9261, 0.3264],
            ]),
            np.array([
                [-0.5982, -0.4384, -0.6708],
                [0.5237, -0.8475, 0.0869],
                [-0.6066, -0.2993, 0.7365],
            ]),
            np.array([
                [0.6056, -0.7372, 0.2995],
                [-0.4138, 0.0298, 0.9099],
                [-0.6797, -0.6750, -0.2871],
            ])
        ]:
            UB_hex = Rotation.from_euler("XYZ", [0.05, -0.1, 0.4], degrees=True).as_matrix() @ U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertTrue(check_S_N_OR(self.cubic_grain, self.hex_grain, 0.4))

    def test_valid_OR_rotated(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make 1 degree off
        U_cubic = Rotation.from_euler("Z", 91, degrees=True).as_matrix()
        # U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR(self.cubic_grain, self.hex_grain, 1.5))

    def test_valid_OR_rotated_2(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        # make some complex rotation
        U_cubic = Rotation.from_euler("XYZ", [0.2, 0.1, -0.4], degrees=True).as_matrix()
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("XYZ", [-0.4, 1.2, -0.3], degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertTrue(check_S_N_OR(self.cubic_grain, self.hex_grain, 2))

    def test_invalid_OR_rotated_too_far(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = np.array([
            [0.4082, 0.4082, -0.8165],
            [-0.7071, 0.7071, -0.0000],
            [0.5774, 0.5774, 0.5774],
        ]).T @ Rotation.from_euler("X", 2.5, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertFalse(check_S_N_OR(self.cubic_grain, self.hex_grain, 2.4))

    def test_invalid_OR(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)
        # invert this because
        U_hex = Rotation.from_euler("X", 45, degrees=True).as_matrix()
        UB_hex = U_hex @ B_hex
        UBI_hex = np.linalg.inv(UB_hex)

        self.hex_grain = BaseMapGrain(
            pos=np.array([1.1, 2, 3]),
            UBI=UBI_hex,
            volume=100.0,
            gid=1,
            grain_map=self.hex_grain_map)

        self.assertFalse(check_S_N_OR(self.cubic_grain, self.hex_grain, 0.5))


# TODO: check rotated variant detection
class TestGetKSORVariant(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.gamma_phase = Phase(name="test_gamma_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.aprime_phase = Phase(name="test_aprime_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                  symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.gamma_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.gamma_phase)
        self.aprime_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.aprime_phase)

    def test_valid_OR(self):
        unitcell_gamma = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_gamma = tools.form_b_mat(unitcell_gamma)
        U_gamma = np.eye(3)
        UB_gamma = U_gamma @ B_gamma
        UBI_gamma = np.linalg.inv(UB_gamma)

        self.gamma_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_gamma,
            volume=100.0,
            gid=1,
            grain_map=self.gamma_grain_map)

        unitcell_aprime = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_aprime = tools.form_b_mat(unitcell_aprime)

        K_S_variant_matrices = np.array([
            [[0.7416, 0.6498, 0.1667],
             [-0.6667, 0.7416, 0.0749],
             [-0.0749, -0.1667, 0.9832]]
            ,
            [[0.0749, 0.1667, -0.9832],
             [0.6667, -0.7416, -0.0749],
             [-0.7416, -0.6498, -0.1667]]
            ,
            [[-0.6667, 0.7416, 0.0749],
             [-0.0749, -0.1667, 0.9832],
             [0.7416, 0.6498, 0.1667]]
            ,
            [[0.6667, -0.7416, -0.0749],
             [-0.7416, -0.6498, -0.1667],
             [0.0749, 0.1667, -0.9832]]
            ,
            [[-0.0749, -0.1667, 0.9832],
             [0.7416, 0.6498, 0.1667],
             [-0.6667, 0.7416, 0.0749]]
            ,
            [[-0.7416, -0.6498, -0.1667],
             [0.0749, 0.1667, -0.9832],
             [0.6667, -0.7416, -0.0749]]
            ,
            [[-0.0749, -0.1667, 0.9832],
             [0.6667, -0.7416, -0.0749],
             [0.7416, 0.6498, 0.1667]]
            ,
            [[-0.7416, -0.6498, -0.1667],
             [-0.6667, 0.7416, 0.0749],
             [0.0749, 0.1667, -0.9832]]
            ,
            [[0.7416, 0.6498, 0.1667],
             [0.0749, 0.1667, -0.9832],
             [-0.6667, 0.7416, 0.0749]]
            ,
            [[0.0749, 0.1667, -0.9832],
             [0.7416, 0.6498, 0.1667],
             [0.6667, -0.7416, -0.0749]]
            ,
            [[-0.6667, 0.7416, 0.0749],
             [-0.7416, -0.6498, -0.1667],
             [-0.0749, -0.1667, 0.9832]]
            ,
            [[0.6667, -0.7416, -0.0749],
             [-0.0749, -0.1667, 0.9832],
             [-0.7416, -0.6498, -0.1667]]
            ,
            [[0.6667, -0.7416, -0.0749],
             [0.7416, 0.6498, 0.1667],
             [-0.0749, -0.1667, 0.9832]]
            ,
            [[-0.6667, 0.7416, 0.0749],
             [0.0749, 0.1667, -0.9832],
             [-0.7416, -0.6498, -0.1667]]
            ,
            [[0.0749, 0.1667, -0.9832],
             [-0.6667, 0.7416, 0.0749],
             [0.7416, 0.6498, 0.1667]]
            ,
            [[0.7416, 0.6498, 0.1667],
             [0.6667, -0.7416, -0.0749],
             [0.0749, 0.1667, -0.9832]]
            ,
            [[-0.7416, -0.6498, -0.1667],
             [-0.0749, -0.1667, 0.9832],
             [-0.6667, 0.7416, 0.0749]]
            ,
            [[-0.0749, -0.1667, 0.9832],
             [-0.7416, -0.6498, -0.1667],
             [0.6667, -0.7416, -0.0749]]
            ,
            [[0.7416, 0.6498, 0.1667],
             [-0.0749, -0.1667, 0.9832],
             [0.6667, -0.7416, -0.0749]]
            ,
            [[0.0749, 0.1667, -0.9832],
             [-0.7416, -0.6498, -0.1667],
             [-0.6667, 0.7416, 0.0749]]
            ,
            [[-0.6667, 0.7416, 0.0749],
             [0.7416, 0.6498, 0.1667],
             [0.0749, 0.1667, -0.9832]]
            ,
            [[0.6667, -0.7416, -0.0749],
             [0.0749, 0.1667, -0.9832],
             [0.7416, 0.6498, 0.1667]]
            ,
            [[-0.0749, -0.1667, 0.9832],
             [-0.6667, 0.7416, 0.0749],
             [-0.7416, -0.6498, -0.1667]]
            ,
            [[-0.7416, -0.6498, -0.1667],
             [0.6667, -0.7416, -0.0749],
             [-0.0749, -0.1667, 0.9832]]
        ])

        for inc, var in enumerate(K_S_variant_matrices):
            U_aprime = var
            UB_aprime = U_aprime @ B_aprime
            UBI_aprime = np.linalg.inv(UB_aprime)

            self.aprime_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_aprime,
                volume=100.0,
                gid=1,
                grain_map=self.aprime_grain_map)

            self.assertEqual(get_K_S_variant(self.gamma_grain, self.aprime_grain), inc)


class TestGetSNORVariant(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make two blank phases
        self.cubic_phase = Phase(name="test_cubic_phase", reference_unit_cell=np.array([1, 1, 1, 90, 90, 90]),
                                 symmetry=Symmetry.cubic, lattice=Lattice.cubic(1))
        self.hex_phase = Phase(name="test_hex_phase", reference_unit_cell=np.array([1, 1, 3, 90, 90, 120]),
                               symmetry=Symmetry.hexagonal, lattice=Lattice.hexagonal(1, 3))
        self.cubic_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.cubic_phase)
        self.hex_grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.hex_phase)

    def test_valid_OR_all_variants(self):
        unitcell_cubic = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, 90., 90., 90.])
        B_cubic = tools.form_b_mat(unitcell_cubic)
        U_cubic = np.eye(3)
        UB_cubic = U_cubic @ B_cubic
        UBI_cubic = np.linalg.inv(UB_cubic)

        self.cubic_grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=UBI_cubic,
            volume=100.0,
            gid=1,
            grain_map=self.cubic_grain_map)

        unitcell_hex = np.array([2 * np.pi, 2 * np.pi, 3 * 2 * np.pi, 90., 90., 120.])
        B_hex = tools.form_b_mat(unitcell_hex)

        S_N_variant_matrices = np.array(
            [[[0.4082, -0.7071, 0.5774],
              [0.4082, 0.7071, 0.5774],
              [-0.8165, 0.0000, 0.5774]]
                ,
             [[-0.8165, 0.0000, 0.5774],
              [-0.4082, -0.7071, -0.5774],
              [0.4082, -0.7071, 0.5774]]
                ,
             [[-0.4082, -0.7071, -0.5774],
              [0.4082, -0.7071, 0.5774],
              [-0.8165, 0.0000, 0.5774]]
                ,
             [[0.4082, -0.7071, 0.5774],
              [-0.8165, 0.0000, 0.5774],
              [-0.4082, -0.7071, -0.5774]]]
        )

        for inc, var in enumerate(S_N_variant_matrices):
            U_hex = var
            UB_hex = U_hex @ B_hex
            UBI_hex = np.linalg.inv(UB_hex)

            self.hex_grain = BaseMapGrain(
                pos=np.array([1.1, 2, 3]),
                UBI=UBI_hex,
                volume=100.0,
                gid=1,
                grain_map=self.hex_grain_map)

            self.assertEqual(get_S_N_variant(self.cubic_grain, self.hex_grain), inc)




class TestUpperTriangularToSymmetric(unittest.TestCase):
    def test_valid_matrix(self):
        valid_matrix = np.array([1., 2, 3, 4, 5, 6])
        symmetric_result = upper_triangular_to_symmetric(upper_tri=valid_matrix)
        desired_result = np.array([[1., 6, 5], [6, 2, 4], [5, 4, 3]])
        self.assertTrue(np.allclose(symmetric_result, desired_result))

    def test_wrong_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            symmetric_result = upper_triangular_to_symmetric(upper_tri=invalid_matrix_type)

    def test_wrong_matrix_shape(self):
        invalid_matrix_shape_too_many_dimensions = np.array([[1., 2, 3, 4, 5, 6]])
        with self.assertRaises(ValueError):
            symmetric_result = upper_triangular_to_symmetric(upper_tri=invalid_matrix_shape_too_many_dimensions)

        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            symmetric_result = upper_triangular_to_symmetric(upper_tri=invalid_matrix_shape_just_wrong)

    def test_wrong_matrix_dtype(self):
        invalid_matrix_dtype = np.array([1, 2, 3, 4, 5, 6], dtype=int)
        with self.assertRaises(TypeError):
            symmetric_result = upper_triangular_to_symmetric(upper_tri=invalid_matrix_dtype)


class TestSymmetricToUpperTriangular(unittest.TestCase):
    def test_valid_matrix(self):
        valid_matrix = np.array([[1., 2, 3],
                                 [2, 4, 6],
                                 [3, 6, 5]])
        upper_triangular_result = symmetric_to_upper_triangular(symmetric=valid_matrix)
        desired_result = np.array([1., 4, 5, 6, 3, 2])
        self.assertTrue(np.allclose(upper_triangular_result, desired_result))

    def test_wrong_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            upper_triangular_result = symmetric_to_upper_triangular(symmetric=invalid_matrix_type)

    def test_wrong_matrix_shape(self):
        invalid_matrix_shape_too_many_dimensions = np.array([[1., 2, 3, 4, 5, 6]])
        with self.assertRaises(ValueError):
            upper_triangular_result = symmetric_to_upper_triangular(symmetric=invalid_matrix_shape_too_many_dimensions)

        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            upper_triangular_result = symmetric_to_upper_triangular(symmetric=invalid_matrix_shape_just_wrong)

    def test_wrong_matrix_dtype(self):
        invalid_matrix_dtype = np.array([[1, 2, 3],
                                         [2, 4, 6],
                                         [3, 6, 5]], dtype=int)
        with self.assertRaises(TypeError):
            upper_triangular_result = symmetric_to_upper_triangular(symmetric=invalid_matrix_dtype)

    def test_non_symmetric_matrix(self):
        non_symmetric_matrix = np.array([[1., 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]])
        with self.assertRaises(ValueError):
            upper_triangular_result = symmetric_to_upper_triangular(symmetric=non_symmetric_matrix)


class TestCustomArrayToString(unittest.TestCase):
    def test_right_attribute_array(self):
        UBI = np.array([
            [2.3994061, -1.56941634, -0.118949452],
            [1.04918116, 1.75653739, -2.01200437],
            [1.17321589, 1.63886394, 2.04256107]
        ])
        calced_string = custom_array_to_string(input_array=UBI)
        desired_string = "2.39940610 -1.56941634 -0.11894945 1.04918116 1.75653739 -2.01200437 1.17321589 1.63886394 2.04256107"
        self.assertEqual(calced_string, desired_string)

    def test_wrong_array_type(self):
        invalid_array_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            custom_array_string = custom_array_to_string(input_array=invalid_array_type)


class TestMVCOBMatrix(unittest.TestCase):
    def test_identity_matrix(self):
        self.assertTrue(np.allclose(MVCOBMatrix(np.identity(3)), np.identity(6)))

    def test_random_matrix(self):
        R = Rotation.random().as_matrix()
        T = np.zeros((6, 6), dtype='float64')

        T[0, 0] = R[0, 0] ** 2
        T[0, 1] = R[0, 1] ** 2
        T[0, 2] = R[0, 2] ** 2
        T[0, 3] = np.sqrt(2) * R[0, 1] * R[0, 2]
        T[0, 4] = np.sqrt(2) * R[0, 0] * R[0, 2]
        T[0, 5] = np.sqrt(2) * R[0, 0] * R[0, 1]
        T[1, 0] = R[1, 0] ** 2
        T[1, 1] = R[1, 1] ** 2
        T[1, 2] = R[1, 2] ** 2
        T[1, 3] = np.sqrt(2) * R[1, 1] * R[1, 2]
        T[1, 4] = np.sqrt(2) * R[1, 0] * R[1, 2]
        T[1, 5] = np.sqrt(2) * R[1, 0] * R[1, 1]
        T[2, 0] = R[2, 0] ** 2
        T[2, 1] = R[2, 1] ** 2
        T[2, 2] = R[2, 2] ** 2
        T[2, 3] = np.sqrt(2) * R[2, 1] * R[2, 2]
        T[2, 4] = np.sqrt(2) * R[2, 0] * R[2, 2]
        T[2, 5] = np.sqrt(2) * R[2, 0] * R[2, 1]
        T[3, 0] = np.sqrt(2) * R[1, 0] * R[2, 0]
        T[3, 1] = np.sqrt(2) * R[1, 1] * R[2, 1]
        T[3, 2] = np.sqrt(2) * R[1, 2] * R[2, 2]
        T[3, 3] = R[1, 2] * R[2, 1] + R[1, 1] * R[2, 2]
        T[3, 4] = R[1, 2] * R[2, 0] + R[1, 0] * R[2, 2]
        T[3, 5] = R[1, 1] * R[2, 0] + R[1, 0] * R[2, 1]
        T[4, 0] = np.sqrt(2) * R[0, 0] * R[2, 0]
        T[4, 1] = np.sqrt(2) * R[0, 1] * R[2, 1]
        T[4, 2] = np.sqrt(2) * R[0, 2] * R[2, 2]
        T[4, 3] = R[0, 2] * R[2, 1] + R[0, 1] * R[2, 2]
        T[4, 4] = R[0, 2] * R[2, 0] + R[0, 0] * R[2, 2]
        T[4, 5] = R[0, 1] * R[2, 0] + R[0, 0] * R[2, 1]
        T[5, 0] = np.sqrt(2) * R[0, 0] * R[1, 0]
        T[5, 1] = np.sqrt(2) * R[0, 1] * R[1, 1]
        T[5, 2] = np.sqrt(2) * R[0, 2] * R[1, 2]
        T[5, 3] = R[0, 2] * R[1, 1] + R[0, 1] * R[1, 2]
        T[5, 4] = R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0]
        T[5, 5] = R[0, 1] * R[1, 0] + R[0, 0] * R[1, 1]

        self.assertTrue(np.allclose(T, MVCOBMatrix(R=R)))

    def test_wrong_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            R = MVCOBMatrix(R=invalid_matrix_type)

    def test_wrong_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            R = MVCOBMatrix(R=invalid_matrix_shape_just_wrong)

    def test_wrong_matrix_dtype(self):
        invalid_matrix_dtype = np.array([[1, 2, 3],
                                         [2, 4, 6],
                                         [3, 6, 5]], dtype=int)
        with self.assertRaises(TypeError):
            R = MVCOBMatrix(R=invalid_matrix_dtype)


class TestSymmToMVVec(unittest.TestCase):
    def test_identity_matrix(self):
        self.assertTrue(np.allclose(symmToMVvec(np.identity(3)), np.array([1., 1, 1, 0, 0, 0])))

    def test_random_matrix(self):
        R = Rotation.random().as_matrix()
        A = np.dot(R, R.T)
        mvvec = np.zeros(6, dtype='float64')
        mvvec[0] = A[0, 0]
        mvvec[1] = A[1, 1]
        mvvec[2] = A[2, 2]
        mvvec[3] = np.sqrt(2.) * A[1, 2]
        mvvec[4] = np.sqrt(2.) * A[0, 2]
        mvvec[5] = np.sqrt(2.) * A[0, 1]

        self.assertTrue(np.allclose(mvvec, symmToMVvec(A=A)))

    def test_wrong_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            mvvec = symmToMVvec(A=invalid_matrix_type)

    def test_wrong_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            mvvec = symmToMVvec(A=invalid_matrix_shape_just_wrong)

    def test_wrong_matrix_dtype(self):
        invalid_matrix_dtype = np.array([[1, 2, 3],
                                         [2, 4, 6],
                                         [3, 6, 5]], dtype=int)
        with self.assertRaises(TypeError):
            mvvec = symmToMVvec(A=invalid_matrix_dtype)

    def test_non_symmetric_matrix(self):
        non_symmetric_matrix = np.array([[1., 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]])
        with self.assertRaises(ValueError):
            mvvec = symmToMVvec(A=non_symmetric_matrix)


class TestMVVecToSymm(unittest.TestCase):
    def test_easy_vector(self):
        input_vector = np.array([0., 1, 2, 3, 4, 5])
        output_matrix = MVvecToSymm(A=input_vector)
        desired_output_matrix = np.array([[0., 5 * (1 / np.sqrt(2)), 4 * (1 / np.sqrt(2))],
                                          [5 * (1 / np.sqrt(2)), 1, 3 * (1 / np.sqrt(2))],
                                          [4 * (1 / np.sqrt(2)), 3 * (1 / np.sqrt(2)), 2]])
        self.assertTrue(np.allclose(output_matrix, desired_output_matrix))

    def test_wrong_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            symm = MVvecToSymm(A=invalid_matrix_type)

    def test_wrong_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            symm = MVvecToSymm(A=invalid_matrix_shape_just_wrong)

    def test_wrong_matrix_dtype(self):
        invalid_matrix_dtype = np.array([1, 2, 3, 4, 5, 6], dtype=int)
        with self.assertRaises(TypeError):
            symm = MVvecToSymm(A=invalid_matrix_dtype)

    def test_round_robin(self):
        R = Rotation.random().as_matrix()
        A = np.dot(R, R.T)
        output = MVvecToSymm(symmToMVvec(A=A))
        self.assertTrue(np.allclose(A, output))


class TestStrainToStress(unittest.TestCase):
    def setUp(self):
        self.E = 200e9
        self.v = 0.30

        # Hooke's Law:
        front_constant = self.E / ((1 + self.v) * (1 - 2 * self.v))
        self.c11 = front_constant * (1 - self.v)
        self.c12 = front_constant * self.v
        self.c44 = front_constant * (1 - 2 * self.v)

        self.C = np.array([[self.c11, self.c12, self.c12, 0, 0, 0],
                           [self.c12, self.c11, self.c12, 0, 0, 0],
                           [self.c12, self.c12, self.c11, 0, 0, 0],
                           [0, 0, 0, self.c44, 0, 0],
                           [0, 0, 0, 0, self.c44, 0],
                           [0, 0, 0, 0, 0, self.c44]])

        self.valid_strain_tensor = np.array([[1.0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]])

    def test_uniaxial_stress(self):
        # Uniaxial stress condition

        sigma_axial = 100.0e6  # 100 MPa stress
        sigma_lateral = 0.0

        # Convert to strain

        eps_axial = sigma_axial / self.E
        eps_lateral = -self.v * eps_axial

        # Uniaxial stress tensor
        desired_stress_tensor = np.array([[sigma_axial, 0, 0],
                                          [0, sigma_lateral, 0],
                                          [0, 0, sigma_lateral]])

        # Equivalent strain tensor
        uniaxial_strain_tensor = np.array([[eps_axial, 0, 0],
                                           [0, eps_lateral, 0],
                                           [0, 0, eps_lateral]])

        calculated_stress_tensor = strain2stress(uniaxial_strain_tensor, self.C)

        print("\n")
        print(uniaxial_strain_tensor)
        print(desired_stress_tensor / 1e6)
        print(calculated_stress_tensor / 1e6)

        self.assertTrue(np.allclose(desired_stress_tensor, calculated_stress_tensor))

    def test_uniaxial_strain(self):
        # Uniaxial strain condition

        eps_axial = 1.0 / 100  # 1% axial strain
        eps_lateral = 0

        # Convert to stress

        uniaxial_strain_modulus = (self.E * (1 - self.v)) / ((1 + self.v) * (1 - 2 * self.v))
        beta = self.v / (1 - self.v)

        sigma_axial = uniaxial_strain_modulus * eps_axial
        sigma_lateral = beta * sigma_axial

        uniaxial_strain_tensor = np.array([[eps_axial, 0, 0],
                                           [0, eps_lateral, 0],
                                           [0, 0, eps_lateral]])

        # Equivalent stress tensor
        desired_stress_tensor = np.array([[sigma_axial, 0, 0],
                                          [0, sigma_lateral, 0],
                                          [0, 0, sigma_lateral]])

        calculated_stress_tensor = strain2stress(uniaxial_strain_tensor, self.C)

        print("\n")
        print(uniaxial_strain_tensor)
        print(desired_stress_tensor / 1e6)
        print(calculated_stress_tensor / 1e6)

        self.assertTrue(np.allclose(desired_stress_tensor, calculated_stress_tensor))

    def test_wrong_eps_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig = strain2stress(epsilon=invalid_matrix_type, C=self.C)

    def test_wrong_eps_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            sig = strain2stress(epsilon=invalid_matrix_shape_just_wrong, C=self.C)

    def test_wrong_eps_matrix_dtype(self):
        invalid_matrix_dtype = np.array([1, 2, 3, 4, 5, 6], dtype=int)
        with self.assertRaises(TypeError):
            sig = strain2stress(epsilon=self.valid_strain_tensor.astype(int), C=self.C)

    def test_non_symmetric_eps_matrix(self):
        non_symmetric_matrix = np.array([[1., 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]])
        with self.assertRaises(ValueError):
            sig = strain2stress(epsilon=non_symmetric_matrix, C=self.C)

    def test_wrong_C_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig = strain2stress(epsilon=self.valid_strain_tensor, C=invalid_matrix_type)

    def test_wrong_C_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            sig = strain2stress(epsilon=self.valid_strain_tensor, C=invalid_matrix_shape_just_wrong)

    def test_wrong_C_matrix_dtype(self):
        invalid_matrix_dtype = np.array([[11, 12, 12, 0, 0, 0],
                                         [12, 11, 12, 0, 0, 0],
                                         [12, 12, 11, 0, 0, 0],
                                         [0, 0, 0, 44, 0, 0],
                                         [0, 0, 0, 0, 44, 0],
                                         [0, 0, 0, 0, 0, 44]], dtype=int)
        with self.assertRaises(TypeError):
            sig = strain2stress(epsilon=self.valid_strain_tensor, C=invalid_matrix_dtype)

    def test_non_symmetric_C_matrix(self):
        non_symmetric_matrix = np.array([[11, 12, 12, 0, 0, 0.],
                                         [12, 11, 12, 0, 0, 0],
                                         [12, 12, 11, 0, 0, 0],
                                         [0, 0, 0, 44, 0, 0],
                                         [0, 0, 0, 0, 44, 0],
                                         [0, 0, 5, 0, 0, 44]])
        with self.assertRaises(ValueError):
            sig = strain2stress(epsilon=self.valid_strain_tensor, C=non_symmetric_matrix)


class TestEpsErrorToSigError(unittest.TestCase):
    def setUp(self):
        self.E = 200e3  # in MPa
        self.v = 0.30

        # Hooke's Law:
        front_constant = self.E / ((1 + self.v) * (1 - 2 * self.v))
        self.c11 = front_constant * (1 - self.v)
        self.c12 = front_constant * self.v
        self.c44 = front_constant * (1 - 2 * self.v)

        self.C = np.array([[self.c11, self.c12, self.c12, 0, 0, 0],
                           [self.c12, self.c11, self.c12, 0, 0, 0],
                           [self.c12, self.c12, self.c11, 0, 0, 0],
                           [0, 0, 0, self.c44, 0, 0],
                           [0, 0, 0, 0, self.c44, 0],
                           [0, 0, 0, 0, 0, self.c44]])

        self.valid_eps_error = np.identity(3) * 1.0e-5

    def test_valid_error(self):
        expected_sig_error = np.identity(3) * 3.1482126
        calculated_sig_error = eps_error_to_sig_error(self.valid_eps_error, self.C)
        self.assertTrue(np.allclose(expected_sig_error, calculated_sig_error))

    def test_actual_calculation(self):
        # Make some random small 3x3 error matrix
        random_error_matrix = np.random.rand(3, 3) * np.sqrt(1e-4)
        # Make it symmetric
        random_error_matrix_symmetric = random_error_matrix @ random_error_matrix.T
        # Symmetric error matrix with elements all around 1e-5 to 1e-4
        print('\n')
        print(random_error_matrix_symmetric)
        deps00 = random_error_matrix_symmetric[0, 0]
        deps01 = random_error_matrix_symmetric[0, 1]
        deps02 = random_error_matrix_symmetric[0, 2]
        deps10 = random_error_matrix_symmetric[1, 0]
        deps11 = random_error_matrix_symmetric[1, 1]
        deps12 = random_error_matrix_symmetric[1, 2]
        deps20 = random_error_matrix_symmetric[2, 0]
        deps21 = random_error_matrix_symmetric[2, 1]
        deps22 = random_error_matrix_symmetric[2, 2]

        C = self.C

        # Manually calculate it from the original commit
        # This makes sure if we break the error calculation, it'll be caught
        sig_error = np.zeros((3, 3), dtype='float64')

        sig_error[0, 0] = np.sqrt(
            np.power(C[0, 0] * deps00, 2) +
            np.power(C[0, 1] * deps11, 2) +
            np.power(C[0, 2] * deps22, 2) +
            2 * np.power(C[0, 3] * deps12, 2) +
            2 * np.power(C[0, 4] * deps02, 2) +
            2 * np.power(C[0, 5] * deps01, 2)
        )

        sig_error[0, 1] = np.sqrt(
            (1 / 2.) * np.power(C[5, 0] * deps00, 2) +
            (1 / 2.) * np.power(C[5, 1] * deps11, 2) +
            (1 / 2.) * np.power(C[5, 2] * deps22, 2) +
            np.power(C[5, 3] * deps12, 2) +
            np.power(C[5, 4] * deps02, 2) +
            np.power(C[5, 5] * deps01, 2)
        )

        sig_error[0, 2] = np.sqrt(
            (1 / 2.) * np.power(C[4, 0] * deps00, 2) +
            (1 / 2.) * np.power(C[4, 1] * deps11, 2) +
            (1 / 2.) * np.power(C[4, 2] * deps22, 2) +
            np.power(C[4, 3] * deps12, 2) +
            np.power(C[4, 4] * deps02, 2) +
            np.power(C[4, 5] * deps01, 2)
        )

        sig_error[1, 0] = sig_error[0, 1]

        sig_error[1, 1] = np.sqrt(
            np.power(C[1, 0] * deps00, 2) +
            np.power(C[1, 1] * deps11, 2) +
            np.power(C[1, 2] * deps22, 2) +
            2 * np.power(C[1, 3] * deps12, 2) +
            2 * np.power(C[1, 4] * deps02, 2) +
            2 * np.power(C[1, 5] * deps01, 2)
        )

        sig_error[1, 2] = np.sqrt(
            (1 / 2.) * np.power(C[3, 0] * deps00, 2) +
            (1 / 2.) * np.power(C[3, 1] * deps11, 2) +
            (1 / 2.) * np.power(C[3, 2] * deps22, 2) +
            np.power(C[3, 3] * deps12, 2) +
            np.power(C[3, 4] * deps02, 2) +
            np.power(C[3, 5] * deps01, 2)
        )

        sig_error[2, 0] = sig_error[0, 2]

        sig_error[2, 1] = sig_error[1, 2]

        sig_error[2, 2] = np.sqrt(
            np.power(C[2, 0] * deps00, 2) +
            np.power(C[2, 1] * deps11, 2) +
            np.power(C[2, 2] * deps22, 2) +
            2 * np.power(C[2, 3] * deps12, 2) +
            2 * np.power(C[2, 4] * deps02, 2) +
            2 * np.power(C[2, 5] * deps01, 2)
        )

        calculated_sig_error = eps_error_to_sig_error(eps_error=random_error_matrix_symmetric, stiffnessMV=C)

        print(sig_error)
        print(calculated_sig_error)

        self.assertTrue(np.allclose(sig_error, calculated_sig_error))

    def test_wrong_eps_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_error = eps_error_to_sig_error(eps_error=invalid_matrix_type, stiffnessMV=self.C)

    def test_wrong_eps_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            sig_error = eps_error_to_sig_error(eps_error=invalid_matrix_shape_just_wrong, stiffnessMV=self.C)

    def test_wrong_eps_matrix_dtype(self):
        invalid_matrix_dtype = self.valid_eps_error.astype(int)
        with self.assertRaises(TypeError):
            sig_error = eps_error_to_sig_error(eps_error=invalid_matrix_dtype, stiffnessMV=self.C)

    def test_non_symmetric_eps_matrix(self):
        non_symmetric_matrix = self.valid_eps_error
        non_symmetric_matrix[1, 2] = 1
        with self.assertRaises(ValueError):
            sig_error = eps_error_to_sig_error(eps_error=non_symmetric_matrix, stiffnessMV=self.C)

    def test_wrong_C_matrix_type(self):
        invalid_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_error = eps_error_to_sig_error(eps_error=self.valid_eps_error, stiffnessMV=invalid_matrix_type)

    def test_wrong_C_matrix_shape(self):
        invalid_matrix_shape_just_wrong = np.array([1., 2, 3, 4, 5, 6, 7])
        with self.assertRaises(ValueError):
            sig_error = eps_error_to_sig_error(eps_error=self.valid_eps_error,
                                               stiffnessMV=invalid_matrix_shape_just_wrong)

    def test_wrong_C_matrix_dtype(self):
        invalid_matrix_dtype = np.array([[11, 12, 12, 0, 0, 0],
                                         [12, 11, 12, 0, 0, 0],
                                         [12, 12, 11, 0, 0, 0],
                                         [0, 0, 0, 44, 0, 0],
                                         [0, 0, 0, 0, 44, 0],
                                         [0, 0, 0, 0, 0, 44]], dtype=int)
        with self.assertRaises(TypeError):
            sig_error = eps_error_to_sig_error(eps_error=self.valid_eps_error, stiffnessMV=invalid_matrix_dtype)

    def test_non_symmetric_C_matrix(self):
        non_symmetric_matrix = np.array([[11, 12, 12, 0, 0, 0.],
                                         [12, 11, 12, 0, 0, 0],
                                         [12, 12, 11, 0, 0, 0],
                                         [0, 0, 0, 44, 0, 0],
                                         [0, 0, 0, 0, 44, 0],
                                         [0, 0, 5, 0, 0, 44]])
        with self.assertRaises(ValueError):
            sig_error = eps_error_to_sig_error(eps_error=self.valid_eps_error, stiffnessMV=non_symmetric_matrix)


class TestSigErrorToSigSError(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_U = Rotation.random().as_matrix()
        self.valid_U_error = self.valid_U / 100  # 1% error
        random_strain_matrix_not_symm = np.random.rand(3, 3) * np.sqrt(1e-3)
        self.valid_sig = random_strain_matrix_not_symm @ random_strain_matrix_not_symm.T
        self.valid_sig_error = self.valid_sig / 100  # 1% error

    def test_valid_inputs(self):
        sig_lab_error = np.zeros((3, 3), dtype='float64')

        U00 = self.valid_U[0, 0]
        U01 = self.valid_U[0, 1]
        U02 = self.valid_U[0, 2]
        U10 = self.valid_U[1, 0]
        U11 = self.valid_U[1, 1]
        U12 = self.valid_U[1, 2]
        U20 = self.valid_U[2, 0]
        U21 = self.valid_U[2, 1]
        U22 = self.valid_U[2, 2]

        dU00 = self.valid_U_error[0, 0]
        dU01 = self.valid_U_error[0, 1]
        dU02 = self.valid_U_error[0, 2]
        dU10 = self.valid_U_error[1, 0]
        dU11 = self.valid_U_error[1, 1]
        dU12 = self.valid_U_error[1, 2]
        dU20 = self.valid_U_error[2, 0]
        dU21 = self.valid_U_error[2, 1]
        dU22 = self.valid_U_error[2, 2]

        # Fractional errors:

        fU00 = dU00 / U00
        fU01 = dU01 / U01
        fU02 = dU02 / U02
        fU10 = dU10 / U10
        fU11 = dU11 / U11
        fU12 = dU12 / U12
        fU20 = dU20 / U20
        fU21 = dU21 / U21
        fU22 = dU22 / U22

        sig00 = self.valid_sig[0, 0]
        sig01 = self.valid_sig[0, 1]
        sig02 = self.valid_sig[0, 2]
        sig10 = self.valid_sig[1, 0]
        sig11 = self.valid_sig[1, 1]
        sig12 = self.valid_sig[1, 2]
        sig20 = self.valid_sig[2, 0]
        sig21 = self.valid_sig[2, 1]
        sig22 = self.valid_sig[2, 2]

        dsig00 = self.valid_sig_error[0, 0]
        dsig01 = self.valid_sig_error[0, 1]
        dsig02 = self.valid_sig_error[0, 2]
        dsig10 = self.valid_sig_error[1, 0]
        dsig11 = self.valid_sig_error[1, 1]
        dsig12 = self.valid_sig_error[1, 2]
        dsig20 = self.valid_sig_error[2, 0]
        dsig21 = self.valid_sig_error[2, 1]
        dsig22 = self.valid_sig_error[2, 2]

        fsig00 = dsig00 / sig00
        fsig01 = dsig01 / sig01
        fsig02 = dsig02 / sig02
        fsig10 = dsig10 / sig10
        fsig11 = dsig11 / sig11
        fsig12 = dsig12 / sig12
        fsig20 = dsig20 / sig20
        fsig21 = dsig21 / sig21
        fsig22 = dsig22 / sig22

        sig_lab_error[0, 0] = np.sqrt(
            ((((U00 ** 2) * sig00) ** 2) * ((2 * fU00) ** 2 + fsig00 ** 2)) +
            ((((U01 ** 2) * sig11) ** 2) * ((2 * fU01) ** 2 + fsig11 ** 2)) +
            ((((U02 ** 2) * sig22) ** 2) * ((2 * fU02) ** 2 + fsig22 ** 2)) +
            (4 * (((U00 * U01 * sig01) ** 2) * (fU00 ** 2 + fU01 ** 2 + fsig01 ** 2))) +
            (4 * (((U00 * U02 * sig02) ** 2) * (fU00 ** 2 + fU02 ** 2 + fsig02 ** 2))) +
            (4 * (((U01 * U02 * sig12) ** 2) * (fU01 ** 2 + fU02 ** 2 + fsig12 ** 2)))
        )

        sig_lab_error[1, 1] = np.sqrt(
            ((((U10 ** 2) * sig00) ** 2) * ((2 * fU10) ** 2 + fsig00 ** 2)) +
            ((((U11 ** 2) * sig11) ** 2) * ((2 * fU11) ** 2 + fsig11 ** 2)) +
            ((((U12 ** 2) * sig22) ** 2) * ((2 * fU12) ** 2 + fsig22 ** 2)) +
            (4 * (((U10 * U11 * sig01) ** 2) * (fU10 ** 2 + fU11 ** 2 + fsig01 ** 2))) +
            (4 * (((U10 * U12 * sig02) ** 2) * (fU10 ** 2 + fU12 ** 2 + fsig02 ** 2))) +
            (4 * (((U11 * U12 * sig12) ** 2) * (fU11 ** 2 + fU12 ** 2 + fsig12 ** 2)))
        )

        sig_lab_error[2, 2] = np.sqrt(
            ((((U20 ** 2) * sig00) ** 2) * ((2 * fU20) ** 2 + fsig00 ** 2)) +
            ((((U21 ** 2) * sig11) ** 2) * ((2 * fU21) ** 2 + fsig11 ** 2)) +
            ((((U22 ** 2) * sig22) ** 2) * ((2 * fU22) ** 2 + fsig22 ** 2)) +
            (4 * (((U20 * U21 * sig01) ** 2) * (fU20 ** 2 + fU21 ** 2 + fsig01 ** 2))) +
            (4 * (((U20 * U22 * sig02) ** 2) * (fU20 ** 2 + fU22 ** 2 + fsig02 ** 2))) +
            (4 * (((U21 * U22 * sig12) ** 2) * (fU21 ** 2 + fU22 ** 2 + fsig12 ** 2)))
        )

        sig_lab_error[0, 1] = np.sqrt(
            ((sig01 ** 2) * ((U00 * U11 + U01 * U10) ** 2) * ((((((U00 * U11) ** 2) * (fU00 ** 2 + fU11 ** 2)) + (
                    ((U01 * U10) ** 2) * (fU01 ** 2 + fU10 ** 2))) / ((
                                                                              U00 * U11 + U01 * U10) ** 2)) + fsig01 ** 2)) +
            ((sig02 ** 2) * ((U00 * U12 + U02 * U10) ** 2) * ((((((U00 * U12) ** 2) * (fU00 ** 2 + fU12 ** 2)) + (
                    ((U02 * U10) ** 2) * (fU02 ** 2 + fU10 ** 2))) / ((
                                                                              U00 * U12 + U02 * U10) ** 2)) + fsig02 ** 2)) +
            ((sig12 ** 2) * ((U01 * U12 + U02 * U11) ** 2) * ((((((U01 * U12) ** 2) * (fU01 ** 2 + fU12 ** 2)) + (
                    ((U02 * U11) ** 2) * (fU02 ** 2 + fU11 ** 2))) / ((
                                                                              U01 * U12 + U02 * U11) ** 2)) + fsig12 ** 2)) +
            (((U00 * U10 * sig00) ** 2) * (fU00 ** 2 + fU10 ** 2 + fsig00 ** 2)) +
            (((U01 * U11 * sig11) ** 2) * (fU01 ** 2 + fU11 ** 2 + fsig11 ** 2)) +
            (((U02 * U12 * sig22) ** 2) * (fU02 ** 2 + fU12 ** 2 + fsig22 ** 2))
        )

        sig_lab_error[0, 2] = np.sqrt(
            ((sig01 ** 2) * ((U00 * U21 + U01 * U20) ** 2) * ((((((U00 * U21) ** 2) * (fU00 ** 2 + fU21 ** 2)) + (
                    ((U01 * U20) ** 2) * (fU01 ** 2 + fU20 ** 2))) / ((
                                                                              U00 * U21 + U01 * U20) ** 2)) + fsig01 ** 2)) +
            ((sig02 ** 2) * ((U00 * U22 + U02 * U20) ** 2) * ((((((U00 * U22) ** 2) * (fU00 ** 2 + fU22 ** 2)) + (
                    ((U02 * U20) ** 2) * (fU02 ** 2 + fU20 ** 2))) / ((
                                                                              U00 * U22 + U02 * U20) ** 2)) + fsig02 ** 2)) +
            ((sig12 ** 2) * ((U01 * U22 + U02 * U21) ** 2) * ((((((U01 * U22) ** 2) * (fU01 ** 2 + fU22 ** 2)) + (
                    ((U02 * U21) ** 2) * (fU02 ** 2 + fU21 ** 2))) / ((
                                                                              U01 * U22 + U02 * U21) ** 2)) + fsig12 ** 2)) +
            (((U00 * U20 * sig00) ** 2) * (fU00 ** 2 + fU20 ** 2 + fsig00 ** 2)) +
            (((U01 * U21 * sig11) ** 2) * (fU01 ** 2 + fU21 ** 2 + fsig11 ** 2)) +
            (((U02 * U22 * sig22) ** 2) * (fU02 ** 2 + fU22 ** 2 + fsig22 ** 2))
        )

        sig_lab_error[1, 0] = sig_lab_error[0, 1]

        sig_lab_error[1, 2] = np.sqrt(
            ((sig01 ** 2) * ((U10 * U21 + U11 * U20) ** 2) * ((((((U10 * U21) ** 2) * (fU10 ** 2 + fU21 ** 2)) + (
                    ((U11 * U20) ** 2) * (fU11 ** 2 + fU20 ** 2))) / ((
                                                                              U10 * U21 + U11 * U20) ** 2)) + fsig01 ** 2)) +
            ((sig02 ** 2) * ((U10 * U22 + U12 * U20) ** 2) * ((((((U10 * U22) ** 2) * (fU10 ** 2 + fU22 ** 2)) + (
                    ((U12 * U20) ** 2) * (fU12 ** 2 + fU20 ** 2))) / ((
                                                                              U10 * U22 + U12 * U20) ** 2)) + fsig02 ** 2)) +
            ((sig12 ** 2) * ((U11 * U22 + U12 * U21) ** 2) * ((((((U11 * U22) ** 2) * (fU11 ** 2 + fU22 ** 2)) + (
                    ((U12 * U21) ** 2) * (fU12 ** 2 + fU21 ** 2))) / ((
                                                                              U11 * U22 + U12 * U21) ** 2)) + fsig12 ** 2)) +
            (((U10 * U20 * sig00) ** 2) * (fU10 ** 2 + fU20 ** 2 + fsig00 ** 2)) +
            (((U11 * U21 * sig11) ** 2) * (fU11 ** 2 + fU21 ** 2 + fsig11 ** 2)) +
            (((U12 * U22 * sig22) ** 2) * (fU12 ** 2 + fU22 ** 2 + fsig22 ** 2))
        )

        sig_lab_error[2, 0] = sig_lab_error[0, 2]

        sig_lab_error[2, 1] = sig_lab_error[1, 2]

        calculated_sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                              U_error=self.valid_U_error,
                                                              sig=self.valid_sig,
                                                              sig_error=self.valid_sig_error)

        print('\n')
        print(self.valid_U)
        print(self.valid_sig_error)
        print(sig_lab_error)
        print(calculated_sig_lab_error)

        self.assertTrue(np.allclose(sig_lab_error, calculated_sig_lab_error))

    def test_wrong_U_matrix_type(self):
        invalid_U_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=invalid_U_type,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_U_matrix_shape(self):
        invalid_U_shape = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=invalid_U_shape,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_U_matrix_dtype(self):
        invalid_U_dtype = self.valid_U.astype(str)
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=invalid_U_dtype,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_U_error_matrix_type(self):
        invalid_U_error_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=invalid_U_error_type,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_U_error_matrix_shape(self):
        invalid_U_error_shape = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=invalid_U_error_shape,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_U_error_matrix_dtype(self):
        invalid_U_error_dtype = self.valid_U_error.astype(str)
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=invalid_U_error_dtype,
                                                       sig=self.valid_sig,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_sig_matrix_type(self):
        invalid_sig_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=invalid_sig_matrix_type,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_sig_matrix_shape(self):
        invalid_sig_matrix_shape = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=invalid_sig_matrix_shape,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_sig_matrix_dtype(self):
        invalid_sig_matrix_dtype = self.valid_sig.astype(str)
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=invalid_sig_matrix_dtype,
                                                       sig_error=self.valid_sig_error)

    def test_non_symmetric_sig_matrix(self):
        non_symmetric_sig_matrix = self.valid_sig
        non_symmetric_sig_matrix[0, 1] = 1e-4
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=non_symmetric_sig_matrix,
                                                       sig_error=self.valid_sig_error)

    def test_wrong_sig_error_matrix_type(self):
        invalid_sig_error_matrix_type = [1., 2, 3, 4, 5, 6]
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=invalid_sig_error_matrix_type)

    def test_wrong_sig_error_matrix_shape(self):
        invalid_sig_error_matrix_shape = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=invalid_sig_error_matrix_shape)

    def test_wrong_sig_error_matrix_dtype(self):
        invalid_sig_error_matrix_dtype = self.valid_sig.astype(str)
        with self.assertRaises(TypeError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=invalid_sig_error_matrix_dtype)

    def test_non_symmetric_sig_error_matrix(self):
        non_symmetric_sig_error_matrix = self.valid_sig_error
        non_symmetric_sig_error_matrix[0, 1] = 1e-4
        with self.assertRaises(ValueError):
            sig_lab_error = sig_error_to_sig_lab_error(U=self.valid_U,
                                                       U_error=self.valid_U_error,
                                                       sig=self.valid_sig,
                                                       sig_error=non_symmetric_sig_error_matrix)

# TODO: Finish These!
