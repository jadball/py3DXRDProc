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

import unittest

import numpy as np
from py3DXRDProc.grain import BaseGrain, RawGrain, VirtualGrain, \
    validate_grains_list, filter_grain_list, merge_grains, find_multiple_observations, \
    find_all_grain_pair_matches_from_list, \
    combine_matching_grain_pairs_into_groups, inclination_angle, BaseMapGrain
from py3DXRDProc.grain_map import RawGrainsMap, BaseGrainsMap
from py3DXRDProc.grain_volume import GrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample
from pymicro.crystal.lattice import Symmetry, Lattice
from scipy.spatial.transform import Rotation
from xfab import tools


class TestValidateGrainsList(unittest.TestCase):
    def setUp(self):
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        self.valid_grain_1 = BaseGrain(pos=np.array([1., 2, 3]),
                                       UBI=raw_ubi_1,
                                       volume=250.0)

        self.valid_grain_2 = BaseGrain(pos=np.array([1, 2, 3.1]),
                                       UBI=raw_ubi_2,
                                       volume=750.0)

        # 3.075

    def test_valid_list(self):
        self.assertTrue(validate_grains_list([self.valid_grain_1, self.valid_grain_2]))

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            validate_grains_list((self.valid_grain_1, self.valid_grain_2))

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            validate_grains_list([])

    def test_mixed_type_list(self):
        with self.assertRaises(TypeError):
            validate_grains_list([self.valid_grain_1, "valid_grain_2"])


class TestFilterGrainList(unittest.TestCase):
    def setUp(self):
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        self.valid_grain_1 = BaseGrain(pos=np.array([1., 2, 3]),
                                       UBI=raw_ubi_1,
                                       volume=250.0)

        self.valid_grain_2 = BaseGrain(pos=np.array([1, 2, 3.1]),
                                       UBI=raw_ubi_2,
                                       volume=750.0)

    def test_valid_regular_pos(self):
        self.assertSequenceEqual(
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], 0, 2, 0, 3, 0, 3, use_adjusted_pos=False),
            [self.valid_grain_1])

    def test_wrong_bound_type(self):
        with self.assertRaises(TypeError):
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], "0", 2, 0, 3, 0, 3, use_adjusted_pos=False)

    def test_x_max_less_than_min(self):
        with self.assertRaises(ValueError):
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], 2, 0, 0, 3, 0, 3, use_adjusted_pos=False)

    def test_y_max_less_than_min(self):
        with self.assertRaises(ValueError):
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], 0, 2, 3, 0, 0, 3, use_adjusted_pos=False)

    def test_z_max_less_than_min(self):
        with self.assertRaises(ValueError):
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], 0, 2, 0, 3, 3, 0, use_adjusted_pos=False)

    def test_wrong_adjusted_pos(self):
        with self.assertRaises(TypeError):
            filter_grain_list([self.valid_grain_1, self.valid_grain_2], 0, 2, 0, 3, 0, 3, use_adjusted_pos=True)

    def test_valid_adjusted_pos(self):
        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        sample = Sample(name="test_sample_name")
        # Make a blank load step
        load_step = LoadStep(name="test_load_step_name", sample=sample)
        # Make a blank grain volume
        grain_volume = GrainVolume(name="test_grain_volume", load_step=load_step,
                                   index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                   material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                   offset_origin=np.array([0., 0., 0.]))

        raw_grain_map = RawGrainsMap(grain_volume=grain_volume, phase=self.phase)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=250.0,
                               grain_map=raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=250.0,
                               grain_map=raw_grain_map,
                               mean_peak_intensity=1.0)

        self.assertSequenceEqual(filter_grain_list([raw_grain_1, raw_grain_2], 0, 2, 0, 3, 0, 3, use_adjusted_pos=True),
                                 [raw_grain_1])


class TestMergeGrains(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume", load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),  # 0.8 mm cubed
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain_volume.add_raw_map(self.raw_grain_map)

        self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(self.raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)

        self.valid_grain_1 = RawGrain(gid=1,
                                      pos=np.array([1., 2, 3]),
                                      UBI=raw_ubi_1,
                                      volume=0.2,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        self.valid_grain_2 = RawGrain(gid=2,
                                      pos=np.array([1, 2, 3.1]),
                                      UBI=raw_ubi_2,
                                      volume=0.6,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=3.0)

        # 3.075

        self.raw_grain_map.add_grains([self.valid_grain_1, self.valid_grain_2])

    def test_valid_merge(self):
        # check scaled volume

        merged_grain = merge_grains(self.raw_grain_map.grains)
        self.assertTrue(np.allclose(merged_grain.pos, np.array([1, 2, 3.075])))
        self.assertTrue(np.allclose(merged_grain.U, self.raw_orientation_2))
        self.assertEqual(merged_grain.volume, self.valid_grain_1.volume + self.valid_grain_2.volume)
        self.assertTrue(np.allclose(merged_grain.pos_offset, np.array([1, 2, 3.075])))
        self.assertTrue(isinstance(merged_grain, VirtualGrain))

    def test_single_grain_merge(self):
        merged_grain = merge_grains([self.valid_grain_1])
        self.assertTrue(np.allclose(merged_grain.pos, self.valid_grain_1.pos))
        self.assertTrue(np.allclose(merged_grain.UBI, self.valid_grain_1.UBI))
        self.assertTrue(np.allclose(merged_grain.volume, self.valid_grain_1.volume))
        self.assertTrue(np.allclose(merged_grain.pos_offset, self.valid_grain_1.pos_offset))


# class TestAreGrainsSimilar(unittest.TestCase):
#     def setUp(self):
#         # Make a blank phase
#         self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
#                            symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
#
#         self.raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         self.raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(self.raw_orientation_1)).as_matrix()
#         self.raw_orientation_3 = (Rotation.from_matrix(self.raw_orientation_1) * Rotation.from_euler('x', 1, degrees=True)).as_matrix()
#         self.raw_orientation_4 = Rotation.from_euler('x', 1, degrees=True).as_matrix().T @ self.raw_orientation_1
#         self.raw_orientation_5 = self.raw_orientation_1 @ Rotation.from_euler('x', 1, degrees=True).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
#         raw_ubi_4 = tools.u_to_ubi(self.raw_orientation_4, self.phase.reference_unit_cell)
#         raw_ubi_5 = tools.u_to_ubi(self.raw_orientation_5, self.phase.reference_unit_cell)
#         self.valid_grain_1 = VirtualGrain(pos=np.array([1., 2, 3]),
#                                           UBI=raw_ubi_1,
#                                           volume=250.0,
#                                           pos_offset=np.array([1., 2, 3]),
#                                           phase=self.phase)
#
#         self.valid_grain_2 = VirtualGrain(pos=np.array([1, 2, 3.1]),
#                                           UBI=raw_ubi_2,
#                                           volume=750.0,
#                                           pos_offset=np.array([1., 2, 3.1]),
#                                           phase=self.phase)
#         self.valid_grain_3 = VirtualGrain(pos=np.array([1, 2, 3.1]),
#                                           UBI=raw_ubi_3,
#                                           volume=750.0,
#                                           pos_offset=np.array([1., 2, 3.1]),
#                                           phase=self.phase)
#         self.valid_grain_4 = VirtualGrain(pos=np.array([1, 2, 3.1]),
#                                           UBI=raw_ubi_4,
#                                           volume=750.0,
#                                           pos_offset=np.array([1., 2, 3.1]),
#                                           phase=self.phase)
#         self.valid_grain_5 = VirtualGrain(pos=np.array([1, 2, 3.1]),
#                                           UBI=raw_ubi_5,
#                                           volume=750.0,
#                                           pos_offset=np.array([1., 2, 3.1]),
#                                           phase=self.phase)
#
#     def test_similar_pair(self):
#         self.assertTrue(are_grains_similar((self.valid_grain_1, self.valid_grain_2), dist_tol=0.15, angle_tol=2.0))
#         self.assertTrue(are_grains_similar((self.valid_grain_2, self.valid_grain_1), dist_tol=0.15, angle_tol=2.0))
#         self.assertTrue(are_grains_similar((self.valid_grain_1, self.valid_grain_3), dist_tol=0.15, angle_tol=2.0))
#         self.assertTrue(are_grains_similar((self.valid_grain_1, self.valid_grain_4), dist_tol=0.15, angle_tol=2.0))
#         self.assertTrue(are_grains_similar((self.valid_grain_1, self.valid_grain_5), dist_tol=0.15, angle_tol=2.0))
#
#     def test_too_far_apart(self):
#         self.assertFalse(are_grains_similar((self.valid_grain_1, self.valid_grain_2), dist_tol=0.09, angle_tol=2.0))
#
#     def test_angle_too_far(self):
#         self.assertFalse(are_grains_similar((self.valid_grain_1, self.valid_grain_2), dist_tol=0.15, angle_tol=0.5))
#
#     def test_wrong_dist_tol_type(self):
#         with self.assertRaises(TypeError):
#             are_grains_similar((self.valid_grain_1, self.valid_grain_2), dist_tol="0.15", angle_tol=2.0)
#
#     def test_wrong_angle_tol_type(self):
#         with self.assertRaises(TypeError):
#             are_grains_similar((self.valid_grain_1, self.valid_grain_2), dist_tol=0.15, angle_tol="2.0")
#
#     def test_wrong_grain_type(self):
#         with self.assertRaises(TypeError):
#             are_grains_similar((self.valid_grain_1, "grain_2"), dist_tol=0.15, angle_tol=2.0)
#
#         raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
#         base_grain_1 = BaseGrain(pos=self.valid_grain_1.pos + np.array([0., 0., 0.1]), UBI=raw_ubi_1, volume=100.0)
#
#         with self.assertRaises(TypeError):
#             are_grains_similar((self.valid_grain_1, base_grain_1), dist_tol=0.15, angle_tol=2.0)
#
#     def test_same_grain_twice(self):
#         self.assertFalse(are_grains_similar((self.valid_grain_1, self.valid_grain_1), dist_tol=0.09, angle_tol=2.0))


class TestFindMatchingGrainPairs(unittest.TestCase):
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
        self.raw_orientation_2 = Rotation.from_euler('x', 1, degrees=True).as_matrix() @ self.raw_orientation_1
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)

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

        # 3.075

        self.raw_grain_map.add_grains([self.valid_grain_1, self.valid_grain_2, self.valid_grain_3])

    def test_valid_grains_list(self):
        matching_grain_pairs = find_all_grain_pair_matches_from_list(self.raw_grain_map.grains, dist_tol=0.15,
                                                                     angle_tol=2.0)

        self.assertSequenceEqual(matching_grain_pairs, [(self.valid_grain_1, self.valid_grain_2)])


class TestGroupMatchingGrains(unittest.TestCase):
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
        self.raw_orientation_2 = Rotation.from_euler('x', 1, degrees=True).as_matrix() @ self.raw_orientation_1
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_4 = Rotation.from_euler('x', -1, degrees=True).as_matrix() @ self.raw_orientation_1

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_4 = tools.u_to_ubi(self.raw_orientation_4, self.phase.reference_unit_cell)

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
                                      pos=np.array([1.1, 2., 3.]),
                                      UBI=raw_ubi_4,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains([self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4])

    def test_valid_grains_list(self):
        matching_grain_pairs = find_all_grain_pair_matches_from_list(self.raw_grain_map.grains, dist_tol=0.15,
                                                                     angle_tol=2.0)
        isolated_groups = combine_matching_grain_pairs_into_groups(self.raw_grain_map.grains, matching_grain_pairs)

        self.assertSequenceEqual(isolated_groups,
                                 [[self.valid_grain_1, self.valid_grain_2, self.valid_grain_4], [self.valid_grain_3]])


class TestFindMultipleObservations(unittest.TestCase):
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
        self.raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(self.raw_orientation_1)).as_matrix()
        self.raw_orientation_3 = Rotation.random(1).as_matrix()[0]
        self.raw_orientation_4 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(self.raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(self.raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(self.raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(self.raw_orientation_3, self.phase.reference_unit_cell)
        raw_ubi_4 = tools.u_to_ubi(self.raw_orientation_4, self.phase.reference_unit_cell)

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
                                      pos=np.array([1.1, 2., 3.]),
                                      UBI=raw_ubi_4,
                                      volume=500.0,
                                      grain_map=self.raw_grain_map,
                                      mean_peak_intensity=1.0)

        # 3.075

        self.raw_grain_map.add_grains([self.valid_grain_1, self.valid_grain_2, self.valid_grain_3, self.valid_grain_4])

    def test_valid_grains_list(self):
        self.assertSequenceEqual(find_multiple_observations(self.raw_grain_map.grains, dist_tol=0.15, angle_tol=2.0),
                                 [[self.valid_grain_1, self.valid_grain_2, self.valid_grain_4], [self.valid_grain_3]])


class TestInclinationAngle(unittest.TestCase):
    def test_valid_basegrain_pair_vertical(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        grain_b = BaseGrain(pos=np.array([0., 0., 1.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 0.0)

    def test_valid_basegrain_pair_horizontal(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        grain_b = BaseGrain(pos=np.array([0., 1., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 90.0)

    def test_valid_basegrain_pair_diagonal(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        grain_b = BaseGrain(pos=np.array([0., 1., 1.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 45.0)

    def test_valid_basemapgrain_pair_vertical(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume_a = GrainVolume(name="test_grain_volume_a",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 0., 0.]))

        # Make this GrainVolume above the others
        self.grain_volume_b = GrainVolume(name="test_grain_volume_b",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 0., 1.]))

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map_a = BaseGrainsMap(grain_volume=self.grain_volume_a, phase=self.phase)
        self.grain_map_b = BaseGrainsMap(grain_volume=self.grain_volume_b, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_a)

        grain_b = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_b)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 0.0)

    def test_valid_basemapgrain_pair_horizontal(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume_a = GrainVolume(name="test_grain_volume_a",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 0., 0.]))

        # Make this GrainVolume above the others
        self.grain_volume_b = GrainVolume(name="test_grain_volume_b",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 1., 0.]))

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map_a = BaseGrainsMap(grain_volume=self.grain_volume_a, phase=self.phase)
        self.grain_map_b = BaseGrainsMap(grain_volume=self.grain_volume_b, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_a)

        grain_b = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_b)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 90.0)

    def test_valid_basemapgrain_pair_diagonal(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume_a = GrainVolume(name="test_grain_volume_a",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 0., 0.]))

        # Make this GrainVolume above the others
        self.grain_volume_b = GrainVolume(name="test_grain_volume_b",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0., 1., 1.]))

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map_a = BaseGrainsMap(grain_volume=self.grain_volume_a, phase=self.phase)
        self.grain_map_b = BaseGrainsMap(grain_volume=self.grain_volume_b, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_a)

        grain_b = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map_b)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 45.0)

    def test_valid_basemapgrain_pair_horizontal_same_vol(self):
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

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map)

        grain_b = BaseMapGrain(pos=np.array([0., 1., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=2,
                               grain_map=self.grain_map)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 90.0)

    def test_valid_basemapgrain_pair_vertical_same_vol(self):
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

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map)

        grain_b = BaseMapGrain(pos=np.array([0., 0., 1.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=2,
                               grain_map=self.grain_map)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 0.0)

    def test_valid_basemapgrain_pair_diagonal_same_vol(self):
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

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        grain_a = BaseMapGrain(pos=np.array([0., 0., 0.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=1,
                               grain_map=self.grain_map)

        grain_b = BaseMapGrain(pos=np.array([0., 1., 1.]),
                               UBI=np.array([[3., 0, 0],
                                             [0., 3, 0],
                                             [0., 0, 3]]),
                               volume=100.0,
                               gid=2,
                               grain_map=self.grain_map)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 45.0)

    def test_invalid_types(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        with self.assertRaises(TypeError):
            inclination_angle(grain_a, "grain_b")

    def test_b_below_a(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)
        grain_b = BaseGrain(pos=np.array([0., 0., -1.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 0.0)

    def test_b_behind_a(self):
        grain_a = BaseGrain(pos=np.array([0., 0., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)
        grain_b = BaseGrain(pos=np.array([0., -1., 0.]),
                            UBI=np.array([[3., 0, 0],
                                          [0., 3, 0],
                                          [0., 0, 3]]),
                            volume=100.0)

        self.assertAlmostEqual(inclination_angle(grain_a, grain_b), 90.0)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
