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
from py3DXRDProc.grain import CleanGrain, BaseGrain, TrackedGrain
from py3DXRDProc.grain_map import CleanedGrainsMap, RawGrainsMap, RawGrain, StitchedGrainsMap, TrackedGrainsMap
from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume, TrackedGrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample
from pymicro.crystal.lattice import Symmetry, Lattice
from scipy.spatial.transform import Rotation
from xfab import tools


class TestWrongInitTypes(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step_1 = LoadStep(name="test_load_step_name_1", sample=self.sample)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1_step1 = GrainVolume(name="test_grain_volume_1",
                                                load_step=self.load_step_1,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2_step1 = GrainVolume(name="test_grain_volume_2",
                                                load_step=self.load_step_1,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1_step1 = RawGrainsMap(grain_volume=self.grain_volume_1_step1, phase=self.phase)
        self.raw_grain_map_2_step1 = RawGrainsMap(grain_volume=self.grain_volume_2_step1, phase=self.phase)

        self.grain_volume_1_step1.add_raw_map(self.raw_grain_map_1_step1)
        self.grain_volume_2_step1.add_raw_map(self.raw_grain_map_2_step1)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1_step1 = RawGrain(gid=1,
                                                pos=np.array([1., 2, 3.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_1_step1 = RawGrain(gid=2,
                                                pos=np.array([1, 2, 3.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_1_step1 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 4.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_1_map_2_step1 = RawGrain(gid=1,
                                                pos=np.array([1.1, 2, 2.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_2_step1 = RawGrain(gid=2,
                                                pos=np.array([1.1, 2, 2.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_2_step1 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 3.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_map_1_step1.add_grains(
            [self.raw_grain_1_map_1_step1, self.raw_grain_2_map_1_step1, self.raw_grain_3_map_1_step1])

        self.raw_grain_map_2_step1.add_grains(
            [self.raw_grain_1_map_2_step1, self.raw_grain_2_map_2_step1, self.raw_grain_3_map_2_step1])

        self.clean_grain_map_1_step1 = CleanedGrainsMap(raw_map=self.raw_grain_map_1_step1)

        self.clean_grain_map_2_step1 = CleanedGrainsMap(raw_map=self.raw_grain_map_2_step1)

        self.clean_grain_1_map_1_step1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1_step1,
                                                                         self.raw_grain_2_map_1_step1],
                                                                     grain_map=self.clean_grain_map_1_step1)

        self.clean_grain_2_map_1_step1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1_step1],
                                                                     grain_map=self.clean_grain_map_1_step1)

        self.clean_grain_map_1_step1.add_grains([self.clean_grain_1_map_1_step1, self.clean_grain_2_map_1_step1])

        self.clean_grain_1_map_2_step1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2_step1,
                                                                         self.raw_grain_2_map_2_step1],
                                                                     grain_map=self.clean_grain_map_2_step1)

        self.clean_grain_2_map_2_step1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2_step1],
                                                                     grain_map=self.clean_grain_map_2_step1)

        self.clean_grain_map_2_step1.add_grains([self.clean_grain_1_map_2_step1, self.clean_grain_2_map_2_step1])

        self.stitched_volume_step1 = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                         material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                         offset_origin=np.array([0., 0., 0.]),
                                                         contrib_vols_list=[self.grain_volume_1_step1,
                                                                            self.grain_volume_2_step1])

        self.stitched_map_step1 = StitchedGrainsMap.from_clean_maps_list(clean_maps_list=[self.clean_grain_map_1_step1,
                                                                                          self.clean_grain_map_2_step1],
                                                                         merged_volume=self.stitched_volume_step1,
                                                                         filter_before_merge=False,
                                                                         dist_tol_xy=0.15,
                                                                         dist_tol_z=0.15,
                                                                         angle_tol=2.0)

        ### LOAD STEP 2

        # Make a blank load step
        self.load_step_2 = LoadStep(name="test_load_step_name_2", sample=self.sample)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1_step2 = GrainVolume(name="test_grain_volume_1",
                                                load_step=self.load_step_2,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2_step2 = GrainVolume(name="test_grain_volume_2",
                                                load_step=self.load_step_2,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1_step2 = RawGrainsMap(grain_volume=self.grain_volume_1_step2, phase=self.phase)
        self.raw_grain_map_2_step2 = RawGrainsMap(grain_volume=self.grain_volume_2_step2, phase=self.phase)

        self.grain_volume_1_step2.add_raw_map(self.raw_grain_map_1_step2)
        self.grain_volume_2_step2.add_raw_map(self.raw_grain_map_2_step2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1_step2 = RawGrain(gid=1,
                                                pos=np.array([1., 2, 3.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_1_step2 = RawGrain(gid=2,
                                                pos=np.array([1, 2, 3.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_1_step2 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 4.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_1_map_2_step2 = RawGrain(gid=1,
                                                pos=np.array([1.1, 2, 2.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_2_step2 = RawGrain(gid=2,
                                                pos=np.array([1.1, 2, 2.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_2_step2 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 3.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_map_1_step2.add_grains(
            [self.raw_grain_1_map_1_step2, self.raw_grain_2_map_1_step2, self.raw_grain_3_map_1_step2])

        self.raw_grain_map_2_step2.add_grains(
            [self.raw_grain_1_map_2_step2, self.raw_grain_2_map_2_step2, self.raw_grain_3_map_2_step2])

        self.clean_grain_map_1_step2 = CleanedGrainsMap(raw_map=self.raw_grain_map_1_step2)

        self.clean_grain_map_2_step2 = CleanedGrainsMap(raw_map=self.raw_grain_map_2_step2)

        self.clean_grain_1_map_1_step2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1_step2,
                                                                         self.raw_grain_2_map_1_step2],
                                                                     grain_map=self.clean_grain_map_1_step2)

        self.clean_grain_2_map_1_step2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1_step2],
                                                                     grain_map=self.clean_grain_map_1_step2)

        self.clean_grain_map_1_step2.add_grains([self.clean_grain_1_map_1_step2, self.clean_grain_2_map_1_step2])

        self.clean_grain_1_map_2_step2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2_step2,
                                                                         self.raw_grain_2_map_2_step2],
                                                                     grain_map=self.clean_grain_map_2_step2)

        self.clean_grain_2_map_2_step2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2_step2],
                                                                     grain_map=self.clean_grain_map_2_step2)

        self.clean_grain_map_2_step2.add_grains([self.clean_grain_1_map_2_step2, self.clean_grain_2_map_2_step2])

        self.stitched_volume_step2 = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                         material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                         offset_origin=np.array([0., 0., 0.]),
                                                         contrib_vols_list=[self.grain_volume_1_step2,
                                                                            self.grain_volume_2_step2])

        self.stitched_map_step2 = StitchedGrainsMap.from_clean_maps_list(clean_maps_list=[self.clean_grain_map_1_step2,
                                                                                          self.clean_grain_map_2_step2],
                                                                         merged_volume=self.stitched_volume_step2,
                                                                         filter_before_merge=False,
                                                                         dist_tol_xy=0.15,
                                                                         dist_tol_z=0.15,
                                                                         angle_tol=2.0)

        self.tracked_volume = TrackedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                 material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                 offset_origin=np.array([0., 0., 0.]),
                                                 contrib_vols_list=[self.stitched_volume_step1,
                                                                    self.stitched_volume_step2])

        self.tracked_map = TrackedGrainsMap(grain_volume=self.tracked_volume,
                                            stitch_maps_list=[self.stitched_map_step1, self.stitched_map_step2])

    def test_wrong_grain_map_type(self):
        with self.assertRaises(TypeError):
            tracked_grain = TrackedGrain(gid=1,
                                         pos_offset=np.array([1., 2., 3.]),
                                         pos_sample=np.array([1., 2., 3.]),
                                         UBI=self.raw_grain_1_map_1_step1.UBI,
                                         volume=1000.0,
                                         grain_map="self.tracked_map",
                                         parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                               self.stitched_map_step2.grains[0]])

    def test_wrong_pos_type(self):
        with self.assertRaises(TypeError):
            tracked_grain = TrackedGrain(gid=1,
                                         pos_offset=[1., 2., 3.],
                                         pos_sample=np.array([1., 2., 3.]),
                                         UBI=self.raw_grain_1_map_1_step1.UBI,
                                         volume=1000.0,
                                         grain_map=self.tracked_map,
                                         parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                               self.stitched_map_step2.grains[0]])

    def test_wrong_pos_shape(self):
        with self.assertRaises(ValueError):
            tracked_grain = TrackedGrain(gid=1,
                                         pos_offset=np.array([1., 2., 3., 4.]),
                                         pos_sample=np.array([1, 2, 3]),
                                         UBI=self.raw_grain_1_map_1_step1.UBI,
                                         volume=1000.0,
                                         grain_map=self.tracked_map,
                                         parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                               self.stitched_map_step2.grains[0]])

    def test_wrong_pos_element_type(self):
        with self.assertRaises(TypeError):
            tracked_grain = TrackedGrain(gid=1,
                                         pos_offset=np.array([1, 2, 3]),
                                         pos_sample=np.array([1., 2., 3.]),
                                         UBI=self.raw_grain_1_map_1_step1.UBI,
                                         volume=1000.0,
                                         grain_map=self.tracked_map,
                                         parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                               self.stitched_map_step2.grains[0]])

    def test_wrong_parent_grains_type(self):
        with self.assertRaises(TypeError):
            tracked_grain = TrackedGrain(gid=1,
                                         pos_offset=np.array([1., 2., 3.]),
                                         pos_sample=np.array([1., 2., 3.]),
                                         UBI=self.raw_grain_1_map_1_step1.UBI,
                                         volume=1000.0,
                                         grain_map=self.tracked_map,
                                         parent_stitch_grains=(
                                             self.stitched_map_step1.grains[0], self.stitched_map_step2.grains[0]))


class TestParentGrains(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step_1 = LoadStep(name="test_load_step_name_1", sample=self.sample)
        # Add the load_step to the Sample
        self.sample.add_load_step(self.load_step_1)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1_step1 = GrainVolume(name="test_grain_volume_1",
                                                load_step=self.load_step_1,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2_step1 = GrainVolume(name="test_grain_volume_2",
                                                load_step=self.load_step_1,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1_step1 = RawGrainsMap(grain_volume=self.grain_volume_1_step1, phase=self.phase)
        self.raw_grain_map_2_step1 = RawGrainsMap(grain_volume=self.grain_volume_2_step1, phase=self.phase)

        self.grain_volume_1_step1.add_raw_map(self.raw_grain_map_1_step1)
        self.grain_volume_2_step1.add_raw_map(self.raw_grain_map_2_step1)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1_step1 = RawGrain(gid=1,
                                                pos=np.array([1., 2, 3.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_1_step1 = RawGrain(gid=2,
                                                pos=np.array([1, 2, 3.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_1_step1 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 4.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_1_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_1_map_2_step1 = RawGrain(gid=1,
                                                pos=np.array([1.1, 2, 2.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_2_step1 = RawGrain(gid=2,
                                                pos=np.array([1.1, 2, 2.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_2_step1 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 3.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_2_step1,
                                                mean_peak_intensity=1.0)

        self.raw_grain_map_1_step1.add_grains(
            [self.raw_grain_1_map_1_step1, self.raw_grain_2_map_1_step1, self.raw_grain_3_map_1_step1])

        self.raw_grain_map_2_step1.add_grains(
            [self.raw_grain_1_map_2_step1, self.raw_grain_2_map_2_step1, self.raw_grain_3_map_2_step1])

        self.clean_grain_map_1_step1 = CleanedGrainsMap(raw_map=self.raw_grain_map_1_step1)

        self.clean_grain_map_2_step1 = CleanedGrainsMap(raw_map=self.raw_grain_map_2_step1)

        self.clean_grain_1_map_1_step1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1_step1,
                                                                         self.raw_grain_2_map_1_step1],
                                                                     grain_map=self.clean_grain_map_1_step1)

        self.clean_grain_2_map_1_step1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1_step1],
                                                                     grain_map=self.clean_grain_map_1_step1)

        self.clean_grain_map_1_step1.add_grains([self.clean_grain_1_map_1_step1, self.clean_grain_2_map_1_step1])

        self.clean_grain_1_map_2_step1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2_step1,
                                                                         self.raw_grain_2_map_2_step1],
                                                                     grain_map=self.clean_grain_map_2_step1)

        self.clean_grain_2_map_2_step1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2_step1],
                                                                     grain_map=self.clean_grain_map_2_step1)

        self.clean_grain_map_2_step1.add_grains([self.clean_grain_1_map_2_step1, self.clean_grain_2_map_2_step1])

        self.stitched_volume_step1 = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                         material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                         offset_origin=np.array([0., 0., 0.]),
                                                         contrib_vols_list=[self.grain_volume_1_step1,
                                                                            self.grain_volume_2_step1])

        self.stitched_map_step1 = StitchedGrainsMap.from_clean_maps_list(
            clean_maps_list=[self.clean_grain_map_1_step1, self.clean_grain_map_2_step1],
            merged_volume=self.stitched_volume_step1,
            filter_before_merge=False,
            filter_bounds=None,
            dist_tol_xy=0.2,
            dist_tol_z=0.2,
            angle_tol=2.0)

        ### LOAD STEP 2

        # Make a blank load step
        self.load_step_2 = LoadStep(name="test_load_step_name_2", sample=self.sample)
        # Add the load_step to the Sample
        self.sample.add_load_step(self.load_step_2)

        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1_step2 = GrainVolume(name="test_grain_volume_1",
                                                load_step=self.load_step_2,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2_step2 = GrainVolume(name="test_grain_volume_2",
                                                load_step=self.load_step_2,
                                                index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1_step2 = RawGrainsMap(grain_volume=self.grain_volume_1_step2, phase=self.phase)
        self.raw_grain_map_2_step2 = RawGrainsMap(grain_volume=self.grain_volume_2_step2, phase=self.phase)

        self.grain_volume_1_step2.add_raw_map(self.raw_grain_map_1_step2)
        self.grain_volume_2_step2.add_raw_map(self.raw_grain_map_2_step2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1_step2 = RawGrain(gid=1,
                                                pos=np.array([1., 2, 3.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_1_step2 = RawGrain(gid=2,
                                                pos=np.array([1, 2, 3.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_1_step2 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 4.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_1_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_1_map_2_step2 = RawGrain(gid=1,
                                                pos=np.array([1.1, 2, 2.5]),
                                                UBI=raw_ubi_1,
                                                volume=250.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_2_map_2_step2 = RawGrain(gid=2,
                                                pos=np.array([1.1, 2, 2.6]),
                                                UBI=raw_ubi_2,
                                                volume=750.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_3_map_2_step2 = RawGrain(gid=3,
                                                pos=np.array([-1, -2, 3.5]),
                                                UBI=raw_ubi_3,
                                                volume=100.0,
                                                grain_map=self.raw_grain_map_2_step2,
                                                mean_peak_intensity=1.0)

        self.raw_grain_map_1_step2.add_grains(
            [self.raw_grain_1_map_1_step2, self.raw_grain_2_map_1_step2, self.raw_grain_3_map_1_step2])

        self.raw_grain_map_2_step2.add_grains(
            [self.raw_grain_1_map_2_step2, self.raw_grain_2_map_2_step2, self.raw_grain_3_map_2_step2])

        self.clean_grain_map_1_step2 = CleanedGrainsMap(raw_map=self.raw_grain_map_1_step2)

        self.clean_grain_map_2_step2 = CleanedGrainsMap(raw_map=self.raw_grain_map_2_step2)

        self.clean_grain_1_map_1_step2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1_step2,
                                                                         self.raw_grain_2_map_1_step2],
                                                                     grain_map=self.clean_grain_map_1_step2)

        self.clean_grain_2_map_1_step2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1_step2],
                                                                     grain_map=self.clean_grain_map_1_step2)

        self.clean_grain_map_1_step2.add_grains([self.clean_grain_1_map_1_step2, self.clean_grain_2_map_1_step2])

        self.clean_grain_1_map_2_step2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2_step2,
                                                                         self.raw_grain_2_map_2_step2],
                                                                     grain_map=self.clean_grain_map_2_step2)

        self.clean_grain_2_map_2_step2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2_step2],
                                                                     grain_map=self.clean_grain_map_2_step2)

        self.clean_grain_map_2_step2.add_grains([self.clean_grain_1_map_2_step2, self.clean_grain_2_map_2_step2])

        self.stitched_volume_step2 = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                         material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                         offset_origin=np.array([0., 0., 0.]),
                                                         contrib_vols_list=[self.grain_volume_1_step2,
                                                                            self.grain_volume_2_step2])

        # self.stitched_map_step2 = StitchedGrainsMap(grain_volume=self.stitched_volume_step2,
        #                                             clean_maps_list=[self.clean_grain_map_1_step2,
        #                                                              self.clean_grain_map_2_step2])

        self.stitched_map_step2 = StitchedGrainsMap.from_clean_maps_list(
            clean_maps_list=[self.clean_grain_map_1_step2, self.clean_grain_map_2_step2],
            merged_volume=self.stitched_volume_step2,
            filter_before_merge=False,
            filter_bounds=None,
            dist_tol_xy=0.2,
            dist_tol_z=0.2,
            angle_tol=2.0)

        self.tracked_volume = TrackedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                 material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                 offset_origin=np.array([0., 0., 0.]),
                                                 contrib_vols_list=[self.stitched_volume_step1,
                                                                    self.stitched_volume_step2])

        self.tracked_map = TrackedGrainsMap(grain_volume=self.tracked_volume,
                                            stitch_maps_list=[self.stitched_map_step1, self.stitched_map_step2])

    def test_valid_parent_grains(self):
        grain = TrackedGrain(gid=1,
                             pos_offset=np.array([1., 2., 3.]),
                             pos_sample=np.array([1., 2., 3.]),
                             UBI=self.raw_grain_1_map_1_step1.UBI,
                             volume=1000.0,
                             grain_map=self.tracked_map,
                             parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                   self.stitched_map_step2.grains[0]])

        self.assertSequenceEqual(grain.parent_stitch_grains_list,
                                 [self.stitched_map_step1.grains[0], self.stitched_map_step2.grains[0]])

    def test_parent_grain_immutable_strings(self):
        grain = TrackedGrain(gid=1,
                             pos_offset=np.array([1., 2., 3.]),
                             pos_sample=np.array([1., 2., 3.]),
                             UBI=self.raw_grain_1_map_1_step1.UBI,
                             volume=1000.0,
                             grain_map=self.tracked_map,
                             parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                   self.stitched_map_step2.grains[0]])

        self.assertSequenceEqual(grain.parent_stitch_grain_immutable_strings,
                                 [self.stitched_map_step1.grains[0].immutable_string,
                                  self.stitched_map_step2.grains[0].immutable_string])

    def test_not_all_parents_are_grains(self):
        with self.assertRaises(TypeError):
            grain = TrackedGrain(gid=1,
                                 pos_offset=np.array([1., 2., 3.]),
                                 pos_sample=np.array([1., 2., 3.]),
                                 UBI=self.raw_grain_1_map_1_step1.UBI,
                                 volume=1000.0,
                                 grain_map=self.tracked_map,
                                 parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                       "self.stitched_map_step2.grains[0]"])

    def test_wrong_parent_grain_types(self):
        base_grain = BaseGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1_map_1_step1.UBI,
                               volume=200.0)

        with self.assertRaises(TypeError):
            grain = TrackedGrain(gid=1,
                                 pos_offset=np.array([1., 2., 3.]),
                                 pos_sample=np.array([1., 2., 3.]),
                                 UBI=self.raw_grain_1_map_1_step1.UBI,
                                 volume=1000.0,
                                 grain_map=self.tracked_map,
                                 parent_stitch_grains=[self.stitched_map_step1.grains[0],
                                                       base_grain])

    def test_degen_load_steps(self):
        print("HELP!")
        # TODO: Finish up from here
        # Test "from stitched maps list" method
        # This is the only way to test the ACTUAL tracking is working


#
# class TestGetNearestNeighboursFromGrainList(unittest.TestCase):
#
#
#     def test_valid_query(self):
#         grains_list = self.tracked_map.grains
#         print(grains_list)
#         n_neighbours = 10
#         max_distance = 0.2
#         # only grain_2 from the list should be a neighbour of grain_1
#         neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                n_neighbours=n_neighbours,
#                                                                                max_distance=max_distance)
#         self.assertSequenceEqual(neighbour_grains, [self.grain_2])
#
#     def test_limited_n(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         n_neighbours = 1
#         max_distance = 10.0
#         # only grain_2 from the list because it's closest
#         neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                n_neighbours=n_neighbours,
#                                                                                max_distance=max_distance)
#         self.assertSequenceEqual(neighbour_grains, [self.grain_2])
#
#     def test_none_in_range(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         n_neighbours = 6
#         max_distance = 0.05
#         with self.assertRaises(ValueError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)
#
#     def test_empty_list(self):
#         grains_list = []
#         n_neighbours = 6
#         max_distance = 0.05
#         with self.assertRaises(ValueError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)
#
#     def test_mixed_type_list(self):
#         grains_list = [self.grain_2, self.grain_3, "grain_4"]
#         n_neighbours = 6
#         max_distance = 0.05
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)
#
#     def test_mixed_grain_type_list(self):
#         base_grain = BaseGrain(pos=np.array([1.0, 2.0, 3.0]),
#                                UBI=np.array([[3.0, 0, 0],
#                                              [0, 3.0, 0],
#                                              [0, 0, 3.0]]),
#                                volume=100.0)
#         grains_list = [self.grain_2, self.grain_3, base_grain]
#         n_neighbours = 6
#         max_distance = 0.05
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)
#
#     def test_wrong_n_neighbours_type(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         n_neighbours = 6.0
#         max_distance = 0.05
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)
#
#     def test_wrong_max_distance_type(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         n_neighbours = 6
#         max_distance = 1
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    n_neighbours=n_neighbours,
#                                                                                    max_distance=max_distance)


#     def test_mixed_sample(self):
#         # Make a blank sample
#         sample_2 = Sample(name="test_sample_name_2")
#         # Make a blank load step
#         load_step = LoadStep(name="test_load_step_name", sample=sample_2)
#
#         # Make grain volumes
#         grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=load_step,
#                                      index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, 0.5]))
#
#         raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)
#
#         raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#         raw_orientation_3 = (
#                 Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)
#
#         raw_grain_1_map_2 = RawGrain(gid=1,
#                                      pos=np.array([1.1, 2, 2.5]),
#                                      UBI=raw_ubi_1,
#                                      volume=250.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_2_map_2 = RawGrain(gid=2,
#                                      pos=np.array([1.1, 2, 2.6]),
#                                      UBI=raw_ubi_2,
#                                      volume=750.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_3_map_2 = RawGrain(gid=3,
#                                      pos=np.array([-1, -2, 3.5]),
#                                      UBI=raw_ubi_3,
#                                      volume=100.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])
#
#         clean_grain_map_2 = CleanedGrainsMap(grain_volume=grain_volume_2, raw_map=raw_grain_map_2)
#
#         clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
#                                                           grain_map=clean_grain_map_2)
#         with self.assertRaises(ValueError):
#             stitched_grain = StitchedGrain(gid=1,
#                                            pos_offset=np.array([1., 2, 3]),
#                                            UBI=self.raw_grain_1_map_1.UBI,
#                                            volume=1000.0,
#                                            grain_map=self.stitched_map,
#                                            parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])
#
#     def test_mixed_load_step(self):
#         # Make a blank load step
#         load_step = LoadStep(name="test_load_step_name", sample=self.sample)
#
#         # Make grain volumes
#         grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=load_step,
#                                      index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, 0.5]))
#
#         raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)
#
#         raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#         raw_orientation_3 = (
#                 Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)
#
#         raw_grain_1_map_2 = RawGrain(gid=1,
#                                      pos=np.array([1.1, 2, 2.5]),
#                                      UBI=raw_ubi_1,
#                                      volume=250.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_2_map_2 = RawGrain(gid=2,
#                                      pos=np.array([1.1, 2, 2.6]),
#                                      UBI=raw_ubi_2,
#                                      volume=750.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_3_map_2 = RawGrain(gid=3,
#                                      pos=np.array([-1, -2, 3.5]),
#                                      UBI=raw_ubi_3,
#                                      volume=100.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])
#
#         clean_grain_map_2 = CleanedGrainsMap(grain_volume=grain_volume_2, raw_map=raw_grain_map_2)
#
#         clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
#                                                           grain_map=clean_grain_map_2)
#         with self.assertRaises(ValueError):
#             stitched_grain = StitchedGrain(gid=1,
#                                            pos_offset=np.array([1., 2, 3]),
#                                            UBI=self.raw_grain_1_map_1.UBI,
#                                            volume=1000.0,
#                                            grain_map=self.stitched_map,
#                                            parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])
#
#     def test_mixed_phase(self):
#         phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
#                       symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
#         # Make a blank load step
#         load_step = LoadStep(name="test_load_step_name", sample=self.sample)
#
#         # Make grain volumes
#         grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=load_step,
#                                      index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, 0.5]))
#
#         raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=phase)
#
#         raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#         raw_orientation_3 = (
#                 Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)
#
#         raw_grain_1_map_2 = RawGrain(gid=1,
#                                      pos=np.array([1.1, 2, 2.5]),
#                                      UBI=raw_ubi_1,
#                                      volume=250.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_2_map_2 = RawGrain(gid=2,
#                                      pos=np.array([1.1, 2, 2.6]),
#                                      UBI=raw_ubi_2,
#                                      volume=750.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_3_map_2 = RawGrain(gid=3,
#                                      pos=np.array([-1, -2, 3.5]),
#                                      UBI=raw_ubi_3,
#                                      volume=100.0,
#                                      grain_map=raw_grain_map_2)
#         raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])
#
#         clean_grain_map_2 = CleanedGrainsMap(grain_volume=grain_volume_2, raw_map=raw_grain_map_2)
#
#         clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
#                                                           grain_map=clean_grain_map_2)
#         with self.assertRaises(ValueError):
#             stitched_grain = StitchedGrain(gid=1,
#                                            pos_offset=np.array([1., 2, 3]),
#                                            UBI=self.raw_grain_1_map_1.UBI,
#                                            volume=1000.0,
#                                            grain_map=self.stitched_map,
#                                            parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])
#
#     def test_differing_load_step(self):
#         # Make a blank load step
#         load_step = LoadStep(name="test_load_step_name_2", sample=self.sample)
#
#         # Make two grain volumes
#         grain_volume_1 = GrainVolume(name="test_grain_volume_1", load_step=load_step,
#                                      index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, -0.5]))
#         grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=load_step,
#                                      index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, 0.5]))
#
#         raw_grain_map_1 = RawGrainsMap(grain_volume=grain_volume_1, phase=self.phase)
#         raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)
#
#         raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#         raw_orientation_3 = (
#                 Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)
#
#         raw_grain_1_map_1 = RawGrain(gid=1,
#                                      pos=np.array([1., 2, 3.5]),
#                                      UBI=raw_ubi_1,
#                                      volume=250.0,
#                                      grain_map=raw_grain_map_1)
#
#         raw_grain_2_map_1 = RawGrain(gid=2,
#                                      pos=np.array([1, 2, 3.6]),
#                                      UBI=raw_ubi_2,
#                                      volume=750.0,
#                                      grain_map=raw_grain_map_1)
#
#         raw_grain_3_map_1 = RawGrain(gid=3,
#                                      pos=np.array([-1, -2, 4.5]),
#                                      UBI=raw_ubi_3,
#                                      volume=100.0,
#                                      grain_map=raw_grain_map_1)
#
#         raw_grain_1_map_2 = RawGrain(gid=1,
#                                      pos=np.array([1.1, 2, 2.5]),
#                                      UBI=raw_ubi_1,
#                                      volume=250.0,
#                                      grain_map=raw_grain_map_2)
#
#         raw_grain_2_map_2 = RawGrain(gid=2,
#                                      pos=np.array([1.1, 2, 2.6]),
#                                      UBI=raw_ubi_2,
#                                      volume=750.0,
#                                      grain_map=raw_grain_map_2)
#
#         raw_grain_3_map_2 = RawGrain(gid=3,
#                                      pos=np.array([-1, -2, 3.5]),
#                                      UBI=raw_ubi_3,
#                                      volume=100.0,
#                                      grain_map=raw_grain_map_2)
#
#         raw_grain_map_1.add_grains([raw_grain_1_map_1, raw_grain_2_map_1, raw_grain_3_map_1])
#         raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])
#
#         clean_grain_map_1 = CleanedGrainsMap(grain_volume=grain_volume_1, raw_map=raw_grain_map_1)
#         clean_grain_map_2 = CleanedGrainsMap(grain_volume=grain_volume_2, raw_map=raw_grain_map_2)
#
#         clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [raw_grain_1_map_1, raw_grain_2_map_1],
#                                                           grain_map=clean_grain_map_1)
#         clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [raw_grain_3_map_1], grain_map=clean_grain_map_1)
#
#         clean_grain_map_1.add_grains([clean_grain_1_map_1, clean_grain_2_map_1])
#
#         clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
#                                                           grain_map=clean_grain_map_2)
#
#         clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [raw_grain_3_map_2], grain_map=clean_grain_map_2)
#
#         clean_grain_map_2.add_grains([clean_grain_1_map_2, clean_grain_2_map_2])
#         with self.assertRaises(ValueError):
#             stitched_grain = StitchedGrain(gid=1,
#                                            pos_offset=np.array([1., 2, 3]),
#                                            UBI=raw_grain_1_map_1.UBI,
#                                            volume=1000.0,
#                                            grain_map=self.stitched_map,
#                                            parent_clean_grains=[clean_grain_1_map_1, clean_grain_1_map_2])
#
#
# class TestStitchingProperties(unittest.TestCase):
#     def setUp(self):
#         # Make a blank sample
#         self.sample = Sample(name="test_sample_name")
#         # Make a blank load step
#         self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
#         # Make a blank phase
#         self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
#                            symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
#
#         # Make two grain volumes
#         self.grain_volume_1 = GrainVolume(name="test_grain_volume_1", load_step=self.load_step,
#                                           index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, -0.5]))
#         self.grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=self.load_step,
#                                           index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.), (-0.1, 0.1)) offset_origin=np.array([0, 0, 0.5]))
#
#         self.raw_grain_map_1 = RawGrainsMap(grain_volume=self.grain_volume_1, phase=self.phase)
#         self.raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume_2, phase=self.phase)
#
#         raw_orientation_1 = Rotation.random(1).as_matrix()[0]
#         raw_orientation_2 = (
#                 Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#         raw_orientation_3 = (
#                 Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
#
#         raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
#         raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
#         raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)
#
#         self.raw_grain_1_map_1 = RawGrain(gid=1,
#                                           pos=np.array([1., 2, 3.5]),
#                                           UBI=raw_ubi_1,
#                                           volume=250.0,
#                                           grain_map=self.raw_grain_map_1)
#
#         self.raw_grain_2_map_1 = RawGrain(gid=2,
#                                           pos=np.array([1, 2, 3.6]),
#                                           UBI=raw_ubi_2,
#                                           volume=750.0,
#                                           grain_map=self.raw_grain_map_1)
#
#         self.raw_grain_3_map_1 = RawGrain(gid=3,
#                                           pos=np.array([-1, -2, 4.5]),
#                                           UBI=raw_ubi_3,
#                                           volume=100.0,
#                                           grain_map=self.raw_grain_map_1)
#
#         self.raw_grain_1_map_2 = RawGrain(gid=1,
#                                           pos=np.array([1.1, 2, 2.5]),
#                                           UBI=raw_ubi_1,
#                                           volume=250.0,
#                                           grain_map=self.raw_grain_map_2)
#
#         self.raw_grain_2_map_2 = RawGrain(gid=2,
#                                           pos=np.array([1.1, 2, 2.6]),
#                                           UBI=raw_ubi_2,
#                                           volume=750.0,
#                                           grain_map=self.raw_grain_map_2)
#
#         self.raw_grain_3_map_2 = RawGrain(gid=3,
#                                           pos=np.array([-1, -2, 3.5]),
#                                           UBI=raw_ubi_3,
#                                           volume=100.0,
#                                           grain_map=self.raw_grain_map_2)
#
#         self.raw_grain_map_1.add_grains([self.raw_grain_1_map_1, self.raw_grain_2_map_1, self.raw_grain_3_map_1])
#
#         self.raw_grain_map_2.add_grains([self.raw_grain_1_map_2, self.raw_grain_2_map_2, self.raw_grain_3_map_2])
#
#         self.clean_grain_map_1 = CleanedGrainsMap(grain_volume=self.grain_volume_1, raw_map=self.raw_grain_map_1)
#
#         self.clean_grain_map_2 = CleanedGrainsMap(grain_volume=self.grain_volume_2, raw_map=self.raw_grain_map_2)
#
#         self.clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1, self.raw_grain_2_map_1],
#                                                                grain_map=self.clean_grain_map_1)
#
#         self.clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1],
#                                                                grain_map=self.clean_grain_map_1)
#
#         self.clean_grain_map_1.add_grains([self.clean_grain_1_map_1, self.clean_grain_2_map_1])
#
#         self.clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2, self.raw_grain_2_map_2],
#                                                                grain_map=self.clean_grain_map_2)
#
#         self.clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2],
#                                                                grain_map=self.clean_grain_map_2)
#
#         self.clean_grain_map_2.add_grains([self.clean_grain_1_map_2, self.clean_grain_2_map_2])
#
#         self.stitched_volume = StitchedGrainVolume(name="stitched_grain_volume",
#                                                    load_step=self.load_step,
#                                                    index_dimensions=((0., 0.), (0., 0.), (0., 0.)), material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
#                                                    offset_origin=np.array([0., 0., 0.]),
#                                                    contrib_vols_list=[self.grain_volume_1, self.grain_volume_2])
#
#         self.stitched_map = StitchedGrainsMap(grain_volume=self.stitched_volume,
#                                               clean_maps_list=[self.clean_grain_map_1, self.clean_grain_map_2])
#
#     def test_stitched_properties(self):
#         stitched_grain = StitchedGrain.from_grains_list(gid=1,
#                                                         grains_to_merge=[self.clean_grain_1_map_1,
#                                                                          self.clean_grain_1_map_2],
#                                                         grain_map=self.stitched_map)
#         self.assertTrue(np.allclose(stitched_grain.pos, np.array([1.05, 2, 3.075])))
#         self.assertTrue(np.allclose(stitched_grain.UBI, self.raw_grain_2_map_1.UBI))
#         self.assertEqual(stitched_grain.volume, self.clean_grain_1_map_1.volume + self.clean_grain_1_map_2.volume)
#         self.assertTrue(np.allclose(stitched_grain.pos_offset, np.array([1.05, 2, 3.075])))
#         self.assertTrue(isinstance(stitched_grain, StitchedGrain))
#         self.assertEqual(stitched_grain.load_step, self.load_step)
#         self.assertEqual(stitched_grain.sample, self.sample)
#         self.assertSequenceEqual(stitched_grain.parent_clean_grains,
#                                  [self.clean_grain_1_map_1, self.clean_grain_1_map_2])
#         self.assertTrue(
#             np.allclose(stitched_grain.grain_volume.offset_origin, stitched_grain.pos_offset - stitched_grain.pos))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
