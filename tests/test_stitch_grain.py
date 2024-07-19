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
from py3DXRDProc.grain import CleanGrain, BaseGrain, StitchedGrain
from py3DXRDProc.grain_map import CleanedGrainsMap, RawGrainsMap, RawGrain, StitchedGrainsMap
from py3DXRDProc.grain_volume import GrainVolume, StitchedGrainVolume
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
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1 = GrainVolume(name="test_grain_volume_1",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1 = RawGrainsMap(grain_volume=self.grain_volume_1, phase=self.phase)
        self.raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume_2, phase=self.phase)

        self.grain_volume_1.add_raw_map(self.raw_grain_map_1)
        self.grain_volume_2.add_raw_map(self.raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1 = RawGrain(gid=1,
                                          pos=np.array([1., 2, 3.5]),
                                          UBI=raw_ubi_1,
                                          volume=250.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_2_map_1 = RawGrain(gid=2,
                                          pos=np.array([1, 2, 3.6]),
                                          UBI=raw_ubi_2,
                                          volume=750.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_3_map_1 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 4.5]),
                                          UBI=raw_ubi_3,
                                          volume=100.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_1_map_2 = RawGrain(gid=1,
                                          pos=np.array([1.1, 2, 2.5]),
                                          UBI=raw_ubi_1,
                                          volume=250.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_2_map_2 = RawGrain(gid=2,
                                          pos=np.array([1.1, 2, 2.6]),
                                          UBI=raw_ubi_2,
                                          volume=750.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_3_map_2 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 3.5]),
                                          UBI=raw_ubi_3,
                                          volume=100.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_map_1.add_grains([self.raw_grain_1_map_1, self.raw_grain_2_map_1, self.raw_grain_3_map_1])

        self.raw_grain_map_2.add_grains([self.raw_grain_1_map_2, self.raw_grain_2_map_2, self.raw_grain_3_map_2])

        self.clean_grain_map_1 = CleanedGrainsMap(raw_map=self.raw_grain_map_1)

        self.clean_grain_map_2 = CleanedGrainsMap(raw_map=self.raw_grain_map_2)

        self.clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1, self.raw_grain_2_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_map_1.add_grains([self.clean_grain_1_map_1, self.clean_grain_2_map_1])

        self.clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2, self.raw_grain_2_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_map_2.add_grains([self.clean_grain_1_map_2, self.clean_grain_2_map_2])

        self.stitched_volume = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                   material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                   offset_origin=np.array([0., 0., 0.]),
                                                   contrib_vols_list=[self.grain_volume_1, self.grain_volume_2])

        self.stitched_map = StitchedGrainsMap(grain_volume=self.stitched_volume,
                                              clean_maps_list=[self.clean_grain_map_1, self.clean_grain_map_2])

    def test_wrong_grain_map_type(self):
        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2., 3.]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.raw_grain_map_1,
                                           parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

    def test_wrong_pos_type(self):
        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset="position",
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

    def test_wrong_pos_shape(self):
        with self.assertRaises(ValueError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2., 3., 4.]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

    def test_wrong_pos_element_type(self):
        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1, 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

    def test_wrong_parent_grains_type(self):
        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=(self.clean_grain_1_map_1, self.clean_grain_1_map_2))


class TestParentGrains(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1 = GrainVolume(name="test_grain_volume_1",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1 = RawGrainsMap(grain_volume=self.grain_volume_1, phase=self.phase)
        self.raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume_2, phase=self.phase)

        self.grain_volume_1.add_raw_map(self.raw_grain_map_1)
        self.grain_volume_2.add_raw_map(self.raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1 = RawGrain(gid=1,
                                          pos=np.array([1., 2, 3.5]),
                                          UBI=raw_ubi_1,
                                          volume=250.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_2_map_1 = RawGrain(gid=2,
                                          pos=np.array([1, 2, 3.6]),
                                          UBI=raw_ubi_2,
                                          volume=750.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_3_map_1 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 4.5]),
                                          UBI=raw_ubi_3,
                                          volume=100.0,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1.0)

        self.raw_grain_1_map_2 = RawGrain(gid=1,
                                          pos=np.array([1.1, 2, 2.5]),
                                          UBI=raw_ubi_1,
                                          volume=250.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_2_map_2 = RawGrain(gid=2,
                                          pos=np.array([1.1, 2, 2.6]),
                                          UBI=raw_ubi_2,
                                          volume=750.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_3_map_2 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 3.5]),
                                          UBI=raw_ubi_3,
                                          volume=100.0,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_map_1.add_grains([self.raw_grain_1_map_1, self.raw_grain_2_map_1, self.raw_grain_3_map_1])

        self.raw_grain_map_2.add_grains([self.raw_grain_1_map_2, self.raw_grain_2_map_2, self.raw_grain_3_map_2])

        self.clean_grain_map_1 = CleanedGrainsMap(raw_map=self.raw_grain_map_1)

        self.clean_grain_map_2 = CleanedGrainsMap(raw_map=self.raw_grain_map_2)

        self.clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1, self.raw_grain_2_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_map_1.add_grains([self.clean_grain_1_map_1, self.clean_grain_2_map_1])

        self.clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2, self.raw_grain_2_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_map_2.add_grains([self.clean_grain_1_map_2, self.clean_grain_2_map_2])

        self.stitched_volume = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                   material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                   offset_origin=np.array([0., 0., 0.]),
                                                   contrib_vols_list=[self.grain_volume_1, self.grain_volume_2])

        self.stitched_map = StitchedGrainsMap(grain_volume=self.stitched_volume,
                                              clean_maps_list=[self.clean_grain_map_1, self.clean_grain_map_2])

    def test_valid_parent_grains(self):
        grain = StitchedGrain(pos_offset=np.array([1., 2, 3]),
                              UBI=self.raw_grain_1_map_1.UBI,
                              volume=200.0,
                              gid=1,
                              grain_map=self.stitched_map,
                              parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

        self.assertSequenceEqual(grain.parent_clean_grains, [self.clean_grain_1_map_1, self.clean_grain_1_map_2])

    def test_parent_grain_immutable_strings(self):
        grain = StitchedGrain(pos_offset=np.array([1., 2, 3]),
                              UBI=self.raw_grain_1_map_1.UBI,
                              volume=200.0,
                              gid=1,
                              grain_map=self.stitched_map,
                              parent_clean_grains=[self.clean_grain_1_map_1, self.clean_grain_1_map_2])

        self.assertSequenceEqual(grain.parent_clean_grain_immutable_strings,
                                 [self.clean_grain_1_map_1.immutable_string, self.clean_grain_1_map_2.immutable_string])

    def test_not_all_parents_are_grains(self):
        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, "another_grain"])

    def test_wrong_parent_grain_types(self):
        base_grain = BaseGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1_map_1.UBI,
                               volume=200.0)

        with self.assertRaises(TypeError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, base_grain])

    def test_mixed_sample(self):
        # Make a blank sample
        sample_2 = Sample(name="test_sample_name_2")
        # Make a blank load step
        load_step = LoadStep(name="test_load_step_name", sample=sample_2)

        # Make grain volumes
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0, 0, 0.5]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)
        grain_volume_2.add_raw_map(raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        raw_grain_1_map_2 = RawGrain(gid=1,
                                     pos=np.array([1.1, 2, 2.5]),
                                     UBI=raw_ubi_1,
                                     volume=250.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_2_map_2 = RawGrain(gid=2,
                                     pos=np.array([1.1, 2, 2.6]),
                                     UBI=raw_ubi_2,
                                     volume=750.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_3_map_2 = RawGrain(gid=3,
                                     pos=np.array([-1, -2, 3.5]),
                                     UBI=raw_ubi_3,
                                     volume=100.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])

        clean_grain_map_2 = CleanedGrainsMap(raw_map=raw_grain_map_2)

        clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
                                                          grain_map=clean_grain_map_2)
        with self.assertRaises(ValueError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])

    def test_mixed_load_step(self):
        # Make a blank load step
        load_step = LoadStep(name="test_load_step_name", sample=self.sample)

        # Make grain volumes
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0, 0, 0.5]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)
        grain_volume_2.add_raw_map(raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        raw_grain_1_map_2 = RawGrain(gid=1,
                                     pos=np.array([1.1, 2, 2.5]),
                                     UBI=raw_ubi_1,
                                     volume=250.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_2_map_2 = RawGrain(gid=2,
                                     pos=np.array([1.1, 2, 2.6]),
                                     UBI=raw_ubi_2,
                                     volume=750.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_3_map_2 = RawGrain(gid=3,
                                     pos=np.array([-1, -2, 3.5]),
                                     UBI=raw_ubi_3,
                                     volume=100.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])

        clean_grain_map_2 = CleanedGrainsMap(raw_map=raw_grain_map_2)

        clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
                                                          grain_map=clean_grain_map_2)
        with self.assertRaises(ValueError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])

    def test_mixed_phase(self):
        phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                      symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        # Make a blank load step
        load_step = LoadStep(name="test_load_step_name", sample=self.sample)

        # Make grain volumes
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0, 0, 0.5]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=phase)
        grain_volume_2.add_raw_map(raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        raw_grain_1_map_2 = RawGrain(gid=1,
                                     pos=np.array([1.1, 2, 2.5]),
                                     UBI=raw_ubi_1,
                                     volume=250.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_2_map_2 = RawGrain(gid=2,
                                     pos=np.array([1.1, 2, 2.6]),
                                     UBI=raw_ubi_2,
                                     volume=750.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_3_map_2 = RawGrain(gid=3,
                                     pos=np.array([-1, -2, 3.5]),
                                     UBI=raw_ubi_3,
                                     volume=100.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)
        raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])

        clean_grain_map_2 = CleanedGrainsMap(raw_map=raw_grain_map_2)

        clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
                                                          grain_map=clean_grain_map_2)
        with self.assertRaises(ValueError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=self.raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[self.clean_grain_1_map_1, clean_grain_1_map_2])

    def test_differing_load_step(self):
        # Make a blank load step
        load_step = LoadStep(name="test_load_step_name_2", sample=self.sample)

        # Make two grain volumes
        grain_volume_1 = GrainVolume(name="test_grain_volume_1",
                                     load_step=load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0, 0, -0.5]))
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0, 0, 0.5]))

        raw_grain_map_1 = RawGrainsMap(grain_volume=grain_volume_1, phase=self.phase)
        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        grain_volume_1.add_raw_map(raw_grain_map_1)
        grain_volume_2.add_raw_map(raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        raw_grain_1_map_1 = RawGrain(gid=1,
                                     pos=np.array([1., 2, 3.5]),
                                     UBI=raw_ubi_1,
                                     volume=250.0,
                                     grain_map=raw_grain_map_1,
                                     mean_peak_intensity=1.0)

        raw_grain_2_map_1 = RawGrain(gid=2,
                                     pos=np.array([1, 2, 3.6]),
                                     UBI=raw_ubi_2,
                                     volume=750.0,
                                     grain_map=raw_grain_map_1,
                                     mean_peak_intensity=1.0)

        raw_grain_3_map_1 = RawGrain(gid=3,
                                     pos=np.array([-1, -2, 4.5]),
                                     UBI=raw_ubi_3,
                                     volume=100.0,
                                     grain_map=raw_grain_map_1,
                                     mean_peak_intensity=1.0)

        raw_grain_1_map_2 = RawGrain(gid=1,
                                     pos=np.array([1.1, 2, 2.5]),
                                     UBI=raw_ubi_1,
                                     volume=250.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)

        raw_grain_2_map_2 = RawGrain(gid=2,
                                     pos=np.array([1.1, 2, 2.6]),
                                     UBI=raw_ubi_2,
                                     volume=750.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)

        raw_grain_3_map_2 = RawGrain(gid=3,
                                     pos=np.array([-1, -2, 3.5]),
                                     UBI=raw_ubi_3,
                                     volume=100.0,
                                     grain_map=raw_grain_map_2,
                                     mean_peak_intensity=1.0)

        raw_grain_map_1.add_grains([raw_grain_1_map_1, raw_grain_2_map_1, raw_grain_3_map_1])
        raw_grain_map_2.add_grains([raw_grain_1_map_2, raw_grain_2_map_2, raw_grain_3_map_2])

        clean_grain_map_1 = CleanedGrainsMap(raw_map=raw_grain_map_1)
        clean_grain_map_2 = CleanedGrainsMap(raw_map=raw_grain_map_2)

        clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [raw_grain_1_map_1, raw_grain_2_map_1],
                                                          grain_map=clean_grain_map_1)
        clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [raw_grain_3_map_1], grain_map=clean_grain_map_1)

        clean_grain_map_1.add_grains([clean_grain_1_map_1, clean_grain_2_map_1])

        clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [raw_grain_1_map_2, raw_grain_2_map_2],
                                                          grain_map=clean_grain_map_2)

        clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [raw_grain_3_map_2], grain_map=clean_grain_map_2)

        clean_grain_map_2.add_grains([clean_grain_1_map_2, clean_grain_2_map_2])
        with self.assertRaises(ValueError):
            stitched_grain = StitchedGrain(gid=1,
                                           pos_offset=np.array([1., 2, 3]),
                                           UBI=raw_grain_1_map_1.UBI,
                                           volume=1000.0,
                                           grain_map=self.stitched_map,
                                           parent_clean_grains=[clean_grain_1_map_1, clean_grain_1_map_2])


class TestStitchingProperties(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        # Make two grain volumes
        self.grain_volume_1 = GrainVolume(name="test_grain_volume_1",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, -0.5]))
        self.grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                          load_step=self.load_step,
                                          index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                          material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                          offset_origin=np.array([0, 0, 0.5]))

        self.raw_grain_map_1 = RawGrainsMap(grain_volume=self.grain_volume_1, phase=self.phase)
        self.raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume_2, phase=self.phase)

        self.grain_volume_1.add_raw_map(self.raw_grain_map_1)
        self.grain_volume_2.add_raw_map(self.raw_grain_map_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()
        raw_orientation_3 = (
                Rotation.from_euler('x', -1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)
        raw_ubi_3 = tools.u_to_ubi(raw_orientation_3, self.phase.reference_unit_cell)

        self.raw_grain_1_map_1 = RawGrain(gid=1,
                                          pos=np.array([1., 2, 3.5]),
                                          UBI=raw_ubi_1,
                                          volume=0.1,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=1/8)

        self.raw_grain_2_map_1 = RawGrain(gid=2,
                                          pos=np.array([1, 2, 3.6]),
                                          UBI=raw_ubi_2,
                                          volume=0.2,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=2/8)

        self.raw_grain_3_map_1 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 4.5]),
                                          UBI=raw_ubi_3,
                                          volume=0.5,
                                          grain_map=self.raw_grain_map_1,
                                          mean_peak_intensity=5/8)

        self.raw_grain_1_map_2 = RawGrain(gid=1,
                                          pos=np.array([1.1, 2, 2.5]),
                                          UBI=raw_ubi_1,
                                          volume=0.1,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=1.0)

        self.raw_grain_2_map_2 = RawGrain(gid=2,
                                          pos=np.array([1.1, 2, 2.6]),
                                          UBI=raw_ubi_2,
                                          volume=0.3,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=2.0)

        self.raw_grain_3_map_2 = RawGrain(gid=3,
                                          pos=np.array([-1, -2, 3.5]),
                                          UBI=raw_ubi_3,
                                          volume=0.4,
                                          grain_map=self.raw_grain_map_2,
                                          mean_peak_intensity=5.0)

        self.raw_grain_map_1.add_grains([self.raw_grain_1_map_1, self.raw_grain_2_map_1, self.raw_grain_3_map_1])

        self.raw_grain_map_2.add_grains([self.raw_grain_1_map_2, self.raw_grain_2_map_2, self.raw_grain_3_map_2])

        self.clean_grain_map_1 = CleanedGrainsMap(raw_map=self.raw_grain_map_1)

        self.clean_grain_map_2 = CleanedGrainsMap(raw_map=self.raw_grain_map_2)

        self.clean_grain_1_map_1 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_1, self.raw_grain_2_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_2_map_1 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_1],
                                                               grain_map=self.clean_grain_map_1)

        self.clean_grain_map_1.add_grains([self.clean_grain_1_map_1, self.clean_grain_2_map_1])

        self.clean_grain_1_map_2 = CleanGrain.from_grains_list(1, [self.raw_grain_1_map_2, self.raw_grain_2_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_2_map_2 = CleanGrain.from_grains_list(2, [self.raw_grain_3_map_2],
                                                               grain_map=self.clean_grain_map_2)

        self.clean_grain_map_2.add_grains([self.clean_grain_1_map_2, self.clean_grain_2_map_2])

        self.stitched_volume = StitchedGrainVolume(index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                                   material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                   offset_origin=np.array([0., 0., 0.]),
                                                   contrib_vols_list=[self.grain_volume_1, self.grain_volume_2])

        self.stitched_map = StitchedGrainsMap(grain_volume=self.stitched_volume,
                                              clean_maps_list=[self.clean_grain_map_1, self.clean_grain_map_2])

    def test_stitched_properties(self):
        stitched_grain = StitchedGrain.from_grains_list(gid=1,
                                                        grains_to_merge=[self.clean_grain_1_map_1,
                                                                         self.clean_grain_1_map_2],
                                                        grain_map=self.stitched_map)

        clean_grain_1_map_1_pos = self.raw_grain_1_map_1.pos * 1/3 + self.raw_grain_2_map_1.pos * 2/3
        clean_grain_1_map_2_pos = self.raw_grain_1_map_2.pos * 1 / 4 + self.raw_grain_2_map_2.pos * 3 / 4
        self.assertTrue(np.allclose(clean_grain_1_map_1_pos, self.clean_grain_1_map_1.pos))
        self.assertTrue(np.allclose(clean_grain_1_map_2_pos, self.clean_grain_1_map_2.pos))

        self.assertTrue(np.allclose(stitched_grain.pos_offset, self.clean_grain_1_map_1.pos_offset * (self.clean_grain_1_map_1.volume / (self.clean_grain_1_map_1.volume + self.clean_grain_1_map_2.volume)) + self.clean_grain_1_map_2.pos_offset * (self.clean_grain_1_map_2.volume / (self.clean_grain_1_map_1.volume + self.clean_grain_1_map_2.volume))))
        self.assertTrue(np.allclose(stitched_grain.UBI, self.raw_grain_2_map_1.UBI))
        self.assertEqual(stitched_grain.volume, self.clean_grain_1_map_1.volume + self.clean_grain_1_map_2.volume)
        self.assertTrue(np.allclose(stitched_grain.pos_offset, stitched_grain.pos))
        self.assertTrue(isinstance(stitched_grain, StitchedGrain))
        self.assertEqual(stitched_grain.load_step, self.load_step)
        self.assertEqual(stitched_grain.sample, self.sample)
        self.assertSequenceEqual(stitched_grain.parent_clean_grains,
                                 [self.clean_grain_1_map_1, self.clean_grain_1_map_2])
        self.assertTrue(
            np.allclose(stitched_grain.grain_volume.offset_origin, stitched_grain.pos_offset - stitched_grain.pos))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
