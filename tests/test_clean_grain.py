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
from py3DXRDProc.grain import CleanGrain, BaseGrain
from py3DXRDProc.grain_map import CleanedGrainsMap, RawGrainsMap, RawGrain
from py3DXRDProc.grain_volume import GrainVolume
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
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        self.raw_grain_1 = RawGrain(gid=1,
                                    pos=np.array([1., 2, 3]),
                                    UBI=raw_ubi_1,
                                    volume=100.0,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=1.0)

        self.raw_grain_2 = RawGrain(gid=2,
                                    pos=np.array([1, 2, 3.1]),
                                    UBI=raw_ubi_2,
                                    volume=100.0,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=1.0)

        self.raw_grain_map.add_grains([self.raw_grain_1, self.raw_grain_2])

        self.grain_map = CleanedGrainsMap(raw_map=self.raw_grain_map)

    def test_wrong_grain_map_type(self):
        with self.assertRaises(TypeError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.raw_grain_map,
                               parent_grains=[self.raw_grain_1, self.raw_grain_2])

    def test_wrong_parent_grains_type(self):
        with self.assertRaises(TypeError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=(self.raw_grain_1, self.raw_grain_2))


class TestParentGrains(unittest.TestCase):
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
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        self.raw_grain_1 = RawGrain(gid=1,
                                    pos=np.array([1., 2, 3]),
                                    UBI=raw_ubi_1,
                                    volume=100.0,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=1.0)

        self.raw_grain_2 = RawGrain(gid=2,
                                    pos=np.array([1, 2, 3.1]),
                                    UBI=raw_ubi_2,
                                    volume=100.0,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=1.0)

        self.raw_grain_map.add_grains([self.raw_grain_1, self.raw_grain_2])

        self.grain_map = CleanedGrainsMap(raw_map=self.raw_grain_map)

    def test_valid_parent_grains(self):
        grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                           UBI=self.raw_grain_1.UBI,
                           volume=200.0,
                           gid=1,
                           grain_map=self.grain_map,
                           parent_grains=[self.raw_grain_1, self.raw_grain_2])

        self.assertSequenceEqual(grain.parent_grains, self.raw_grain_map.grains)

    def test_parent_grain_immutable_strings(self):
        grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                           UBI=self.raw_grain_1.UBI,
                           volume=200.0,
                           gid=1,
                           grain_map=self.grain_map,
                           parent_grains=[self.raw_grain_1, self.raw_grain_2])

        self.assertSequenceEqual(grain.parent_grain_immutable_strings,
                                 [self.raw_grain_1.immutable_string, self.raw_grain_2.immutable_string])

    def test_not_all_parents_are_grains(self):
        with self.assertRaises(TypeError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[self.raw_grain_1, "raw_grain_2"])

    def test_wrong_parent_grain_types(self):
        base_grain = BaseGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0)

        with self.assertRaises(TypeError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[self.raw_grain_1, base_grain])

    def test_mixed_sample(self):
        # Make a blank sample
        sample_2 = Sample(name="test_sample_name_2")
        # Make a blank load step
        load_step_2 = LoadStep(name="test_load_step_name_2", sample=sample_2)
        # Make a blank grain volume
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step_2,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0., 0., 0.]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_mixed_load_step(self):
        # Make a blank load step
        load_step_2 = LoadStep(name="test_load_step_name_2", sample=self.sample)
        # Make a blank grain volume
        grain_volume_2 = GrainVolume(name="test_grain_volume_2", load_step=load_step_2,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0., 0., 0.]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_mixed_grain_volume(self):
        # Make a blank grain volume
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=self.load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0., 0., 0.]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_mixed_map(self):
        raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_mixed_phase(self):
        phase_2 = Phase(name="test_phase_2", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                        symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume, phase=phase_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, phase_2.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=self.raw_grain_map,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_different_phase(self):
        phase_2 = Phase(name="test_phase_2", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                        symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))

        raw_grain_map_2 = RawGrainsMap(grain_volume=self.grain_volume, phase=phase_2)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, phase_2.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, phase_2.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_different_load_step(self):
        # Make a blank load step
        load_step_2 = LoadStep(name="test_load_step_name_2", sample=self.sample)
        # Make a blank grain volume
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=load_step_2,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0., 0., 0.]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])

    def test_different_grain_volume(self):
        # Make a blank grain volume
        grain_volume_2 = GrainVolume(name="test_grain_volume_2",
                                     load_step=self.load_step,
                                     index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                     material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                     offset_origin=np.array([0., 0., 0.]))

        raw_grain_map_2 = RawGrainsMap(grain_volume=grain_volume_2, phase=self.phase)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        raw_grain_1 = RawGrain(gid=1,
                               pos=np.array([1., 2, 3]),
                               UBI=raw_ubi_1,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        raw_grain_2 = RawGrain(gid=2,
                               pos=np.array([1, 2, 3.1]),
                               UBI=raw_ubi_2,
                               volume=100.0,
                               grain_map=raw_grain_map_2,
                               mean_peak_intensity=1.0)

        with self.assertRaises(ValueError):
            grain = CleanGrain(pos=np.array([1, 2, 3.05]),
                               UBI=self.raw_grain_1.UBI,
                               volume=200.0,
                               gid=1,
                               grain_map=self.grain_map,
                               parent_grains=[raw_grain_1, raw_grain_2])


class TestFromGrainsList(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),  # 0.8 mm cubed
                                        offset_origin=np.array([0, 0, 1.0]))
        # Make a blank phase
        self.phase = Phase(name="test_phase",
                           reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic,
                           lattice=Lattice.cubic(3))

        self.raw_grain_map = RawGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain_volume.add_raw_map(self.raw_grain_map)

        raw_orientation_1 = Rotation.random(1).as_matrix()[0]
        raw_orientation_2 = (
                Rotation.from_euler('x', 1, degrees=True) * Rotation.from_matrix(raw_orientation_1)).as_matrix()

        raw_ubi_1 = tools.u_to_ubi(raw_orientation_1, self.phase.reference_unit_cell)
        raw_ubi_2 = tools.u_to_ubi(raw_orientation_2, self.phase.reference_unit_cell)

        self.raw_grain_1 = RawGrain(gid=1,
                                    pos=np.array([1., 2, 3]),
                                    UBI=raw_ubi_1,
                                    volume=0.2,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=1.0)

        self.raw_grain_2 = RawGrain(gid=2,
                                    pos=np.array([1, 2, 3.1]),
                                    UBI=raw_ubi_2,
                                    volume=0.6,
                                    grain_map=self.raw_grain_map,
                                    mean_peak_intensity=2.0)

        self.raw_grain_map.add_grains([self.raw_grain_1, self.raw_grain_2])

        self.grain_map = CleanedGrainsMap.from_cleaning_grain_map(input_grain_map=self.raw_grain_map,
                                                                  dist_tol=0.2,
                                                                  angle_tol=2.0)

    def test_valid_grains_list(self):
        new_grain_obj = CleanGrain.from_grains_list(1, self.raw_grain_map.grains, self.grain_map)
        self.assertTrue(np.allclose(new_grain_obj.pos, np.array([1, 2, 3.075])))
        self.assertTrue(np.allclose(new_grain_obj.U, self.raw_grain_2.U))
        self.assertEqual(new_grain_obj.volume, self.raw_grain_1.volume + self.raw_grain_2.volume)
        self.assertTrue(np.allclose(new_grain_obj.pos_offset, np.array([1, 2, 4.075])))
        self.assertTrue(isinstance(new_grain_obj, CleanGrain))
        self.assertEqual(new_grain_obj.load_step, self.load_step)
        self.assertEqual(new_grain_obj.sample, self.sample)
        self.assertSequenceEqual(new_grain_obj.parent_grains, self.raw_grain_map.grains)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
