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
from ImageD11.grain import grain as id11_grain
from py3DXRDProc.grain import BaseMapGrain, TBaseGrain, BaseGrain
from py3DXRDProc.grain_map import BaseGrainsMap
from py3DXRDProc.grain_volume import GrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample
from pymicro.crystal.lattice import Symmetry, Lattice


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
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

    def test_wrong_gid_type(self):
        with self.assertRaises(TypeError):
            grain = BaseMapGrain(pos=np.array([1., 2, 3]),
                                 UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                               [1.04918116, 1.75653739, -2.01200437],
                                               [1.17321589, 1.63886394, 2.04256107]]),
                                 volume=228.872582,
                                 gid=1.0,
                                 grain_map=self.grain_map)

    def test_wrong_grain_map_type(self):
        with self.assertRaises(TypeError):
            grain = BaseMapGrain(pos=np.array([1., 2, 3]),
                                 UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                               [1.04918116, 1.75653739, -2.01200437],
                                               [1.17321589, 1.63886394, 2.04256107]]),
                                 volume=228.872582,
                                 gid=1,
                                 grain_map="grainmap")


class TestGID(unittest.TestCase):
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
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_gid(self):
        self.assertEqual(self.grain.gid, 1)

    def test_set_gid_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.gid = 2


class TestPosOffset(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_pos_offset(self):
        self.assertTrue(np.array_equal(self.grain.pos_offset, np.array([1, 1, 1]) + self.grain.pos))

    def test_set_pos_offset_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.pos_offset = 4


class TestVolumeScaled(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=200.0,
            gid=1,
            grain_map=self.grain_map)


class TestImmutableString(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_immutable_string(self):
        self.assertEqual(self.grain.immutable_string,
                         "test_sample_name:test_load_step_name:test_grain_volume:test_phase:1")

    def test_set_immutable_string_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.immutable_string = 4


class TestGrainMap(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_grain_map(self):
        self.assertEqual(self.grain.grain_map, self.grain_map)

    def test_set_grain_map_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.grain_map = 4


class TestPhase(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_phase(self):
        self.assertEqual(self.grain.phase, self.phase)

    def test_set_phase_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.phase = 4


class TestGrainVolume(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_grain_volume(self):
        self.assertEqual(self.grain.grain_volume, self.grain_volume)

    def test_set_grain_volume_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.grain_volume = 4


class TestLoadStep(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_load_step(self):
        self.assertEqual(self.grain.load_step, self.load_step)

    def test_set_load_step_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.load_step = 4


class TestSample(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_sample(self):
        self.assertEqual(self.grain.sample, self.sample)

    def test_set_sample_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.sample = 4


class TestRefUnitCell(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    def test_get_ref_unit_cell(self):
        self.assertTrue(np.allclose(self.grain.reference_unit_cell, self.phase.reference_unit_cell))

    def test_set_ref_unit_cell_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.reference_unit_cell = 4


class TestEps(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)
        self.ID11_grain = id11_grain(ubi=np.array([[2.3994061, -1.56941634, -0.118949452],
                                                   [1.04918116, 1.75653739, -2.01200437],
                                                   [1.17321589, 1.63886394, 2.04256107]]),
                                     translation=np.array([1, 2, 3]))

    def test_get_eps(self):
        self.assertTrue(
            np.allclose(self.grain.eps, self.ID11_grain.eps_grain_matrix(dzero_cell=self.grain.reference_unit_cell)))

    def test_set_eps_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.eps = 4


class TestEpsSampleRef(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)
        self.ID11_grain = id11_grain(ubi=np.array([[2.3994061, -1.56941634, -0.118949452],
                                                   [1.04918116, 1.75653739, -2.01200437],
                                                   [1.17321589, 1.63886394, 2.04256107]]),
                                     translation=np.array([1, 2, 3]))

    def test_get_eps_lab(self):
        self.assertTrue(
            np.allclose(self.grain.eps_lab,
                        self.ID11_grain.eps_sample_matrix(dzero_cell=self.grain.reference_unit_cell)))

    def test_set_eps_lab_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.eps_lab = 4


class TestEpsHydro(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)
        self.ID11_grain = id11_grain(ubi=np.array([[2.3994061, -1.56941634, -0.118949452],
                                                   [1.04918116, 1.75653739, -2.01200437],
                                                   [1.17321589, 1.63886394, 2.04256107]]),
                                     translation=np.array([1, 2, 3]))

    def test_get_eps_hydro(self):
        self.assertAlmostEqual(self.grain.eps_hydro[0,0], (1 / 3) * float(
            self.ID11_grain.eps_sample_matrix(dzero_cell=self.grain.reference_unit_cell)[0, 0] +
            self.ID11_grain.eps_sample_matrix(dzero_cell=self.grain.reference_unit_cell)[1, 1] +
            self.ID11_grain.eps_sample_matrix(dzero_cell=self.grain.reference_unit_cell)[2, 2]))

    def test_set_eps_hydro_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.eps_hydro = 4


class TestToGffLine(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample_name")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
        # Make a blank grain volume that's offset
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
                                        material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                        offset_origin=np.array([1., 1, 1]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)

        self.grain = BaseMapGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582,
            gid=1,
            grain_map=self.grain_map)

    # def test_to_gff_line(self):
    #     self.maxDiff = None
    #     calced_string = self.grain.to_gff_line(header_list=["gid", "volume", "pos", "rod", "U", "eps", "eps_lab"])
    #
    #     desired_string = "1 " \
    #                      "228.87258200 " \
    #                      "1.00000000 2.00000000 3.00000000 " \
    #                      "0.40260729 -0.14249629 0.28877142 " \
    #                      "0.83615861 0.36562459 0.40884891 -0.54691909 0.61212809 0.57112058 -0.04145218 -0.70115467 0.71180324 " \
    #                      "-4.34804844e-02 -4.34804821e-02 -4.34804838e-02 -1.56732946e-10 -1.94158033e-10 6.30481684e-10 -4.34804838e-02 -4.34804838e-02 -4.34804828e-02 -4.80604268e-10 -9.25968434e-10 7.40177926e-10 \n"
    #     self.assertEqual(calced_string, desired_string)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            self.grain.to_gff_line("wow!")


# class TestGetNearestNeighboursFromGrainList(unittest.TestCase):
#     def setUp(self):
#         # Make a blank sample
#         self.sample = Sample(name="test_sample_name")
#         # Make a blank load step
#         self.load_step = LoadStep(name="test_load_step_name", sample=self.sample)
#         # Make a blank grain volume that's offset
#         self.grain_volume = GrainVolume(name="test_grain_volume",
#                                         load_step=self.load_step,
#                                         index_dimensions=((0., 0.), (0., 0.), (0., 0.)),
#                                         material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
#                                         offset_origin=np.array([1., 1, 1]))
#         # Make a blank phase
#         self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
#                            symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
#         self.grain_map = BaseGrainsMap(grain_volume=self.grain_volume, phase=self.phase)
#
#         self.grain_1 = BaseMapGrain(
#             pos=np.array([0., 0., 0.]),
#             UBI=np.array([[3.0, 0, 0],
#                           [0, 3.0, 0],
#                           [0, 0, 3.0]]),
#             volume=100.0,
#             gid=1,
#             grain_map=self.grain_map)
#
#         self.grain_2 = BaseMapGrain(
#             pos=np.array([0., 0., 0.1]),
#             UBI=np.array([[3.0, 0, 0],
#                           [0, 3.0, 0],
#                           [0, 0, 3.0]]),
#             volume=100.0,
#             gid=2,
#             grain_map=self.grain_map)
#
#         self.grain_3 = BaseMapGrain(
#             pos=np.array([0., 0., 1.]),
#             UBI=np.array([[3.0, 0, 0],
#                           [0, 3.0, 0],
#                           [0, 0, 3.0]]),
#             volume=100.0,
#             gid=3,
#             grain_map=self.grain_map)
#
#         self.grain_4 = BaseMapGrain(
#             pos=np.array([0., 0., 2]),
#             UBI=np.array([[3.0, 0, 0],
#                           [0, 3.0, 0],
#                           [0, 0, 3.0]]),
#             volume=100.0,
#             gid=4,
#             grain_map=self.grain_map)
#
#         self.grain_map.add_grains([self.grain_1, self.grain_2, self.grain_3, self.grain_4])
#
#     def test_valid_query(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         max_distance = 0.2
#         # only grain_2 from the list should be a neighbour of grain_1
#         neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                max_distance=max_distance)
#         self.assertSequenceEqual(neighbour_grains, [self.grain_2])
#
#     def test_same_length(self):
#         grains_list = [self.grain_1, self.grain_2, self.grain_3, self.grain_4]
#         max_distance = 10.0
#
#         neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                max_distance=max_distance)
#         self.assertSequenceEqual(neighbour_grains, [self.grain_2, self.grain_3, self.grain_4])
#
#     def test_valid_query_larger(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         max_distance = 1.1
#         # only grain_2 from the list should be a neighbour of grain_1
#         neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                max_distance=max_distance)
#         self.assertSequenceEqual(neighbour_grains, [self.grain_2, self.grain_3])
#
#     def test_none_in_range(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         max_distance = 0.05
#         with self.assertRaises(ValueError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)
#
#     def test_none_in_range_already_in_list(self):
#         grains_list = [self.grain_2, self.grain_1, self.grain_3, self.grain_4]
#         max_distance = 0.05
#         with self.assertRaises(ValueError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)
#
#     def test_empty_list(self):
#         grains_list = []
#         max_distance = 0.05
#         with self.assertRaises(ValueError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)
#
#     def test_mixed_type_list(self):
#         grains_list = [self.grain_2, self.grain_3, "grain_4"]
#         max_distance = 0.05
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)
#
#     def test_mixed_grain_type_list(self):
#         base_grain = BaseGrain(pos=np.array([1.0, 2.0, 3.0]),
#                                UBI=np.array([[3.0, 0, 0],
#                                              [0, 3.0, 0],
#                                              [0, 0, 3.0]]),
#                                volume=100.0)
#         grains_list = [self.grain_2, self.grain_3, base_grain]
#         max_distance = 0.05
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)
#
#     def test_wrong_max_distance_type(self):
#         grains_list = [self.grain_2, self.grain_3, self.grain_4]
#         max_distance = 1
#         with self.assertRaises(TypeError):
#             neighbour_grains = self.grain_1.get_nearest_neighbours_from_grain_list(grains_list=grains_list,
#                                                                                    max_distance=max_distance)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
