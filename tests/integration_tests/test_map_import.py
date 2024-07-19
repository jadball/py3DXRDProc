import os
import unittest

import numpy as np
from py3DXRDProc.grain_map import RawGrainsMap
from py3DXRDProc.grain_volume import GrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample
from pymicro.crystal.lattice import Symmetry, Lattice


class TestFullImport(unittest.TestCase):
    def setUp(self):
        # Make a blank sample
        self.sample = Sample(name="test_sample")
        # Make a blank load step
        self.load_step = LoadStep(name="test_load_step", sample=self.sample)
        # Make a blank grain volume
        self.grain_volume = GrainVolume(name="test_grain_volume",
                                        load_step=self.load_step,
                                        index_dimensions=((-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1)),
                                        material_dimensions=((-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1)),
                                        offset_origin=np.array([0., 0., 0.]))
        # Make a blank phase
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        # horrible hack

        class Object(object):
            pass

        self.sample.pars = Object()
        self.sample.pars.phases = Object()
        self.sample.pars.phases.names = ["test_phase"]
        self.sample.pars.makemap = Object()
        self.sample.pars.makemap.minpeaks = [0]

    def test_import(self):
        self.raw_grain_map = RawGrainsMap.import_from_map(map_path=os.path.join(os.path.dirname(__file__), "../test_data/scans/test_map.map"),
                                                          phase=self.phase,
                                                          grain_volume=self.grain_volume,
                                                          errors_folder=None)
        self.grain_1, self.grain_2, self.grain_3 = self.raw_grain_map.grains
        # Assert grain properties
        # Position
        self.assertTrue(np.allclose(self.grain_1.pos, np.array([-0.1, 0.1, 0.075])))
        self.assertTrue(np.allclose(self.grain_2.pos, np.array([0.1, -0.1, 0.0])))
        self.assertTrue(np.allclose(self.grain_3.pos, np.array([0.0, 0.0, -0.075])))
        # UBI
        self.assertTrue(np.allclose(self.grain_1.UBI, np.array([[3.0, 0.0, 0.0],
                                                                [0.0, 3.0, 0.0],
                                                                [0.0, 0.0, 3.0]])))
        self.assertTrue(np.allclose(self.grain_2.UBI, np.array([[0.0,  3.0, 0.0],
                                                                [-3.0, 0.0, 0.0],
                                                                [0.0,  0.0, 3.0]])))
        self.assertTrue(np.allclose(self.grain_3.UBI, np.array([[3.0, 0.0,  0.0],
                                                                [0.0, 0.0, -3.0],
                                                                [0.0, 3.0,  0.0]])))
        # Work out the volumes
        desired_material_volume = 0.4 * 0.4 * 0.2

        mean_peak_intensity_array = np.array([100, 150, 50])

        mean_peak_intensity_sum = np.sum(mean_peak_intensity_array)
        fractional_peak_intensity_array = mean_peak_intensity_array / mean_peak_intensity_sum
        desired_grain_volumes = desired_material_volume * fractional_peak_intensity_array
        self.assertAlmostEqual(desired_grain_volumes[0], self.grain_1.volume)
        self.assertAlmostEqual(desired_grain_volumes[1], self.grain_2.volume)
        self.assertAlmostEqual(desired_grain_volumes[2], self.grain_3.volume)

        # Check GIDs
        self.assertEqual(self.grain_1.gid, 0)
        self.assertEqual(self.grain_2.gid, 1)
        self.assertEqual(self.grain_3.gid, 2)

        # Check grain maps
        self.assertEqual(self.grain_1.grain_map, self.raw_grain_map)
        self.assertEqual(self.grain_2.grain_map, self.raw_grain_map)
        self.assertEqual(self.grain_3.grain_map, self.raw_grain_map)
