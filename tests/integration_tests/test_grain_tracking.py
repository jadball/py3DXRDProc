import unittest

import numpy as np
import scipy.spatial.transform
from pymicro.crystal.lattice import Symmetry, Lattice
from scipy.spatial.transform import Rotation
from xfab import tools

from py3DXRDProc.conversions import disorientation_single_numba
from py3DXRDProc.grain import RawGrain, merge_grains
from py3DXRDProc.grain_map import RawGrainsMap
from py3DXRDProc.grain_volume import GrainVolume
from py3DXRDProc.load_step import LoadStep
from py3DXRDProc.phase import Phase
from py3DXRDProc.sample import Sample


class TestFullGrainTracking(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.sample = Sample(name="test_sample_name")
        self.phase = Phase(name="test_phase", reference_unit_cell=np.array([3, 3, 3, 90, 90, 90]),
                           symmetry=Symmetry.cubic, lattice=Lattice.cubic(3))
        # Load steps
        self.load_step_1 = LoadStep(name="test_load_step_name_1", sample=self.sample)
        self.load_step_2 = LoadStep(name="test_load_step_name_2", sample=self.sample)
        # Grain volumes
        self.load_step_1_volume_1 = GrainVolume(name="letterbox_0",
                                                load_step=self.load_step_1,
                                                index_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0., 0., 0.]))
        self.load_step_1_volume_2 = GrainVolume(name="letterbox_1",
                                                load_step=self.load_step_1,
                                                index_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0., 0., 0.1]))
        self.load_step_2_volume_1 = GrainVolume(name="letterbox_0",
                                                load_step=self.load_step_2,
                                                index_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0., 0., 0.]))
        self.load_step_2_volume_2 = GrainVolume(name="letterbox_1",
                                                load_step=self.load_step_2,
                                                index_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                material_dimensions=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1)),
                                                offset_origin=np.array([0., 0., 0.1]))
        # Grain maps
        self.load_step_1_volume_1_raw_map = RawGrainsMap(grain_volume=self.load_step_1_volume_1,
                                                         phase=self.phase)
        self.load_step_1_volume_2_raw_map = RawGrainsMap(grain_volume=self.load_step_1_volume_2,
                                                         phase=self.phase)
        self.load_step_2_volume_1_raw_map = RawGrainsMap(grain_volume=self.load_step_2_volume_1,
                                                         phase=self.phase)
        self.load_step_2_volume_2_raw_map = RawGrainsMap(grain_volume=self.load_step_2_volume_2,
                                                         phase=self.phase)

        # Define 3 grains
        pos_1 = np.array([0.5, 0.5, 0.05])
        pos_2 = np.array([0.3, 0.3, 0.03])
        pos_3 = np.array([0.1, 0.1, 0.01])

        u_1 = np.identity(3)
        u_2 = Rotation.from_euler('XYZ', [90, 0, 0], degrees=True).as_matrix() @ u_1
        u_3 = Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True).as_matrix() @ u_1
        u_4 = Rotation.from_euler('XYZ', [0.01, 0, 0], degrees=True).as_matrix() @ u_1

        ubi_1 = tools.u_to_ubi(u_mat=u_1, unit_cell=self.phase.reference_unit_cell)
        ubi_2 = tools.u_to_ubi(u_mat=u_2, unit_cell=self.phase.reference_unit_cell)
        ubi_3 = tools.u_to_ubi(u_mat=u_3, unit_cell=self.phase.reference_unit_cell)
        ubi_4 = tools.u_to_ubi(u_mat=u_4, unit_cell=self.phase.reference_unit_cell)

        self.load_step_1_volume_1_raw_grain_1 = RawGrain(pos=pos_1,
                                                         UBI=ubi_1,
                                                         volume=0.8*100/(100+75+125+101),
                                                         gid=1,
                                                         grain_map=self.load_step_1_volume_1_raw_map,
                                                         mean_peak_intensity=100.0)
        self.load_step_1_volume_1_raw_grain_2 = RawGrain(pos=pos_2,
                                                         UBI=ubi_2,
                                                         volume=0.8*75/(100+75+125+101),
                                                         gid=2,
                                                         grain_map=self.load_step_1_volume_1_raw_map,
                                                         mean_peak_intensity=75.0)
        self.load_step_1_volume_1_raw_grain_3 = RawGrain(pos=pos_3,
                                                         UBI=ubi_3,
                                                         volume=0.8*125/(100+75+125+101),
                                                         gid=3,
                                                         grain_map=self.load_step_1_volume_1_raw_map,
                                                         mean_peak_intensity=125.0)
        # a duplicate of self.load_step_1_volume_1_raw_grain_1
        self.load_step_1_volume_1_raw_grain_4 = RawGrain(pos=pos_1 + np.array([0., 0., 0.0001]),
                                                         UBI=ubi_4,
                                                         volume=0.8*101/(100+75+125+101),
                                                         gid=4,
                                                         grain_map=self.load_step_1_volume_1_raw_map,
                                                         mean_peak_intensity=101.0)

        self.load_step_1_volume_2_raw_grain_1 = RawGrain(pos=pos_1 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_1,
                                                         volume=0.8*100/(100+75+125+101),
                                                         gid=1,
                                                         grain_map=self.load_step_1_volume_2_raw_map,
                                                         mean_peak_intensity=100.0)
        self.load_step_1_volume_2_raw_grain_2 = RawGrain(pos=pos_2 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_2,
                                                         volume=0.8*75/(100+75+125+101),
                                                         gid=2,
                                                         grain_map=self.load_step_1_volume_2_raw_map,
                                                         mean_peak_intensity=75.0)
        self.load_step_1_volume_2_raw_grain_3 = RawGrain(pos=pos_3 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_3,
                                                         volume=0.8*125/(100+75+125+101),
                                                         gid=3,
                                                         grain_map=self.load_step_1_volume_2_raw_map,
                                                         mean_peak_intensity=125.0)

        self.load_step_2_volume_1_raw_grain_1 = RawGrain(pos=pos_1,
                                                         UBI=ubi_1,
                                                         volume=0.8*100/(100+75+125+101),
                                                         gid=1,
                                                         grain_map=self.load_step_2_volume_1_raw_map,
                                                         mean_peak_intensity=100.0)
        self.load_step_2_volume_1_raw_grain_2 = RawGrain(pos=pos_2,
                                                         UBI=ubi_2,
                                                         volume=0.8*75/(100+75+125+101),
                                                         gid=2,
                                                         grain_map=self.load_step_2_volume_1_raw_map,
                                                         mean_peak_intensity=75.0)
        self.load_step_2_volume_1_raw_grain_3 = RawGrain(pos=pos_3,
                                                         UBI=ubi_3,
                                                         volume=0.8*125/(100+75+125+101),
                                                         gid=3,
                                                         grain_map=self.load_step_2_volume_1_raw_map,
                                                         mean_peak_intensity=125.0)

        self.load_step_2_volume_2_raw_grain_1 = RawGrain(pos=pos_1 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_1,
                                                         volume=0.8*100/(100+75+125+101),
                                                         gid=1,
                                                         grain_map=self.load_step_2_volume_2_raw_map,
                                                         mean_peak_intensity=100.0)
        self.load_step_2_volume_2_raw_grain_2 = RawGrain(pos=pos_2 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_2,
                                                         volume=0.8*75/(100+75+125+101),
                                                         gid=2,
                                                         grain_map=self.load_step_2_volume_2_raw_map,
                                                         mean_peak_intensity=75.0)
        self.load_step_2_volume_2_raw_grain_3 = RawGrain(pos=pos_3 + np.array([0, 0, -0.1]),
                                                         UBI=ubi_3,
                                                         volume=0.8*125/(100+75+125+101),
                                                         gid=3,
                                                         grain_map=self.load_step_2_volume_2_raw_map,
                                                         mean_peak_intensity=125.0)

        # Add the raw grains to the raw maps
        self.load_step_1_volume_1_raw_map.add_grains([self.load_step_1_volume_1_raw_grain_1,
                                                      self.load_step_1_volume_1_raw_grain_2,
                                                      self.load_step_1_volume_1_raw_grain_3,
                                                      self.load_step_1_volume_1_raw_grain_4])

        self.load_step_1_volume_2_raw_map.add_grains([self.load_step_1_volume_2_raw_grain_1,
                                                      self.load_step_1_volume_2_raw_grain_2,
                                                      self.load_step_1_volume_2_raw_grain_3])

        self.load_step_2_volume_1_raw_map.add_grains([self.load_step_2_volume_1_raw_grain_1,
                                                      self.load_step_2_volume_1_raw_grain_2,
                                                      self.load_step_2_volume_1_raw_grain_3])

        self.load_step_2_volume_2_raw_map.add_grains([self.load_step_2_volume_2_raw_grain_1,
                                                      self.load_step_2_volume_2_raw_grain_2,
                                                      self.load_step_2_volume_2_raw_grain_3])

        # Add the raw maps to the volumes
        self.load_step_1_volume_1.add_raw_map(self.load_step_1_volume_1_raw_map)
        self.load_step_1_volume_2.add_raw_map(self.load_step_1_volume_2_raw_map)
        self.load_step_2_volume_1.add_raw_map(self.load_step_2_volume_1_raw_map)
        self.load_step_2_volume_2.add_raw_map(self.load_step_2_volume_2_raw_map)

        # Add the volumes to the load steps
        self.load_step_1.add_grain_volumes([self.load_step_1_volume_1,
                                            self.load_step_1_volume_2])
        self.load_step_2.add_grain_volumes([self.load_step_2_volume_1,
                                            self.load_step_2_volume_2])

        # Add the load steps to the sample
        self.sample.add_load_steps([self.load_step_1,
                                    self.load_step_2])

    def test_full_process(self):
        # Make sure the sample.load_steps dict is as expected
        self.assertDictEqual(self.sample.load_steps, {"test_load_step_name_1": self.load_step_1,
                                                      "test_load_step_name_2": self.load_step_2})
        # Make sure the load_step.grain_volumes dicts are as expected
        self.assertDictEqual(self.load_step_1.grain_volumes, {"letterbox_0": self.load_step_1_volume_1,
                                                              "letterbox_1": self.load_step_1_volume_2})
        self.assertDictEqual(self.load_step_2.grain_volumes, {"letterbox_0": self.load_step_2_volume_1,
                                                              "letterbox_1": self.load_step_2_volume_2})

        # Clean everything
        self.sample.clean(dist_tol=0.05, angle_tol=0.05)
        self.load_step_1_volume_1_clean_map = self.load_step_1_volume_1.clean_maps_list[0]
        self.load_step_1_volume_2_clean_map = self.load_step_1_volume_2.clean_maps_list[0]
        self.load_step_2_volume_1_clean_map = self.load_step_2_volume_1.clean_maps_list[0]
        self.load_step_2_volume_2_clean_map = self.load_step_2_volume_2.clean_maps_list[0]

        # Each clean map should only have 3 grains
        self.assertEqual(len(self.load_step_1_volume_1_clean_map.grains), 3)
        self.assertEqual(len(self.load_step_1_volume_2_clean_map.grains), 3)
        self.assertEqual(len(self.load_step_2_volume_1_clean_map.grains), 3)
        self.assertEqual(len(self.load_step_2_volume_2_clean_map.grains), 3)

        # Check clean map metadata
        self.assertEqual(self.load_step_1_volume_1_clean_map.raw_map, self.load_step_1_volume_1_raw_map)
        self.assertEqual(self.load_step_1_volume_2_clean_map.raw_map, self.load_step_1_volume_2_raw_map)
        self.assertEqual(self.load_step_2_volume_1_clean_map.raw_map, self.load_step_2_volume_1_raw_map)
        self.assertEqual(self.load_step_2_volume_2_clean_map.raw_map, self.load_step_2_volume_2_raw_map)

        # Check for phase equality
        self.assertEqual(self.load_step_1_volume_1_clean_map.phase, self.load_step_1_volume_1_raw_map.phase)
        self.assertEqual(self.load_step_1_volume_2_clean_map.phase, self.load_step_1_volume_2_raw_map.phase)
        self.assertEqual(self.load_step_2_volume_1_clean_map.phase, self.load_step_2_volume_1_raw_map.phase)
        self.assertEqual(self.load_step_2_volume_2_clean_map.phase, self.load_step_2_volume_2_raw_map.phase)

        # Check the merged grain
        # Test against direct merge commands
        # Positions should be the same as the output from merge_grains
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[0].pos, merge_grains(
            [self.load_step_1_volume_1_raw_grain_1, self.load_step_1_volume_1_raw_grain_4]).pos))
        # Orientations should be the same as the output from merge_grains
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[0].UBI, merge_grains(
            [self.load_step_1_volume_1_raw_grain_1, self.load_step_1_volume_1_raw_grain_4]).UBI))
        # Positions should be the same as this hardcoded value
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[0].pos, np.array([0.5, 0.5, 0.05005])))
        # Volumes should be summed
        self.assertEqual(self.load_step_1_volume_1_clean_map.grains[0].volume, 0.8*(100+101)/(100+75+125+101))
        # UBI equality
        self.assertTrue(
            np.allclose(self.load_step_1_volume_1_clean_map.grains[0].UBI, self.load_step_1_volume_1_raw_grain_4.UBI))
        # Misorientation
        self.assertAlmostEqual(disorientation_single_numba(
            (self.load_step_1_volume_1_clean_map.grains[0].U, self.load_step_1_volume_1_raw_grain_1.U),
            symmetries=self.phase.symmetry.symmetry_operators()), 0.01)
        # Check metadata
        self.assertEqual(self.load_step_1_volume_1_clean_map.grain_volume,
                         self.load_step_1_volume_1_raw_map.grain_volume)
        self.assertEqual(self.load_step_1_volume_1_clean_map.load_step, self.load_step_1_volume_1_raw_map.load_step)
        self.assertEqual(self.load_step_1_volume_1_clean_map.sample, self.load_step_1_volume_1_raw_map.sample)
        self.assertCountEqual(self.load_step_1_volume_1_clean_map.grains[0].parent_grains,
                              [self.load_step_1_volume_1_raw_grain_1, self.load_step_1_volume_1_raw_grain_4])

        # Check the other grain positions
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[1].pos,
                                    self.load_step_1_volume_1_raw_map.grains[1].pos))
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[2].pos,
                                    self.load_step_1_volume_1_raw_map.grains[2].pos))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[0].pos,
                                    self.load_step_1_volume_2_raw_map.grains[0].pos))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[1].pos,
                                    self.load_step_1_volume_2_raw_map.grains[1].pos))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[2].pos,
                                    self.load_step_1_volume_2_raw_map.grains[2].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[0].pos,
                                    self.load_step_2_volume_1_raw_map.grains[0].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[1].pos,
                                    self.load_step_2_volume_1_raw_map.grains[1].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[2].pos,
                                    self.load_step_2_volume_1_raw_map.grains[2].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[0].pos,
                                    self.load_step_2_volume_2_raw_map.grains[0].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[1].pos,
                                    self.load_step_2_volume_2_raw_map.grains[1].pos))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[2].pos,
                                    self.load_step_2_volume_2_raw_map.grains[2].pos))

        # Check the other grain UBIs
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[1].UBI,
                                    self.load_step_1_volume_1_raw_map.grains[1].UBI))
        self.assertTrue(np.allclose(self.load_step_1_volume_1_clean_map.grains[2].UBI,
                                    self.load_step_1_volume_1_raw_map.grains[2].UBI))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[0].UBI,
                                    self.load_step_1_volume_2_raw_map.grains[0].UBI))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[1].UBI,
                                    self.load_step_1_volume_2_raw_map.grains[1].UBI))
        self.assertTrue(np.allclose(self.load_step_1_volume_2_clean_map.grains[2].UBI,
                                    self.load_step_1_volume_2_raw_map.grains[2].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[0].UBI,
                                    self.load_step_2_volume_1_raw_map.grains[0].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[1].UBI,
                                    self.load_step_2_volume_1_raw_map.grains[1].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_1_clean_map.grains[2].UBI,
                                    self.load_step_2_volume_1_raw_map.grains[2].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[0].UBI,
                                    self.load_step_2_volume_2_raw_map.grains[0].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[1].UBI,
                                    self.load_step_2_volume_2_raw_map.grains[1].UBI))
        self.assertTrue(np.allclose(self.load_step_2_volume_2_clean_map.grains[2].UBI,
                                    self.load_step_2_volume_2_raw_map.grains[2].UBI))

        # Check the other grain volumes
        self.assertEqual(self.load_step_1_volume_1_clean_map.grains[1].volume,
                         self.load_step_1_volume_1_raw_map.grains[1].volume)
        self.assertEqual(self.load_step_1_volume_1_clean_map.grains[2].volume,
                         self.load_step_1_volume_1_raw_map.grains[2].volume)
        self.assertEqual(self.load_step_1_volume_2_clean_map.grains[0].volume,
                         self.load_step_1_volume_2_raw_map.grains[0].volume)
        self.assertEqual(self.load_step_1_volume_2_clean_map.grains[1].volume,
                         self.load_step_1_volume_2_raw_map.grains[1].volume)
        self.assertEqual(self.load_step_1_volume_2_clean_map.grains[2].volume,
                         self.load_step_1_volume_2_raw_map.grains[2].volume)
        self.assertEqual(self.load_step_2_volume_1_clean_map.grains[0].volume,
                         self.load_step_2_volume_1_raw_map.grains[0].volume)
        self.assertEqual(self.load_step_2_volume_1_clean_map.grains[1].volume,
                         self.load_step_2_volume_1_raw_map.grains[1].volume)
        self.assertEqual(self.load_step_2_volume_1_clean_map.grains[2].volume,
                         self.load_step_2_volume_1_raw_map.grains[2].volume)
        self.assertEqual(self.load_step_2_volume_2_clean_map.grains[0].volume,
                         self.load_step_2_volume_2_raw_map.grains[0].volume)
        self.assertEqual(self.load_step_2_volume_2_clean_map.grains[1].volume,
                         self.load_step_2_volume_2_raw_map.grains[1].volume)
        self.assertEqual(self.load_step_2_volume_2_clean_map.grains[2].volume,
                         self.load_step_2_volume_2_raw_map.grains[2].volume)

        # Check the other grain parent grains
        self.assertSequenceEqual(self.load_step_1_volume_1_clean_map.grains[1].parent_grains,
                                 [self.load_step_1_volume_1_raw_map.grains[1]])
        self.assertSequenceEqual(self.load_step_1_volume_1_clean_map.grains[2].parent_grains,
                                 [self.load_step_1_volume_1_raw_map.grains[2]])
        self.assertSequenceEqual(self.load_step_1_volume_2_clean_map.grains[0].parent_grains,
                                 [self.load_step_1_volume_2_raw_map.grains[0]])
        self.assertSequenceEqual(self.load_step_1_volume_2_clean_map.grains[1].parent_grains,
                                 [self.load_step_1_volume_2_raw_map.grains[1]])
        self.assertSequenceEqual(self.load_step_1_volume_2_clean_map.grains[2].parent_grains,
                                 [self.load_step_1_volume_2_raw_map.grains[2]])
        self.assertSequenceEqual(self.load_step_2_volume_1_clean_map.grains[0].parent_grains,
                                 [self.load_step_2_volume_1_raw_map.grains[0]])
        self.assertSequenceEqual(self.load_step_2_volume_1_clean_map.grains[1].parent_grains,
                                 [self.load_step_2_volume_1_raw_map.grains[1]])
        self.assertSequenceEqual(self.load_step_2_volume_1_clean_map.grains[2].parent_grains,
                                 [self.load_step_2_volume_1_raw_map.grains[2]])
        self.assertSequenceEqual(self.load_step_2_volume_2_clean_map.grains[0].parent_grains,
                                 [self.load_step_2_volume_2_raw_map.grains[0]])
        self.assertSequenceEqual(self.load_step_2_volume_2_clean_map.grains[1].parent_grains,
                                 [self.load_step_2_volume_2_raw_map.grains[1]])
        self.assertSequenceEqual(self.load_step_2_volume_2_clean_map.grains[2].parent_grains,
                                 [self.load_step_2_volume_2_raw_map.grains[2]])

        # Check the load_step.maps dicts have updated
        self.assertDictEqual(self.load_step_1_volume_1.maps,
                             {
                                 'test_sample_name:test_load_step_name_1:letterbox_0:test_phase:raw': self.load_step_1_volume_1_raw_map,
                                 'test_sample_name:test_load_step_name_1:letterbox_0:test_phase:cleaned':
                                     self.load_step_1_volume_1_clean_map})
        self.assertDictEqual(self.load_step_1_volume_2.maps,
                             {
                                 'test_sample_name:test_load_step_name_1:letterbox_1:test_phase:raw': self.load_step_1_volume_2_raw_map,
                                 'test_sample_name:test_load_step_name_1:letterbox_1:test_phase:cleaned':
                                     self.load_step_1_volume_2_clean_map})
        self.assertDictEqual(self.load_step_2_volume_1.maps,
                             {
                                 'test_sample_name:test_load_step_name_2:letterbox_0:test_phase:raw': self.load_step_2_volume_1_raw_map,
                                 'test_sample_name:test_load_step_name_2:letterbox_0:test_phase:cleaned':
                                     self.load_step_2_volume_1_clean_map})
        self.assertDictEqual(self.load_step_2_volume_2.maps,
                             {
                                 'test_sample_name:test_load_step_name_2:letterbox_1:test_phase:raw': self.load_step_2_volume_2_raw_map,
                                 'test_sample_name:test_load_step_name_2:letterbox_1:test_phase:cleaned':
                                     self.load_step_2_volume_2_clean_map})

        # Stitch everything
        self.sample.stitch(dist_tol_xy=0.05, dist_tol_z=0.05, angle_tol=0.05)

        self.load_step_1_stitched_volume = self.load_step_1.stitched_grain_volumes_list[0]
        self.load_step_2_stitched_volume = self.load_step_2.stitched_grain_volumes_list[0]

        # Stitched volumes should have 3 grains each
        self.assertEqual(len(self.load_step_1_stitched_volume.all_grains), 3)
        self.assertEqual(len(self.load_step_2_stitched_volume.all_grains), 3)

        # Check load_step.stitched_grain_volumes dicts are as expected
        self.assertDictEqual(self.load_step_1.stitched_grain_volumes,
                             {"test_load_step_name_1_stitched": self.load_step_1_stitched_volume})
        self.assertDictEqual(self.load_step_2.stitched_grain_volumes,
                             {"test_load_step_name_2_stitched": self.load_step_2_stitched_volume})

        # Check load_step.all_grain_volumes dicts are as expected
        self.assertDictEqual(self.load_step_1.all_grain_volumes,
                             {"letterbox_0": self.load_step_1_volume_1,
                              "letterbox_1": self.load_step_1_volume_2,
                              "test_load_step_name_1_stitched": self.load_step_1_stitched_volume})
        self.assertDictEqual(self.load_step_2.all_grain_volumes,
                             {"letterbox_0": self.load_step_2_volume_1,
                              "letterbox_1": self.load_step_2_volume_2,
                              "test_load_step_name_2_stitched": self.load_step_2_stitched_volume})

        # Check phases are unchanged
        self.assertEqual(self.load_step_1_stitched_volume.maps_list[0].phase, self.load_step_1_volume_1_raw_map.phase)
        self.assertEqual(self.load_step_2_stitched_volume.maps_list[0].phase, self.load_step_2_volume_1_raw_map.phase)

        # Check parent volumes from stitched volumes
        self.assertCountEqual(self.load_step_1_stitched_volume.all_contrib_volumes,
                              [self.load_step_1_volume_1,
                               self.load_step_1_volume_2])
        self.assertCountEqual(self.load_step_2_stitched_volume.all_contrib_volumes,
                              [self.load_step_2_volume_1,
                               self.load_step_2_volume_2])

        # Check parent maps from stitched maps
        self.assertCountEqual(self.load_step_1_stitched_volume.maps_list[0].all_contrib_maps,
                              [self.load_step_1_volume_1_clean_map,
                               self.load_step_1_volume_2_clean_map])
        self.assertCountEqual(self.load_step_2_stitched_volume.maps_list[0].all_contrib_maps,
                              [self.load_step_2_volume_1_clean_map,
                               self.load_step_2_volume_2_clean_map])

        # check individual grains
        # Check grain parents
        self.assertCountEqual(self.load_step_1_stitched_volume.maps_list[0].grains[0].parent_clean_grains,
                              [self.load_step_1_volume_2_clean_map.grains[0],
                               self.load_step_1_volume_1_clean_map.grains[0]])
        self.assertCountEqual(self.load_step_1_stitched_volume.maps_list[0].grains[1].parent_clean_grains,
                              [self.load_step_1_volume_2_clean_map.grains[1],
                               self.load_step_1_volume_1_clean_map.grains[1]])
        self.assertCountEqual(self.load_step_1_stitched_volume.maps_list[0].grains[2].parent_clean_grains,
                              [self.load_step_1_volume_2_clean_map.grains[2],
                               self.load_step_1_volume_1_clean_map.grains[2]])
        self.assertCountEqual(self.load_step_2_stitched_volume.maps_list[0].grains[0].parent_clean_grains,
                              [self.load_step_2_volume_2_clean_map.grains[0],
                               self.load_step_2_volume_1_clean_map.grains[0]])
        self.assertCountEqual(self.load_step_2_stitched_volume.maps_list[0].grains[1].parent_clean_grains,
                              [self.load_step_2_volume_2_clean_map.grains[1],
                               self.load_step_2_volume_1_clean_map.grains[1]])
        self.assertCountEqual(self.load_step_2_stitched_volume.maps_list[0].grains[2].parent_clean_grains,
                              [self.load_step_2_volume_2_clean_map.grains[2],
                               self.load_step_2_volume_1_clean_map.grains[2]])

        # Check grain positions
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[0].pos_offset,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[0],
                                                  self.load_step_1_volume_2_clean_map.grains[0]]).pos_offset))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[1].pos_offset,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[1],
                                                  self.load_step_1_volume_2_clean_map.grains[1]]).pos_offset))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[2].pos_offset,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[2],
                                                  self.load_step_1_volume_2_clean_map.grains[2]]).pos_offset))

        # Check grain UBIs
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[0].UBI,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[0],
                                                  self.load_step_1_volume_2_clean_map.grains[0]]).UBI))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[1].UBI,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[1],
                                                  self.load_step_1_volume_2_clean_map.grains[1]]).UBI))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[2].UBI,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[2],
                                                  self.load_step_1_volume_2_clean_map.grains[2]]).UBI))

        # Check grain volumes
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[0].volume,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[0],
                                                  self.load_step_1_volume_2_clean_map.grains[0]]).volume))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[1].volume,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[1],
                                                  self.load_step_1_volume_2_clean_map.grains[1]]).volume))
        self.assertTrue(np.allclose(self.load_step_1_stitched_volume.maps_list[0].grains[2].volume,
                                    merge_grains([self.load_step_1_volume_1_clean_map.grains[2],
                                                  self.load_step_1_volume_2_clean_map.grains[2]]).volume))

        # Track everything
        self.sample.track(dist_tol=0.05, angle_tol=0.05)

        # Should be 3 grains tracked
        self.assertEqual(len(self.sample.tracked_grain_volume.all_grains), 3)
        # All grains should be fully tracked
        self.assertEqual(len(self.sample.tracked_grain_volume.maps_list[0].fully_tracked_grains), 3)
        # Check tracked volume parent volumes
        self.assertCountEqual(self.sample.tracked_grain_volume.all_contrib_volumes,
                              [self.load_step_1_stitched_volume,
                               self.load_step_2_stitched_volume])
        # Check tracked grain map parent grain maps
        self.assertCountEqual(self.sample.tracked_grain_volume.maps_list[0].stitched_maps_list,
                              [self.load_step_1_stitched_volume.maps_list[0],
                               self.load_step_2_stitched_volume.maps_list[0]])

        # Check tracked_grain_volume.maps dict is as expected
        self.assertDictEqual(self.sample.tracked_grain_volume.maps,
                             {"test_sample_name:test_phase:tracked": self.sample.tracked_grain_volume.maps_list[0]})

        # Check phases are unchanged
        self.assertEqual(self.sample.tracked_grain_volume.maps_list[0].phase, self.phase)

        # Check parent grains are correctly assigned
        self.assertCountEqual(self.sample.tracked_grain_volume.maps_list[0].grains[0].parent_stitch_grains_list,
                              [self.load_step_1_stitched_volume.maps_list[0].grains[0],
                               self.load_step_2_stitched_volume.maps_list[0].grains[0]])
        self.assertCountEqual(self.sample.tracked_grain_volume.maps_list[0].grains[1].parent_stitch_grains_list,
                              [self.load_step_1_stitched_volume.maps_list[0].grains[1],
                               self.load_step_2_stitched_volume.maps_list[0].grains[1]])
        self.assertCountEqual(self.sample.tracked_grain_volume.maps_list[0].grains[2].parent_stitch_grains_list,
                              [self.load_step_1_stitched_volume.maps_list[0].grains[2],
                               self.load_step_2_stitched_volume.maps_list[0].grains[2]])

        # Check tracked grain positions
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[0].pos_offset,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[0],
                          self.load_step_2_stitched_volume.maps_list[0].grains[0]]
                         ).pos_offset))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[1].pos_offset,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[1],
                          self.load_step_2_stitched_volume.maps_list[0].grains[1]]
                         ).pos_offset))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[2].pos_offset,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[2],
                          self.load_step_2_stitched_volume.maps_list[0].grains[2]]
                         ).pos_offset))

        # Check tracked grain UBIs
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[0].UBI,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[0],
                          self.load_step_2_stitched_volume.maps_list[0].grains[0]]
                         ).UBI))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[1].UBI,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[1],
                          self.load_step_2_stitched_volume.maps_list[0].grains[1]]
                         ).UBI))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[2].UBI,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[2],
                          self.load_step_2_stitched_volume.maps_list[0].grains[2]]
                         ).UBI))

        # Check tracked grain volumes
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[0].volume,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[0],
                          self.load_step_2_stitched_volume.maps_list[0].grains[0]]
                         ).volume))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[1].volume,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[1],
                          self.load_step_2_stitched_volume.maps_list[0].grains[1]]
                         ).volume))
        self.assertTrue(np.allclose(
            self.sample.tracked_grain_volume.maps_list[0].grains[2].volume,
            merge_grains([self.load_step_1_stitched_volume.maps_list[0].grains[2],
                          self.load_step_2_stitched_volume.maps_list[0].grains[2]]
                         ).volume))
