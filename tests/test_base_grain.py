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
import xfab
from ImageD11.grain import grain as id11_grain
from py3DXRDProc.grain import BaseGrain


class TestWrongInits(unittest.TestCase):
    def test_wrong_pos_type(self):
        with self.assertRaises(TypeError):
            grain = BaseGrain(pos="position",
                              UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                            [1.04918116, 1.75653739, -2.01200437],
                                            [1.17321589, 1.63886394, 2.04256107]]),
                              volume=228.872582)

    def test_wrong_pos_shape(self):
        with self.assertRaises(ValueError):
            grain = BaseGrain(pos=np.array([1., 2, 3, 4]),
                              UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                            [1.04918116, 1.75653739, -2.01200437],
                                            [1.17321589, 1.63886394, 2.04256107]]),
                              volume=228.872582)

    def test_wrong_pos_element_type(self):
        with self.assertRaises(TypeError):
            grain = BaseGrain(pos=np.array([1, 2, 3]),
                              UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                            [1.04918116, 1.75653739, -2.01200437],
                                            [1.17321589, 1.63886394, 2.04256107]]),
                              volume=228.872582)

    def test_wrong_UBI_type(self):
        with self.assertRaises(TypeError):
            grain = BaseGrain(pos=np.array([1., 2, 3]),
                              UBI="UBI",
                              volume=228.872582)

    def test_wrong_UBI_shape(self):
        with self.assertRaises(ValueError):
            grain = BaseGrain(pos=np.array([1., 2, 3]),
                              UBI=np.array([2.3994061, -1.56941634, -0.118949452]),
                              volume=228.872582)

    def test_wrong_volume_type(self):
        with self.assertRaises(TypeError):
            grain = BaseGrain(pos=np.array([1., 2, 3]),
                              UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                            [1.04918116, 1.75653739, -2.01200437],
                                            [1.17321589, 1.63886394, 2.04256107]]),
                              volume="volume")

    def test_negative_volume(self):
        with self.assertRaises(ValueError):
            grain = BaseGrain(pos=np.array([1., 2, 3]),
                              UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                                            [1.04918116, 1.75653739, -2.01200437],
                                            [1.17321589, 1.63886394, 2.04256107]]),
                              volume=-100.0)


class TestPos(unittest.TestCase):
    def setUp(self):
        self.grain = BaseGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582)

    def test_get_pos(self):
        self.assertTrue(np.array_equal(self.grain.pos, np.array([1., 2, 3])))

    def test_set_pos_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.pos = np.array([4., 5, 6])


class TestUBI(unittest.TestCase):
    def setUp(self):
        self.grain = BaseGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582)

    def test_get_ubi(self):
        self.assertTrue(np.array_equal(self.grain.UBI, np.array(
            [[2.3994061, -1.56941634, -0.118949452], [1.04918116, 1.75653739, -2.01200437],
             [1.17321589, 1.63886394, 2.04256107]])))

    def test_set_ubi_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.UBI = np.array([4., 5, 6])


class TestVolume(unittest.TestCase):
    def setUp(self):
        self.grain = BaseGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582)

    def test_get_volume(self):
        self.assertTrue(self.grain.volume == 228.872582)

    def test_set_volume_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.volume = np.array([4., 5, 6])


class TestID11Properties(unittest.TestCase):
    def setUp(self):
        self.grain = BaseGrain(
            pos=np.array([1., 2, 3]),
            UBI=np.array([[2.3994061, -1.56941634, -0.118949452],
                          [1.04918116, 1.75653739, -2.01200437],
                          [1.17321589, 1.63886394, 2.04256107]]),
            volume=228.872582)
        self.ID11_grain = id11_grain(ubi=np.array([[2.3994061, -1.56941634, -0.118949452],
                                                   [1.04918116, 1.75653739, -2.01200437],
                                                   [1.17321589, 1.63886394, 2.04256107]]),
                                     translation=np.array([1., 2, 3]))

    def test_same_UB(self):
        self.assertTrue(np.allclose(self.grain.UB, self.ID11_grain.UB))

    def test_set_UB_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.UB = 4

    def test_same_B(self):
        self.assertTrue(np.allclose(self.grain.B, self.ID11_grain.B))

    def test_set_B_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.B = 4

    def test_same_U(self):
        self.assertTrue(np.allclose(self.grain.U, self.ID11_grain.U))

    def test_set_U_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.U = 4

    def test_same_rod(self):
        self.assertTrue(np.allclose(self.grain.rod, self.ID11_grain.Rod))

    def test_set_rod_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.rod = 4

    def test_same_eul(self):
        self.assertTrue(np.allclose(self.grain.eul, xfab.tools.u_to_euler(self.ID11_grain.U)))

    def test_set_eul_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.eul = 4

    def test_same_mt(self):
        self.assertTrue(np.allclose(self.grain.mt, self.ID11_grain.mt))

    def test_set_mt_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.mt = 4

    def test_same_rmt(self):
        self.assertTrue(np.allclose(self.grain.rmt, self.ID11_grain.rmt))

    def test_set_rmt_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.rmt = 4

    def test_same_unitcell(self):
        self.assertTrue(np.allclose(self.grain.unitcell, self.ID11_grain.unitcell))

    def test_set_unitcell_refused(self):
        with self.assertRaises(AttributeError):
            self.grain.unitcell = 4


class TestAttributeNameToString(unittest.TestCase):
    def setUp(self):
        self.grain = BaseGrain(
            pos=np.array([0.265285, 0.330256, -0.016893]),
            UBI=np.array([
                [2.3994061, -1.56941634, -0.118949452],
                [1.04918116, 1.75653739, -2.01200437],
                [1.17321589, 1.63886394, 2.04256107]
            ]),
            volume=228.872582)

    def test_right_attribute_array(self):
        calced_string = self.grain.attribute_name_to_string("UBI")
        desired_string = "2.39940610 -1.56941634 -0.11894945 1.04918116 1.75653739 -2.01200437 1.17321589 1.63886394 2.04256107"
        self.assertEqual(calced_string, desired_string)

    def test_right_attribute_float(self):
        calced_string = self.grain.attribute_name_to_string("volume")
        desired_string = "228.87258200"
        self.assertEqual(calced_string, desired_string)

    def test_wrong_attribute(self):
        with self.assertRaises(ValueError):
            self.grain.attribute_name_to_string("eps_snord")

    def test_wrong_attribute_type(self):
        with self.assertRaises(TypeError):
            self.grain.attribute_name_to_string(123)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)