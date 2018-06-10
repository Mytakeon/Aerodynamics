import os

import numpy as np

from modules.airfoil import Airfoil
import unittest
import contextlib
from settings import FPATH_NACA2412


class TestAirfoil(unittest.TestCase):


    def test_parsec_input(self):
        """ Generate airfoil from a P array. Xu, Xl automatically generated with cosine distrib. """
        a1 = Airfoil(p=np.array([0.012, 0.3, 0.08, 0, 0.3, 0.06, 0.0, 0.0211, 0.0, 0.0, 0.5]))
        self.assertIsInstance(a1.y, np.ndarray)

    def test_xy_input(self):
        """ Generate airfoil from XY datafile"""
        naca2412 = Airfoil(fpath=FPATH_NACA2412)
        self.assertIsInstance(naca2412.y, np.ndarray)

    def test_naca_string_input(self):
        """ Generate airfoil from NACA4 string designation """
        naca2412 = Airfoil(naca_string='NACA2412')
        self.assertIsInstance(naca2412.y, np.ndarray)

    def test_write_file(self):
        naca2412 = Airfoil(naca_string='NACA2412')
        naca2412.write_xy(name='for_test')



    def test_xy_to_parsec(self):
        """ Reads naca0012 coordinates and then finds best PARSEC array to fit it. """
        naca0012 = Airfoil(fpath=FPATH_NACA2412)
        parsec_naca = Airfoil(x_u=naca0012.x_u,
                              x_l=naca0012.x_l,
                              p=naca0012.xy_to_parsec())
        self.assertIsInstance(parsec_naca.y, np.ndarray)

    def test_airfoil_is_equal(self):
        naca_string = Airfoil(naca_string='NACA2412')
        naca_fpath = Airfoil(fpath=FPATH_NACA2412)
        # self.assertEqual(naca_string, naca_fpath)
