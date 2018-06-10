import unittest

import matplotlib.pyplot as plt
import numpy as np

from modules.airfoil import Airfoil
from settings import FPATH_NACA2412


class TestAirfoil(unittest.TestCase):

    def test_parsec_input(self):
        """ Generate airfoil from a P array. Xu, Xl automatically generated with cosine distrib. """
        a1 = Airfoil(p=np.array([0.012, 0.3, 0.08, 0, 0.3, 0.06, 0.0, 0.0211, 0.0, 0.0, 0.5]))
        self.assertIsInstance(a1.y, np.ndarray)
        # a1.plot(plt)

    def test_xy_input(self):
        """ Generate airfoil from XY datafile"""
        naca2412 = Airfoil(fpath=FPATH_NACA2412)
        self.assertIsInstance(naca2412.y, np.ndarray)
        # naca2412.plot(plt)

    def test_naca_string_input(self):
        """ Generate airfoil from NACA4 string designation """
        naca2412 = Airfoil(naca_string='NACA2412')
        self.assertIsInstance(naca2412.y, np.ndarray)
        # naca2412.plot(plt)

    def test_xy_to_parsec(self):
        """ Reads naca2412 coordinates and then finds best PARSEC array to fit it. """
        naca2412 = Airfoil(fpath=FPATH_NACA2412)
        parsec_naca = Airfoil(x_u=naca2412.x_u,
                              x_l=naca2412.x_l,
                              p=naca2412.xy_to_parsec())
        self.assertIsInstance(parsec_naca.y, np.ndarray)
        parsec_naca.plot(plt, color='b', title='PARSEC NACA2412')

    def test_airfoil_is_equal(self):
        naca_string = Airfoil(naca_string='NACA2412')
        naca_fpath = Airfoil(fpath=FPATH_NACA2412)
        self.assertEqual(naca_string, naca_fpath)
