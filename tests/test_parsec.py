import numpy as np

from modules.airfoil import Airfoil
import unittest


class TestParsec(unittest.TestCase):

    def test_simple_parsec(self):
        """ Creates an airfoil from a P array. Xu, Xl automatically generated with cosine distrib. """
        a1 = Airfoil(p=np.array([0.012, 0.3, 0.08, 0, 0.3, 0.06, 0.0, 0.0211, 0.0, 0.0, 0.5]))
        # a1.plot_airfoil()

    def test_xy_to_parsec(self):
        """ Reads naca0012 coordinates and then finds best PARSEC array to fit it. """
        naca0012 = Airfoil(fpath='../data/airfoils/Coordinates_naca0012')
        parsec_naca = Airfoil(x_u=naca0012.x_u,
                              x_l=naca0012.x_l,
                              p=naca0012.xy_to_parsec())
        parsec_naca.plot_airfoil(color='g')
        # naca0012.plot_airfoil()
