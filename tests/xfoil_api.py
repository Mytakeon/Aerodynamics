from modules.airfoil import Airfoil
from modules.xfoil import XfoilAPI
import unittest
import numpy as np



class TestXfoilAPI(unittest.TestCase):


    def test_instance_1(self):
        # naca0012 = Airfoil(fpath='../data/airfoils/Coordinates_naca0012')
        # Todo: fix: should pass an Airfoil instance, not a relative path string!
        xf = XfoilAPI(airfoil='../data/airfoils/Coordinates_naca0012')
