import unittest

from modules.xfoil import XfoilAPI

import logging
log = logging.getLogger(__name__)



class TestXfoilAPI(unittest.TestCase):


    def test_instance_1(self):
        # naca0012 = Airfoil(fpath='../data/airfoils/Coordinates_naca0012')
        # Todo: fix: should pass an Airfoil instance, not a relative path string!
        log.debug('A new message here!')
        xf = XfoilAPI(airfoil='../data/airfoils/Coordinates_naca0012')
