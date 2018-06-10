import logging.config

import os

FPATH_PROJECT = os.path.dirname(os.path.realpath(__file__))

FPATH_LOGCONFIG = os.path.join(FPATH_PROJECT, 'logging_config.ini')
logging.config.fileConfig(FPATH_LOGCONFIG)

FPATH_XFOIL = os.path.join(FPATH_PROJECT, 'modules/xfoil.exe')
FPATH_LOG = os.path.join(FPATH_PROJECT, 'log.log')
FPATH_OUT = os.path.join(FPATH_PROJECT, 'data/out')
FPATH_NACA2412 = os.path.join(FPATH_PROJECT, 'data/out/NACA2412.dat')
