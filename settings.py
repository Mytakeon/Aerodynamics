import logging.config

from os import path
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
logging.config.fileConfig('../logging_config.ini')

path_xfoil = r'/home/arthur/Coding/Aerodynamics/modules/xfoil.exe'
path_log = r'/home/arthur/Coding/Aerodynamics_py3/log.log'
# path_config = r'/home/arthur/Coding/Aerodynamics/logging_config.ini'
path_output = r'/home/arthur/Coding/Aerodynamics_py3/data/out'


