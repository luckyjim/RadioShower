"""
Created on 6 avr. 2023

@author: jcolley
"""

import os.path
from logging import getLogger


from .zhaires_hdf5 import ZhairesSingleEventHdf5
from .zhaires_txt import ZhairesSingleEventText


logger = getLogger(__name__)


class ZhairesMaster:
    """
    select HDF5 or text format
    """

    def __init__(self, path_zhaires):
        """
        
        :param path_zhaires:
        :type path_zhaires:
        """
        l_path = path_zhaires.split("/")
        pre_path = os.path.join(path_zhaires, l_path[-1])
        nf_hdf5 = pre_path + ".hdf5"
        if os.path.exists(nf_hdf5):
            logger.info(f"Used HDF5 file {nf_hdf5}")
            self.f_zhaires = ZhairesSingleEventHdf5(nf_hdf5)
            self.f_zhaires.get_simu_info()
            return
        logger.info("Used text file ZHAireS")
        self.f_zhaires = ZhairesSingleEventText(path_zhaires)
        self.f_zhaires.read_all()

    def get_status(self):
        return self.f_zhaires.status

    def get_simu_info(self):
        return self.f_zhaires.get_simu_info()

    def get_object_3dtraces(self):
        return self.f_zhaires.get_object_3dtraces()
