"""
Created on 28 mars 2023

@author: jcolley
"""


from logging import getLogger

import numpy as np

from rshower.io.events import sradio_asdf as srfmt


logger = getLogger(__name__)

#
# Some functions to extract data from raw SRY dictionary
#


def get_simu_magnetic_vector(d_simu):
    d_inc = d_simu["geo_mag2"]["inc"]
    r_inc = np.deg2rad(d_inc)
    v_b = np.array([np.cos(r_inc), 0, -np.sin(r_inc)])
    logger.info(f"Vec B: {v_b} , inc: {d_inc:.2f} deg")
    return v_b


def get_simu_xmax(d_simu):
    xmax = 1000.0 * np.array([d_simu["x_max"]["x"], d_simu["x_max"]["y"], d_simu["x_max"]["z"]])
    return xmax


#
# Mother class for ZhairesSingleEventXXXX
#


class ZhairesSingleEventBase:
    def __init__(self, path_zhaires):
        self.path = path_zhaires
        self.dir_simu = path_zhaires.split("/")[-1]
        self.d_info = {}
        # 0 is ok Blues
        self.status = -1

    def get_dict(self):
        """
        #TODO : necessary ?
        :param self:
        :type self:
        """
        d_gen = self.d_info.copy()
        d_gen["traces"] = self.traces
        d_gen["t_start"] = self.t_start
        d_gen["ant_pos"] = self.ants
        return d_gen

    def write_asdf_file(self, p_file):
        """

        :param p_file:
        :type p_file:
        """
        srfmt.save_asdf_single_event(p_file, self.get_object_3dtraces(), self.d_info)
