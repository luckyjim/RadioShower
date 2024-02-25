"""
"""

from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class AntennaLeffStorage:
    """
    Angle convention
        phi : 0 for nord, 90 for west
        theta :O for up, 90 for ground
    """

    def __init__(self):
        # index to sampling frequency of TF Leff in Mhz
        self.freq_mhz = 1
        # index to phi angle in degree
        self.phi_deg = 1
        # index to theta angle in degree
        self.theta_deg = 1
        # complex values of TF leff in phi direction
        self.leff_phi = 1
        # complex values of TF leff in theta direction
        self.leff_theta = 1
        self.name = ""

    def load(self, path_leff):
        f_leff = np.load(path_leff)
        if f_leff["version"][0] == "1.0":
            self.freq_mhz = f_leff["freq_mhz"]
            self.theta_deg = np.arange(91).astype(float)
            self.phi_deg = np.arange(361).astype(float)
            self.leff_phi = f_leff["leff_phi"]
            self.leff_theta = f_leff["leff_theta"]
        else:
            raise
