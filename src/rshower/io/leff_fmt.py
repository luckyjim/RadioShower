"""
"""
import os.path
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

logger = getLogger(__name__)


def _get_leff(path_leff, l_files):
    """Return dictionary with 3 antenna Leff"""
    path_ant = os.path.join(path_leff, l_files[0])
    leff_ew = AntennaLeffStorage()
    leff_ew.name = "EW"
    leff_ew.load(path_ant)
    path_ant = os.path.join(path_leff, l_files[1])
    leff_sn = AntennaLeffStorage()
    leff_sn.load(path_ant)
    leff_sn.name = "SN"
    path_ant = os.path.join(path_leff, l_files[2])
    leff_up = AntennaLeffStorage()
    leff_up.load(path_ant)
    leff_up.name = "UP"
    d_leff = {"sn": leff_sn, "ew": leff_ew, "up": leff_up}
    return d_leff


def get_leff_default(path_leff):
    """Return dictionary with 3 antenna Leff

    :param path_leff: path to file Leff
    :type path_leff: string
    """
    l_files = [
        "Light_GP300Antenna_EWarm_leff.npz",
        "Light_GP300Antenna_SNarm_leff.npz",
        "Light_GP300Antenna_Zarm_leff.npz",
    ]
    return _get_leff(path_leff, l_files)


def get_leff_nec(path_leff):
    """Return dictionary with 3 antenna Leff

    :param path_leff: path to file Leff
    :type path_leff: string
    """
    l_files = [
        "Light_GP300Antenna_nec_EWarm_leff.npz",
        "Light_GP300Antenna_nec_SNarm_leff.npz",
        "Light_GP300Antenna_nec_Zarm_leff.npz",
    ]
    return _get_leff(path_leff, l_files)


class AntennaLeffStorage:
    """
    Angle convention
        phi : 0 for north, 90 for west, like azimuth => absolute
        theta :O for up, 90 for ground, like  polar angle, inclinaison, zenith distance

        shape leff_phi (361, 91, 221) (phi, theta, freq)
        shape leff_theta (361, 91, 221)
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
            # print(f"shape leff_phi {self.leff_phi.shape}")
            # print(f"shape leff_theta {self.leff_theta.shape}")
            assert self.leff_phi.shape[2] == self.freq_mhz.shape[0]
            assert self.leff_theta.shape[2] == self.freq_mhz.shape[0]
        else:
            raise

    def get_ngp_leff(self, azi, d_zen):
        azi_int = int(azi)
        d_zen_int = int(d_zen)
        self.i_phi = azi_int
        self.i_theta = d_zen_int
        return self.leff_phi[azi_int, d_zen_int], self.leff_theta[azi_int, d_zen_int]

    def plot_leff_xx(self, leff, axis):
        plt.figure()
        plt.title(f"e{axis} Leff {self.name} at direction (phi={self.i_phi}, theta={self.i_theta})")
        plt.plot(self.freq_mhz, leff.real, "--", label="Leff real")
        plt.plot(self.freq_mhz, leff.imag, "--", label="Leff imag")
        plt.plot(self.freq_mhz, np.abs(leff), label="|Leff|")
        plt.grid()
        plt.xlabel("MHz")
        plt.legend()

    def plot_leff(self, azi, d_zen):
        l_phi, l_theta = self.get_ngp_leff(azi, d_zen)
        self.plot_leff_xx(l_phi, "$_{phi}$")
        self.plot_leff_xx(l_theta, "$_{theta}$")
