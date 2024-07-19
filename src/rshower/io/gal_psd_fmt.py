"""
Created on 19 juil. 2024

@author: jcolley
"""

import numpy as np


def read_galaxy_psd_integrated(pn_file, f_0_mhz=25.0, step_mhz=1.0):
    gal = np.load(pn_file)
    assert gal.ndim == 3
    assert gal.shape[2] == 3
    freq = f_0_mhz + np.arange(gal.shape[0]) * step_mhz
    sideral_h = np.arange(gal.shape[1]) * (24.0 / gal.shape[1])
    return freq, sideral_h, gal
