"""
Created on 19 juil. 2024

@author: jcolley
"""

import numpy as np


def read_galaxy_psd_integrated(pn_file, f_0_mhz, step_mhz):
    """
    Shape expected : (226,72,3) for (freq, hour, antenna arm)
    Like file: galaxyVout2_per_Hz_gp13_25-250_MHz_new_lna_20dB.npy
    """
    gal = np.load(pn_file)
    assert gal.ndim == 3
    assert gal.shape[2] == 3
    idx2freq = f_0_mhz + np.arange(gal.shape[0]) * step_mhz
    idx2sideral_h = np.arange(gal.shape[1]) * (24.0 / gal.shape[1])
    return idx2freq, idx2sideral_h, gal


def read_galaxyVout2(n_path, n_file):
    """
    
    """
    n_file= "galaxyVout2_per_Hz_gp13_25-250_MHz_new_lna_20dB.npy"
    return read_galaxy_psd_integrated(n_path,n_file)