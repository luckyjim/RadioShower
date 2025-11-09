"""

"""
import os.path
from logging import getLogger

import numpy as np


logger = getLogger(__name__)


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
    return idx2freq, gal, idx2sideral_h


def read_grand_galaxy_vout2(n_path):
    """
    Galaxy Vout (ie with RF chain included)for GRAND Detector GP13

    """
    n_file = "galaxyVout2_per_Hz_gp13_25-250_MHz_new_lna_20dB.npy"
    logger.info(f"Read file {n_file}")
    idx2freq, gal ,idx2sideral_h = read_galaxy_psd_integrated(os.path.join(n_path, n_file), 25, 1)
    gal = np.moveaxis(gal, 0, 2)
    assert idx2freq[-1] == 250
    return idx2freq, gal, idx2sideral_h
