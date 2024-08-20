"""
Global RF chain transfer function
"""

import os.path
import numpy as np


def read_one_column(pn_file, f_0_mhz, step_mhz):
    tfc = np.loadtxt(pn_file)
    assert tfc.ndim == 1
    freq = f_0_mhz + np.arange(tfc.shape[0]) * step_mhz
    return freq, tfc


def read_TFx_yy_fmt(n_path, n_ew, n_ns, n_z):
    """
    like /sps/grand/snonis/TFSparam/TF3_20db_NS
    """
    pn_file = os.path.join(n_path, n_ew)
    _, tf_ew = read_one_column(pn_file, 10, 1)
    pn_file = os.path.join(n_path, n_ns)
    _, tf_ns = read_one_column(pn_file, 10, 1)
    assert tf_ew.shape[0] == tf_ns.shape[0]
    pn_file = os.path.join(n_path, n_z)
    freq, tf_z = read_one_column(pn_file, 10, 1)
    assert tf_z.shape[0] == tf_ns.shape[0]
    return freq, tf_ew, tf_ns, tf_z


def read_TF2_fmt(n_path):
    """
    like /sps/grand/snonis/TFSparam/TF3_20db_NS
    """
    n_ew = "TF2_20db_EW"
    n_ns = "TF2_20db_NS"
    n_z = "TF2_20db_Z"
    return read_TFx_yy_fmt(n_path, n_ew, n_ns, n_z)


def read_TF3_fmt(n_path):
    """
    like /sps/grand/snonis/TFSparam/TF3_20db_NS
    """
    n_ew = "TF3_20db_EW"
    n_ns = "TF3_20db_NS"
    n_z = "TF3_20db_Z"
    return read_TFx_yy_fmt(n_path, n_ew, n_ns, n_z)
