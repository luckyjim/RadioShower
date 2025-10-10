"""
Global RF chain transfer function
"""

import os.path

import numpy as np
import matplotlib.pyplot as plt
import rshower.num.signal as rss

#print("================================= switch SN and WE")


def read_one_column(pn_file, f_0_mhz, step_mhz):
    tfc = np.loadtxt(pn_file, dtype=np.complex128)
    assert tfc.ndim == 1
    freq = f_0_mhz + np.arange(tfc.shape[0]) * step_mhz
    return freq, tfc


def read_TF_numpy_fmt(pf_tf):
    """
    return freq_MHz, tf_ns, tf_ew, tf_z

    :param pf_tf: string complete path name of file
    """
    fr_tf = np.load(pf_tf)
    return fr_tf[0].real, fr_tf[1], fr_tf[2], fr_tf[3]


def read_TFx_yy_fmt(n_path, n_ns, n_ew, n_z):
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
    return freq, tf_ns, tf_ew, tf_z


def read_TF1_fmt(n_path):
    """
    RF Chain version 1

    return freq_MHz, tf_ns, tf_ew, tf_z

    :param n_path:
    """
    return read_TF_numpy_fmt(os.path.join(n_path, "TF_RF_Chain_v1_DC2.npy"))


def read_TF2_fmt(n_path):
    """ """
    n_ew = "TF2_20db_EW"
    n_ns = "TF2_20db_NS"
    n_z = "TF2_20db_Z"
    return read_TFx_yy_fmt(n_path, n_ns, n_ew, n_z)


def read_TF3_fmt(n_path):
    """ """
    n_ew = "TF3_20dB_EW"
    n_ns = "TF3_20dB_NS"
    n_z = "TF3_20dB_Z"
    return read_TFx_yy_fmt(n_path, n_ns, n_ew, n_z)


def plot_global_rf_chain(n_path, f_read, title=""):
    freq, tf_ns, tf_ew, tf_z = f_read(n_path)
    plt.figure()
    plt.title(title)
    plt.plot(freq, np.abs(tf_ns), "k", label="SN")
    plt.plot(freq, np.abs(tf_ew), "y", label="EW")
    plt.plot(freq, np.abs(tf_z), "b", label="UP")
    plt.legend()
    plt.grid()
    plt.xlabel("MHz")
    plt.ylabel("??")


def plot_global_rf_chain_TF1(n_path, title="RF chain version TF1"):
    plot_global_rf_chain(n_path, read_TF1_fmt, title)


def plot_global_rf_chain_TF3(n_path, title="RF chain version TF3"):
    plot_global_rf_chain(n_path, read_TF3_fmt, title)


def interpol_RF(rf_fft, out_freq):
    rf_ns = rss.interpol_at_new_x(rf_fft[0], rf_fft[1], out_freq)
    rf_ew = rss.interpol_at_new_x(rf_fft[0], rf_fft[2], out_freq)
    rf_up = rss.interpol_at_new_x(rf_fft[0], rf_fft[3], out_freq)
    return np.array([rf_ns, rf_ew, rf_up])
