"""
Created on 4 nov. 2025

@author: jcolley
"""

import rshower.model.galatic_ant as gant

from logging import getLogger

import numpy as np
import scipy.fft as sf
import matplotlib.pyplot as plt

from rshower.basis.traces_event import Handling3dTraces
from rshower.model.galatic_ant import GalacticAntComponent
import rshower.io.rf_fmt as rfchain


logger = getLogger(__name__)


class GalacticRespDetectorGenerator:

    def __init__(self, pn_tf_elec, pn_asd_galactic):
        self.o_rfchain = rfchain.read_TF_numpy_fmt(pn_tf_elec)
        self.gal = GalacticAntComponent()
        self.gal.set_model_file(pn_asd_galactic)

    def set_paramters_simu(self, freqs_out_mhz, size_out, nb_du):
        self.freqs_mhz = freqs_out_mhz
        self.size_out = size_out
        self.nb_du = nb_du
        self.fft_rf = rfchain.interpol_RF(self.o_rfchain, freqs_out_mhz)
        print(self.fft_rf.shape)

    def get_galactic_traces(self, lst):
        """ """
        self.gal.set_lst_freq_size_out(lst, self.freqs_mhz, self.size_out)
        fft_gal = self.gal.get_rfft_gal_ant(self.nb_du)
        print(fft_gal.shape)
        fft_gal *= self.fft_rf[None, ...]
        return sf.irfft(fft_gal)[:, : self.size_out].real

    def get_galactic_event(self, lst):
        evt = Handling3dTraces(f"galactic detector response at {lst}h LST")
        traces = self.get_galactic_traces(lst)
        evt.init_traces(traces, f_samp_mhz=2 * self.freqs_mhz[-1], f_noise=True)
        evt.set_unit_axis(r"$\mu V$", "dir", "galactic")
        return evt


def do_sigma_galactic():
    size_out = 1024
    fs_hz = 500_000_000
    freqs_mhz = sf.rfftfreq(size_out, 1 / fs_hz) * 1e-6
    nb_du = 1000
    gresp.set_paramters_simu(freqs_mhz, size_out, nb_du)
    sigma_lst = np.zeros((24, 3), dtype=np.float32)
    for lst in range(24):
        evt = gresp.get_galactic_event(lst)
        assert isinstance(evt, Handling3dTraces)
        # for ADU
        # evt.to_digit(True, np.float64)
        traces = evt.traces
        sigma_lst[lst] = np.std(traces[:, :, -100:], axis=-1).mean(axis=0)
    # to mV
    sigma_lst /= 1000
    plt.figure()
    plt.title("Sigma galactic GP300 response simulation\n1000 traces by point")
    plt.plot(sigma_lst[:, 0], label="NS", color="k")
    plt.plot(sigma_lst[:, 1], label="WE", color="y")
    plt.plot(sigma_lst[:, 2], label="Up", color="b")
    plt.grid()
    plt.xlabel("Local sideral time [h]")
    # plt.ylabel(r'${\mu V}^2$')
    # plt.ylabel(r'ADU')
    plt.ylabel(r"mV")
    plt.legend()


if __name__ == "__main__":
    # site dependancy
    PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"
    # PN_fmodel = "/sps/grand/colley/data/du_model/"
    # fix file model
    pn_tf_detector = PN_fmodel + "TF_RF_Chain_DC2.1rc.npy"
    pn_asd_galactic = PN_fmodel + "ASD_galaxy_ant_HFSS.npy"
    gresp = GalacticRespDetectorGenerator(pn_tf_detector, pn_asd_galactic)
    #
    size_out = 2048
    fs_hz = 500_000_000
    freqs_mhz = sf.rfftfreq(size_out, 1 / fs_hz) * 1e-6
    gresp.set_paramters_simu(freqs_mhz, size_out, 10)
    evt = gresp.get_galactic_event(6)
    evt.plot_trace_du(0)
    evt.plot_trace_du(9)
    evt.plot_psd_trace_du(9)
    evt = gresp.get_galactic_event(18)
    evt.plot_trace_du(0)
    evt.plot_trace_du(9)
    evt.plot_psd_trace_du(9)
    do_sigma_galactic()
    do_sigma_galactic()
    plt.show()
