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
    """Return the galactic signal response of detector defined by model given at init object.

    also named galactic noise
    """

    def __init__(self, pn_tf_elec, pn_asd_galactic):
        self.o_rfchain = rfchain.read_TF_numpy_fmt(pn_tf_elec)
        self.gal = GalacticAntComponent()
        self.gal.set_model_file(pn_asd_galactic)
        self.calib = 1.0
        self.lst = 18

    def set_calib_factor(self, calib):
        self.calib = calib

    def set_paramters_simu(self, f_samp_mhz, size_out):
        """
        :param f_samp_mhz: float
        :param size_out:
        """
        dt_s = 1 / (f_samp_mhz * 1e6)
        self.f_samp_mhz = f_samp_mhz
        self.freqs_mhz = sf.rfftfreq(size_out, dt_s) * 1e-6
        self.size_out = size_out
        self.fft_rf = rfchain.interpol_RF(self.o_rfchain, self.freqs_mhz)

    def get_galactic_traces(self, nb_du, lst=18, size_out=None):
        """Return numpy array (nb_du,3,size_out) with galactic noise"""
        if not size_out:
            size_out = self.size_out
        self.gal.set_lst_freq_size_out(lst, self.freqs_mhz, size_out)
        fft_gal = self.gal.get_rfft_gal_ant(nb_du)
        fft_gal *= self.fft_rf[None, ...]
        gal_resp = self.calib * sf.irfft(fft_gal).real
        return gal_resp

    def add_galactic_component(self, event, lst=18):
        """add to event Handling3dTraces object the galactic noise

        :param event:
        :param lst:
        """
        assert isinstance(event, Handling3dTraces)
        gal_resp = self.get_galactic_traces(event.get_nb_trace(), lst, event.get_size_trace())
        event.traces += gal_resp

    def get_galactic_event(self, nb_du, lst=18):
        """Return Handling3dTraces object with galactic noise

        :param nb_du:
        :param lst:
        """
        evt = Handling3dTraces(f"galactic detector response at {lst}h LST")
        traces = self.get_galactic_traces(nb_du, lst)
        evt.init_traces(traces, f_samp_mhz=2 * self.freqs_mhz[-1], f_noise=True)
        evt.set_unit_axis(r"$\mu V$", "dir", "galactic noise")
        return evt


def do_sigma_galactic():
    size_out = 1024
    fs_mhz = 500
    nb_du = 1000
    gresp.set_paramters_simu(fs_mhz, size_out)
    sigma_lst = np.zeros((24, 3), dtype=np.float32)
    for lst in range(24):
        evt = gresp.get_galactic_event(nb_du, lst)
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
    fs_mhz = 500
    gresp.set_paramters_simu(fs_mhz, size_out)
    evt = gresp.get_galactic_event(11, 6)
    print(evt.traces.shape)
    evt.plot_trace_idx(0)
    evt.plot_trace_idx(9)
    evt.plot_psd_trace_du(9)
    evt = gresp.get_galactic_event(10, 18)
    evt.plot_trace_du(0)
    evt.plot_trace_du(9)
    evt.plot_psd_trace_du(9)
    do_sigma_galactic()
    do_sigma_galactic()
    plt.show()
