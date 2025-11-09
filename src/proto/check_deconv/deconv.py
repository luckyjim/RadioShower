"""
Created on 31 mai 2025

@author: jcolley
"""
from logging import getLogger
import logging

import numpy as np
import scipy.fft as sf
import matplotlib.pyplot as plt

from rshower.basis.traces_event import Handling3dTraces
import rshower.io.rf_fmt as rfchain
from rshower.num.wiener import WienerDeconvolution
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from rshower.io.leff_fmt import get_leff_default

from proto.simu_dc2.simu_ash import DataSimuFile

PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"

G_rfc = rfchain.read_TF_numpy_fmt(PN_fmodel + "/TF_RF_Chain_DC2.1rc.npy")


logger = getLogger(__name__)


def load_event(pn_asdf="../simu_dc2/test_10.asdf", i_e=400, kind="measure"):
    df = DataSimuFile()
    df.read_asdf(pn_asdf)
    evt, ef_ref = df.get_event(i_e, kind)
    return evt, ef_ref


class DeconvGrand:
    def __init__(self):
        # array freq out in MHz
        self.freq_out = None
        self.wiener = WienerDeconvolution()
        self.evt = None
        self.ant3d = None

    def _update_tf_with_pos(self, idt_du, pos_du, pol_rad):
        self.ant3d.set_name_pos(idt_du, pos_du)
        self.leff_pol = self.ant3d.get_leff_pol(pol_rad)
        self.tf = self.leff_pol * self.rf_fft
        self.idx_ok = np.squeeze(np.argwhere(np.absolute(self.tf[0]) > 0.0001))
        logger.info(self.idx_ok.shape)
        logger.info(self.idx_ok[:10])
        logger.info(self.idx_ok[-10:])

    def set_event(self, evt, pol_rad):
        assert isinstance(evt, Handling3dTraces)
        assert self.ant3d
        assert self.rf_def
        assert pol_rad.shape[0] == evt.get_nb_trace()
        self.evt = evt
        self.pol_rad = pol_rad
        freq_hz_sampling = self.evt.f_samp_mhz[0] * 1e6
        self.freq_out = sf.rfftfreq(self.evt.get_size_trace(), 1.0 / freq_hz_sampling)
        self.freq_out /= 1e6
        logger.info(self.freq_out)
        self.ant3d.set_freq_out_mhz(self.freq_out)
        self.ant3d.set_pos_source(evt.network.xmax_pos)
        self.rf_fft = rfchain.interpol_RF(self.rf_def, self.freq_out)

    def set_rfchain(self, rf_def):
        self.rf_def = rf_def

    def set_ant3d(self, ant3d):
        assert isinstance(ant3d, DetectorUnitAntenna3Axis)
        self.ant3d = ant3d

    def set_psd_noise(self, psd_noise):
        pass

    def set_psd_sig(self, psd_sig):
        pass

    def deconv_by_division(self, i_du):
        idt_du = self.evt.idx2idt[i_du]
        pos_du = self.evt.network.du_pos[i_du]
        self._update_tf_with_pos(idt_du, pos_du, self.pol_rad[i_du])
        # self.ant3d.set_name_pos(idt_du, pos_du)
        # self.leff_pol = self.ant3d.get_leff_pol(pol_rad)
        # self.tf = self.leff_pol * self.rf_fft
        #self.idx_ok = np.squeeze(np.argwhere(np.absolute(self.tf[0]) > 0.0001))
        fft_tr = sf.rfft(self.evt.traces[i_du])
        deconv = np.zeros_like(fft_tr)
        deconv[:, self.idx_ok] = fft_tr[:, self.idx_ok] / self.tf[:, self.idx_ok]
        return [deconv]

    def deconv_wiener(self, i_du):
        pass

    def deconv_all(self, f_wiener=True):
        if f_wiener:
            pf_deconv = self.deconv_wiener
        else:
            pf_deconv = self.deconv_by_division
        res_all = []
        tr_deconv = np.empty_like(self.evt.traces)
        for i_du in range(self.evt.get_nb_trace()):
            res = pf_deconv(i_du)
            res_all.append(res)
            tr_deconv[i_du] = sf.irfft(res[0])
            if i_du == 0:
                plt.figure()
                plt.title("DECONV ")
                plt.plot(tr_deconv[i_du, 0])
                plt.plot(tr_deconv[i_du, 1])
                plt.plot(tr_deconv[i_du, 2])
                plt.grid()
                # np.clip(sf.irfft(res[0]), -100000, 100000, out=tr_deconv[i_du])
        self.evt_deconv = self.evt.copy(tr_deconv)
        self.evt_deconv.name = "deconv"
        self.evt_deconv.trace_unit = "$\mu$V/m"
        self.res_all = res_all

    def plot_deconv(self):
        self.evt.get_tmax_vmax()
        self.evt.plot_footprint_val_max()
        self.evt_deconv.get_tmax_vmax()
        self.evt_deconv.plot_footprint_val_max()

    def plot_rf(self):
        plt.figure()
        plt.title("RFchain")
        plt.plot(self.freq_out, np.absolute(self.rf_fft[0]), label="SN")
        plt.plot(self.freq_out, np.absolute(self.rf_fft[1]), label="EW")
        plt.plot(self.freq_out, np.absolute(self.rf_fft[2]), ".-", label="Up")
        plt.legend()
        plt.grid()

    def plot_leff(self):
        plt.figure()
        plt.title("Leff")
        plt.plot(self.freq_out, np.absolute(self.leff_pol[0]), label="SN")
        plt.plot(self.freq_out, np.absolute(self.leff_pol[1]), label="EW")
        plt.plot(self.freq_out, np.absolute(self.leff_pol[2]), ".-", label="Up")
        plt.legend()
        plt.grid()

    def plot_tf(self):
        plt.figure()
        plt.title("Leff*RfChain")
        plt.plot(self.freq_out, np.absolute(self.tf[0]), label="SN")
        plt.plot(self.freq_out, np.absolute(self.tf[1]), label="EW")
        plt.plot(self.freq_out, np.absolute(self.tf[2]), ".-", label="Up")
        plt.legend()
        plt.grid()


def check_deconv_nonoise(i_e=0, fact_sample=0, size_trace=0):
    """
    load data model
    deconv
    """
    evt, ef_ref = load_event(i_e=i_e, kind="sig")
    assert isinstance(evt, Handling3dTraces)
    if fact_sample != 0:
        evt.downsize_sampling(fact_sample)
    if size_trace != 0:
        evt.reduce_nb_sample(size_trace)
    recons = DeconvGrand()
    recons.set_ant3d(DetectorUnitAntenna3Axis(get_leff_default(PN_fmodel)))
    recons.set_rfchain(G_rfc)
    assert evt.get_nb_trace() == len(ef_ref["polar_angle"])
    logger.info(ef_ref["polar_angle"])
    recons.set_event(evt, ef_ref["polar_angle"] + np.deg2rad(0))
    recons.deconv_all(False)
    recons.plot_rf()
    recons.plot_leff()
    recons.plot_deconv()


def check_deconv_noise_no_weiner(i_e=0, fact_sample=0, size_trace=0):
    """
    load data model
    deconv
    """
    evt, ef_ref = load_event(i_e=i_e, kind="measure")
    assert isinstance(evt, Handling3dTraces)
    if fact_sample != 0:
        evt.downsize_sampling(fact_sample)
    if size_trace != 0:
        evt.reduce_nb_sample(size_trace)    
    recons = DeconvGrand()
    recons.set_ant3d(DetectorUnitAntenna3Axis(get_leff_default(PN_fmodel)))
    recons.set_rfchain(G_rfc)
    assert evt.get_nb_trace() == len(ef_ref["polar_angle"])
    logger.info(ef_ref["polar_angle"])
    recons.set_event(evt, ef_ref["polar_angle"] + np.deg2rad(0))
    recons.deconv_all(False)
    recons.plot_rf()
    recons.plot_leff()
    recons.plot_deconv()


TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")

#check_deconv_nonoise()
#check_deconv_nonoise(0,4,2048)
check_deconv_noise_no_weiner()
#check_deconv_noise_no_weiner(0,4,2048)
# load_event(i_e=0)
plt.show()
