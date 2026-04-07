"""
Created on 31 mai 2025

@author: jcolley


Method Polar Wiener for deconvolution with DC2 Polar dataset

ref
https://github.com/grand-mother/NUTRIG1/blob/deconv_0607/shower_radio/src/proto/recons/proto_vout_efield.py

"""

from logging import getLogger
import logging
import copy

import numpy as np
import scipy.fft as sf
import scipy.signal as ssig
import matplotlib.pyplot as plt

from rshower.basis.traces_event import Handling3dTraces, get_psd
import rshower.io.rf_fmt as rfchain
from rshower.num.wiener import WienerDeconvolution
from rshower.num.signal import interpol_at_new_x
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from rshower.io.leff_fmt import get_leff_default
from rshower.io.events.asdf_traces import AsdfReadTraces
from rshower.simu.gal_resp import GalacticRespDetectorGenerator
from rshower.model.psd_efield import AirShowerEfieldPSDmodel, modelPSD_4params

from proto.epsd_model.epsd_dc2 import EfieldModelDataset
from dask.tests.test_typing import assert_isinstance

logger = getLogger(__name__)

np.random.seed(1)


class ParametersJMC:
    def __init__(self):
        self.pn_calib = "/home/jcolley/projet/grand_wk/recons/du_model/"
        self.pn_rfchain = self.pn_calib + "TF_RF_Chain_DC2.1rc.npy"
        self.pn_leff = self.pn_calib
        self.pn_vash = "/home/jcolley/projet/lucky/data/v2/volt-ash_46-24976.asdf"
        self.pn_ef = "/home/jcolley/projet/lucky/data/v2/efield_46-24976.asdf"
        self.pn_out = "/home/jcolley/projet/lucky/data/valid/deconv"
        self.pn_asdgal = self.pn_calib + "ASD_galaxy_ant_HFSS.npy"
        self.pn_epsd = "/home/jcolley/projet/lucky/data/v2"


class ParametersCCIN2P3:
    def __init__(self):
        self.pn_tf = "/sps/grand/colley/grand_wk/recons/du_model/"
        self.pn_rfchain = self.pn_tf + "/TF_RF_Chain_DC2.1rc.npy"
        self.pn_leff = self.pn_tf
        self.pn_vash = "/sps/grand/simu/v2/dc2_polar/v2/volt-ash_46-24976.asdf"
        self.pn_ef = "/sps/grand/simu/v2/dc2_polar/v2/efield_46-24976.asdf"
        self.pn_out = "/sps/grand/colley/grand_wk/recons//valid/deconv"


def polar_angle_model(phi_rad):
    """Crude polar model ..."""
    return np.deg2rad(89.73 - 30.9 * np.sin(phi_rad))


class DeconvGrand:
    """
    Only memory data, no access file, use set_xxx
    """

    def __init__(self):
        # array freq out in MHz
        self.freq_out = None
        self.wiener = WienerDeconvolution()
        self.evt = None
        self.ant3d = None

    def _update_tf_with_pos(self, idt_du, pos_du, pol_rad):
        """

        :param idt_du:
        :param pos_du:
        :param pol_rad:
        """
        self.ant3d.set_name_pos(idt_du, pos_du)
        self.leff_pol = self.ant3d.get_leff_pol(pol_rad)
        self.tf = self.leff_pol * self.rf_fft
        self.idx_ok = np.squeeze(np.argwhere(np.absolute(self.tf[0]) > 0.001))
        # logger.info(self.idx_ok.shape)
        # logger.info(self.idx_ok[:10])
        # logger.info(self.idx_ok[-10:])

    def set_event(self, evt, pol_rad):
        """

        :param evt:  Handling3dTraces
        :param pol_rad: float[nb_trace]
        """
        assert isinstance(evt, Handling3dTraces)
        assert self.ant3d
        assert self.rf_def
        assert pol_rad.shape[0] == evt.get_nb_trace()
        self.evt = evt
        self.pol_rad = pol_rad
        freq_hz_sampling = self.evt.f_samp_mhz[0] * 1e6
        self.freq_out = sf.rfftfreq(self.evt.get_size_trace(), 1.0 / freq_hz_sampling)
        self.freq_out /= 1e6
        # logger.info(self.freq_out)
        self.ant3d.set_freq_out_mhz(self.freq_out)
        self.ant3d.set_pos_source(evt.network.xmax_pos)
        self.rf_fft = rfchain.interpol_RF(self.rf_def, self.freq_out)
        self.dist_xc = evt.d_simu["dist_xmax"]
        print(f"dist XC: {self.dist_xc} km")
        self.vmax = evt.get_max_norm()
        self.evt.set_periodogram(128)
        # define SNR by axis
        hilbert_amp = np.abs(ssig.hilbert(self.evt.traces, axis=-1))
        self.vmax = np.max(hilbert_amp, axis=-1)
        self.sigma = np.std(self.evt.traces[0, :, :200], axis=1)
        self.snr = self.vmax / self.sigma[None, :]
        print(self.snr)

    def estimate_psd_noise(self, idx):
        psd_noise = np.zeros((3, len(self.ant3d.freq_out_mhz)), dtype=np.float32)
        for axis in range(3):
            freqs, psd_noise_w = get_psd(
                self.evt.traces[idx, axis], self.evt.f_samp_mhz[0], 128, "median"
            )
            psd_noise[axis] = interpol_at_new_x(
                freqs, psd_noise_w, self.ant3d.freq_out_mhz
            )
        self.psd_noise = psd_noise

    def estimate_psd_sig(self, idx, axis):
        self.estimate_psd_noise(idx)
        freqs, psd_meas = get_psd(
            self.evt.traces[idx, axis], self.evt.f_samp_mhz[0], 128, "mean"
        )
        psd_meas = interpol_at_new_x(freqs, psd_meas, self.ant3d.freq_out_mhz)
        psd_sig = psd_meas - self.psd_noise[axis]
        psd_sig = np.where(psd_sig > 0, psd_sig, 1e-10)
        nfreqs = self.ant3d.freq_out_mhz
        # plt.figure()
        # plt.plot(nfreqs, psd_meas)
        # plt.plot(nfreqs, self.psd_noise[axis])
        # plt.plot(nfreqs, psd_sig, color='k')
        # plt.semilogy()
        return psd_sig

    def set_rfchain(self, rf_def):
        self.rf_def = rf_def

    def set_ant3d(self, ant3d):
        assert isinstance(ant3d, DetectorUnitAntenna3Axis)
        self.ant3d = ant3d

    # def set_sigma_noise(self, psd_noise, sigma):
    #     self.psd_noise = psd_noise
    #     self.sigma_noise = sigma

    def set_psd_sig_model(self, epsd):
        self.epsd = epsd
        assert isinstance(self.epsd, EfieldModelDataset)

    def deconv_by_division(self, i_du):
        # self.ant3d.set_name_pos(idt_du, pos_du)
        # self.leff_pol = self.ant3d.get_leff_pol(pol_rad)
        # self.tf = self.leff_pol * self.rf_fft
        # self.idx_ok = np.squeeze(np.argwhere(np.absolute(self.tf[0]) > 0.0001))
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
        tr_deconv = np.zeros_like(self.evt.traces)
        # self.wiener.set_band([40,200])
        for i_du in range(self.evt.get_nb_trace()):
            # update pos DU
            idt_du = self.evt.idx2idt[i_du]
            pos_du = self.evt.network.du_pos[i_du]
            self._update_tf_with_pos(idt_du, pos_du, self.pol_rad[i_du])
            # update PSD signal estimation
            axis = np.argmax(self.snr[i_du]).squeeze()
            self.wiener.set_rfft_kernel(self.tf[axis])
            self.wiener.set_band([40, 200])
            psd_sig = self.estimate_psd_sig(i_du, axis)
            psd_sig /= self.wiener.ker_pow2
            self.wiener.set_psd_sig(psd_sig)
            for axis in range(3):
                self.wiener.set_psd_noise(self.psd_noise[axis])
                # self.wiener.plot_psd(f"{idt_du}")
                self.wiener.set_rfft_kernel(self.tf[axis])
                sig, fft_sig = self.wiener.deconv_measure(
                    self.evt.traces[i_du, axis], psd_sig
                )
                # res_all.append(res)
                tr_deconv[i_du, axis] = sig

            if i_du == 0:
                plt.figure()
                plt.title("DECONV ")
                plt.plot(tr_deconv[i_du, 0])
                plt.plot(tr_deconv[i_du, 1])
                plt.plot(tr_deconv[i_du, 2])
                plt.grid()
                # np.clip(sf.irfft(res[0]), -100000, 100000, out=tr_deconv[i_du])
        self.evt_deconv = self.evt.copy(tr_deconv)
        assert isinstance(self.evt_deconv, Handling3dTraces)
        self.evt_deconv.name = "Efield deconv"
        self.evt_deconv.set_unit_axis(r"$\mu$V/m", "dir", "Efield deconv")
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


#
# MAINS
#


def estimate_gal_noise(gresp, lst=18):
    assert isinstance(gresp, GalacticRespDetectorGenerator)
    gresp.set_paramters_simu(gresp.f_samp_mhz, 1024 * 100)
    noise_gal = gresp.get_galactic_traces(1, lst)
    noise = Handling3dTraces()
    noise.init_traces(noise_gal, f_samp_mhz=gresp.f_samp_mhz, f_noise=True)
    noise.set_periodogram(1024)
    psd = noise.get_psd_trace_idx(0)
    # noise.plot_psd_trace_idx(0)
    # noise.plot_trace_idx(0)
    assert np.allclose(psd[0][0], psd[1][0])
    assert np.allclose(psd[0][0], psd[2][0])
    psd_out = np.empty((3, psd[1][1].shape[0]), dtype=np.float32)
    psd_out[0] = psd[0][1]
    psd_out[1] = psd[1][1]
    psd_out[2] = psd[2][1]
    sigma = np.std(noise.traces[0], axis=1)
    assert sigma.shape == (3,)
    print(noise.traces.shape)
    print(sigma)
    # restore initial size
    gresp.set_paramters_simu(gresp.f_samp_mhz, 1024)
    return psd_out, sigma


def load_init_deconv(pars):
    f_volt = AsdfReadTraces(pars.pn_vash)
    f_ef = AsdfReadTraces(pars.pn_ef)
    recons = DeconvGrand()
    recons.set_ant3d(DetectorUnitAntenna3Axis(get_leff_default(pars.pn_leff)))
    recons.set_rfchain(rfchain.read_TF_numpy_fmt(pars.pn_rfchain))
    gresp = GalacticRespDetectorGenerator(pars.pn_rfchain, pars.pn_asdgal)
    # print(f_volt.meta["f_samp_mhz"], f_volt.traces.shape[2])
    f_samp_mhz = f_volt.meta["f_samp_mhz"]
    gresp.set_paramters_simu(f_samp_mhz, f_volt.traces.shape[2])
    # psd, sigma = estimate_gal_noise(gresp)
    # recons.set_psd_noise(psd, sigma)
    recons.wiener.set_f_sample(f_samp_mhz)
    epsd = EfieldModelDataset()
    # 11 is efield_46-24976.asdf
    epsd.init_collect_dataset(pars.pn_epsd, "46-24976")
    epsd.partitioning_dataset(1)
    recons.set_psd_sig_model(epsd)
    return recons, f_volt, f_ef, gresp


def check_deconv_nonoise(i_e=0):
    """
    load data model
    deconv
    """
    # Load data
    pars = ParametersJMC()
    f_volt = AsdfReadTraces(pars.pn_vash)
    f_ef = AsdfReadTraces(pars.pn_ef)
    evolt = f_volt.get_event(i_e)
    eef = f_ef.get_event(i_e)
    assert isinstance(evolt, Handling3dTraces)
    # Init deconvolution
    recons = DeconvGrand()
    recons.set_ant3d(DetectorUnitAntenna3Axis(get_leff_default(pars.pn_leff)))
    recons.set_rfchain(rfchain.read_TF_numpy_fmt(pars.pn_rfchain))
    idx_beg, idx_end = f_volt.get_event_interval(i_e)
    phi_deg = f_volt.mtraces["azi"][idx_beg:idx_end]
    logger.info(np.max(f_volt.mtraces["azi"]))
    assert phi_deg.shape[0] == evolt.get_nb_trace()
    polar_angle_rad = polar_angle_model(np.deg2rad(phi_deg))
    polar_angle_ref = f_ef.d_asdf["pol_angle"][idx_beg:idx_end]
    logger.info(np.max(f_ef.d_asdf["pol_angle"]))
    plt.figure()
    plt.title("True error polar angle")
    plt.hist(np.rad2deg(polar_angle_ref - polar_angle_rad))
    plt.xlabel("Degree")
    plt.grid()
    recons.set_event(evolt, polar_angle_rad)
    recons.deconv_all(False)
    recons.plot_rf()
    recons.plot_leff()
    recons.plot_tf()
    recons.plot_deconv()
    # galr
    eef_ref = eef.copy()
    eef_ref.plot_footprint_val_max()
    eef.apply_bandpass(40, 200, False, 12)
    eef.plot_footprint_val_max()


def check_deconv_noise_no_weiner(i_e=0):
    """# 250, snr 100
    load data model
    deconv
    """
    recons, f_volt, f_ef, galr = load_init_deconv(ParametersJMC())
    evolt = f_volt.get_event(i_e)
    assert isinstance(evolt, Handling3dTraces)
    assert isinstance(galr, GalacticRespDetectorGenerator)
    eef = f_ef.get_event(i_e)
    idx_beg, idx_end = f_volt.get_event_interval(i_e)
    phi_deg = f_volt.mtraces["azi"][idx_beg:idx_end]
    polar_angle_rad = polar_angle_model(np.deg2rad(phi_deg))
    polar_angle_ref = f_ef.d_asdf["pol_angle"][idx_beg:idx_end]
    # Add noise
    evolt_c = evolt.copy()
    evolt_c.plot_footprint_val_max()
    galr.add_galactic_component(evolt, 6)
    evolt.get_tmax_vmax()
    evolt.set_periodogram(128)
    evolt.plot_footprint_val_max()
    # print(np.argmax(evolt.traces[5]))
    snr, _, _, _ = evolt.get_snr_and_noise()
    print(np.argwhere(snr > 6).squeeze())
    print(evolt.idx2idt)
    recons.set_event(evolt, polar_angle_rad)
    evolt.plot_psd_trace_idx(3)
    freqs, psd_we = get_psd(evolt.traces[3, 1], evolt.f_samp_mhz[0], 128, "median")
    plt.plot(freqs, psd_we, label="noise WE")
    plt.legend()

    recons.deconv_all(False)
    recons.plot_rf()
    recons.plot_leff()
    recons.plot_tf()
    recons.plot_deconv()
    print(snr.max())
    eef_ref = eef.copy()
    eef_ref.plot_footprint_val_max()
    eef.apply_bandpass(40, 200, False, 12)
    eef.plot_footprint_val_max()


TPL_FMT_LOGGER = (
    "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
)
logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")

# check_deconv_nonoise()
# check_deconv_nonoise(0,4,2048)
# check_deconv_nonoise(110)
# 250, snr 100
check_deconv_noise_no_weiner(130)
# check_deconv_noise_no_weiner(0,4,2048)
# load_event(i_e=0)
plt.show()
