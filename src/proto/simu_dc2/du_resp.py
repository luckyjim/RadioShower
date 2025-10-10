"""
Created on 4 avr. 2023

@author: jcolley
"""

from logging import getLogger

import numpy as np
import scipy.fft as sf


# from galaxy_alone import galactic_noise
from grand.sim.detector.rf_chain import RFChain

from rshower.basis.traces_event import Handling3dTraces
from rshower.num.signal import get_fastest_size_rfft
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from rshower.model.galatic_ant import GalacticAntComponent
from rshower.io.leff_fmt import get_leff_default
import rshower.io.rf_fmt as rfchain

logger = getLogger(__name__)

PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"
PN_fmodel = "/sps/grand/colley/data/du_model/"

class SimuDetectorUnitResponse:
    """
    Simulation of
      * antenna response
      * electronic
      * galactic signal
    with E field input

    Processing to do:

      * Convolution in time domain : (Efield*(l_eff + noise))*IR_rf_chain

        * '*' is convolution operator
        * l_eff : effective length of antenna, ie impulsional response of antenna
        * noise: galactic noise at local sideral time
        * IR_rf_chain :  impulsional response of electronic chain

    Processing performed:

      * Calculation in Fourier space: (F_Efield.(L_eff + F_noise)).TF_rf_chain

        * in Fourier domain convolution becomes multiplication
        * '.' multiplication term by term
        * L_eff : effective length of antenna in Fourier space, ie transfer function
        * F_noise: FFT of galactic noise at local sideral time
        * TF_rf_chain : transfer function of electronic chain

      * We used a common frequency definition for all calculation stored in freqs_mhz attribute
        and computed with function get_fastest_size_rfft()

    .. note::
       * no IO, only memory processing
       * manage only one event
    """

    def __init__(self):
        """
        Constructor
        """
        # Parameters
        self.params = {
            "flag_add_leff": True,
            "flag_add_rf": True,
            "flag_noise": True,  # add galactic signal
            "fact_padding": 1.0,
            "lst": 18.0,
            "fact_noise": 1,  # to adapt noise level observed
        }
        # object contents Efield and network information
        self.o_efield = Handling3dTraces()
        # self.o_rfchain = RFChain()
        self.o_rfchain = rfchain.read_TF_numpy_fmt(PN_fmodel + "/TF_RF_Chain_DC2.1rc.npy")
        self.o_ant3d = DetectorUnitAntenna3Axis(get_leff_default(PN_fmodel))
        # object of class ShowerEvent
        self.o_shower = None
        # FFT info
        self.sig_size = 0
        #  size_with_pad ~ sig_size*fact_padding
        self.size_with_pad = 0
        # float (size_with_pad,) array of frequencies in MHz in Fourier domain
        self.freqs_out_mhz = 0
        # outputs
        self.fft_noise_gal_3d = None
        self.v_out = None
        self.fft_rf = None
        self.gal = GalacticAntComponent()
        self.gal.set_model_file(PN_fmodel + "/ASD_galaxy_ant_HFSS.npy")

    ### SETTER

    def set_data_efield(self, tr_evt):
        """

        :param tr_evt: object contents Efield and network information
        :type tr_evt: Handling3dTraces
        """
        assert isinstance(tr_evt, Handling3dTraces)
        self.o_efield = tr_evt
        self.v_out = np.zeros_like(self.o_efield.traces)
        logger.debug(self.v_out.shape)
        self.sig_size = self.o_efield.get_size_trace()
        # common frequencies for all processing in Fourier domain
        size_with_pad, self.freqs_out_mhz = get_fastest_size_rfft(
            self.sig_size, self.o_efield.f_samp_mhz[0], self.params["fact_padding"]
        )
        if size_with_pad == self.size_with_pad:
            logger.info("Skip pre-compute")
        else:
            self.size_with_pad = size_with_pad
            logger.debug(self.size_with_pad)
            logger.debug(self.sig_size)
            # precompute interpolation for all antennas
            logger.info("Precompute weight for linear interpolation of Leff in frequency")
            self.o_ant3d.set_freq_out_mhz(self.freqs_out_mhz)
            # compute total transfer function of RF chain
            # self.o_rfchain.compute_for_freqs(self.freqs_out_mhz)
            # self.fft_rf = self.o_rfchain.get_tf()
            self.fft_rf = rfchain.interpol_RF(self.o_rfchain, self.freqs_out_mhz)
            self.gal.set_lst_freq_size_out(self.params["lst"], self.freqs_out_mhz, size_with_pad)
        # FFT Efield
        nb_axis = len(tr_evt.l_axis)
        if nb_axis == 3:
            self.get_resp_ant = self.o_ant3d.get_resp_3d_efield_du
            self.fft_efield = sf.rfft(self.o_efield.traces, n=self.size_with_pad)
            assert self.fft_efield.shape[1] == self.o_efield.traces.shape[1]
        elif nb_axis == 1:
            logger.info("EField POLAR")
            self.get_resp_ant = self.o_ant3d.get_resp_1d_efield_pol
            self.fft_efield = sf.rfft(self.o_efield.traces[:,0], n=self.size_with_pad)
        else:
            raise
        assert self.fft_efield.shape[0] == self.o_efield.traces.shape[0]
        
        # lst: local sideral time, galactic noise max at 18h
        if self.params["flag_noise"]:
            logger.info("Compute galaxy noise for all traces")
            self.v_noise = np.zeros_like(self.o_efield.traces)
            self.fft_noise_gal_3d = self.gal.get_rfft_gal_ant(self.o_efield.get_nb_trace())

    def set_xmax(self, xmax_xcs):
        """

        :param xmax_xcs: position Xmax  in frame [XCore]
        :type xmax_xcs: float (3,)
        """
        self.o_ant3d.set_pos_source(xmax_xcs)

    def set_polar(self, angle_polar):
        """

        :param angle_polar: [rad] 
        :type angle_polar: float 
        """
        self.o_ant3d.interp_leff.set_angle_polar(angle_polar)
        
    ### GETTER / COMPUTER

    def compute_du_all(self):
        """
        Simulate all DU
        """
        if (
            self.params["flag_noise"] == False
            and self.params["flag_add_leff"] == True
            and self.params["flag_add_rf"] == True
        ):
            for idx in range(self.o_efield.get_nb_trace()):
                self.compute_du_idx_prod(idx)
        else:
            for idx in range(self.o_efield.get_nb_trace()):
                self.compute_du_idx(idx)

    def compute_du_idx_prod(self, idx_du):
        """Simulate one DU
        Simulation DU effect computing for DU at idx
        """
        logger.debug(f"==============>  Processing DU with id: {self.o_efield.idx2idt[idx_du]}")
        self.o_ant3d.set_name_pos(
            self.o_efield.idx2idt[idx_du], self.o_efield.network.du_pos[idx_du]
        )
        fft_3d = self.get_resp_ant(self.fft_efield[idx_du])
        fft_3d *= self.fft_rf
        self.v_out[idx_du] = sf.irfft(fft_3d)[:, : self.sig_size]

    def compute_du_idx(self, idx_du):
        """Simulate one DU
        Simulation DU effect computing for DU at idx

        Processing order:

          1. antenna responses
          2. add galactic noise
          3. RF chain effect

        :param idx_du: index of DU in array traces
        :type  idx_du: int
        """
        logger.info(f"==============>  Processing DU with id: {self.o_efield.idx2idt[idx_du]}")
        self.o_ant3d.set_name_pos(
            self.o_efield.idx2idt[idx_du], self.o_efield.network.du_pos[idx_du]
        )
        ########################
        # 1) Antenna responses
        ########################
        # Voltage open circuit
        if self.params["flag_add_leff"]:
            fft_3d = self.get_resp_ant(self.fft_efield[idx_du])
        else:
            fft_3d = self.fft_efield[idx_du].copy()
        fft_noise = self.fft_noise_gal_3d[idx_du]
        ########################
        # 2) RF chain
        ########################
        if self.params["flag_add_rf"]:
            fft_3d *= self.fft_rf
            fft_noise *= self.fft_rf
        # inverse FFT and remove zero-padding wiht vec[:, : self.sig_size]
        self.v_out[idx_du] = sf.irfft(fft_3d)[:, : self.sig_size]
        self.v_noise[idx_du] = sf.irfft(fft_noise)[:, : self.sig_size].real
