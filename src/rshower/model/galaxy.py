"""

"""

from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

from rshower.num.signal import interpol_at_new_x


logger = getLogger(__name__)


class GalaxyModelVolt:
    """
    Galaxy signal in voltage Voc or Vout
    """

    def __init__(self, freq_mhz, psd_volt, sideral_h=None, v_oc=True):
        """
        :param freq_mhz:
        :param psd_volt: (sideral, axis antenna,freq)
        :param sideral_h:
        """
        assert psd_volt.shape[1] <= 3
        assert psd_volt.shape[2] == freq_mhz.shape[0]

        self.gala_voltage = psd_volt
        self.gala_freq = freq_mhz
        self.gala_sid_h = sideral_h
        if sideral_h is None:
            nb_h = psd_volt.shape[1]
            self.gala_sid_h = np.arange(nb_h) * (24.0 / nb_h)
        self.v_oc = v_oc

    def is_v_oc(self):
        # else is V out
        return self.v_oc

    def get_type_volt(self):
        if self.v_oc:
            # Antenna voltage response, aka open circuit
            return "V_{oc}"
        else:
            # V_out: Voltage with RF chain included, ie V_oc*RFchain
            return "V_{out}"
    
    def get_gal(self,f_lst):
        i_lst = int(np.rint(f_lst / self.gala_sid_h[1]))
        return self.gala_voltage[i_lst]        

    def get_volt_all_du(self, f_lst, size_out, freqs_mhz, nb_ant):
        """Return for all DU fft of galaxy signal in voltage

        This program is used as a subroutine to complete the calculation and
        expansion of galactic noise


        :param f_lst: select the galactic noise LST at the LST moment
        :type f_lst: float
        :param size_out: is the extended length
        :type size_out:int
        :param freqs_mhz: array of output frequencies
        :type freqs_mhz:float (nb freq,)
        :param nb_ant: number of antennas
        :type nb_ant:int
        :param show_flag: print figure
        :type show_flag: boll

        :return: FFT of galactic noise for all DU and components
        :rtype: float(nb du, 3, nb freq)
        """
        # lst is used as index
        i_lst = int(np.rint(f_lst / self.gala_sid_h[1]))
        v_amp_model = self.gala_voltage[i_lst]
        # SL
        nb_freq = len(freqs_mhz)
        freq_res = freqs_mhz[1] - freqs_mhz[0]
        v_amp_model = v_amp_model * np.sqrt(freq_res)
        v_amp = np.zeros((3, nb_freq), dtype=np.float32)
        v_amp[0] = interpol_at_new_x(self.gala_freq, v_amp_model[0], freqs_mhz)
        v_amp[1] = interpol_at_new_x(self.gala_freq, v_amp_model[1], freqs_mhz)
        v_amp[2] = interpol_at_new_x(self.gala_freq, v_amp_model[2], freqs_mhz)
        amp = np.random.normal(0, scale=v_amp, size=(nb_ant, 3, nb_freq))
        phase = 2 * np.pi * np.random.random(size=(nb_ant, 3, nb_freq))
        volt_fft = np.abs(amp * size_out / 2) * np.exp(1j * phase)
        # TODO: why /2 ?  must be valided
        logger.warning("Voltage realisation isn't valided !!!!")
        self.freqs_mhz = freqs_mhz
        self.v_ampli = v_amp
        self.lst = self.gala_sid_h[i_lst]
        return volt_fft

    def plot_gal_realisation(self):
        plt.figure()
        volt = self.get_type_volt()
        my_t = f"${volt}$ of galaxy at local sideral time"
        plt.title(f"{my_t} {self.lst}h\n'v_amp' field.")
        plt.semilogy(self.freqs_mhz, self.v_ampli[0], color="k", label="0 / SN axis")
        plt.semilogy(self.freqs_mhz, self.v_ampli[1], color="y", label="1 / EW axis")
        plt.semilogy(self.freqs_mhz, self.v_ampli[2], color="b", label="2 / Up axis")
        plt.xlabel("MHz")
        plt.ylabel("$\mu$V")
        plt.xlim([20, 260])
        plt.grid()
        plt.legend()

    def plot_gal_psd(self, f_lst):
        plt.figure()
        volt = self.get_type_volt()
        my_t = f"${volt}$ of galaxy at local sideral time"
        plt.title(f"{my_t} {f_lst}h")
        gal = self.get_gal(f_lst)
        freqs_mhz = self.gala_freq
        plt.semilogy(freqs_mhz, np.abs(gal[0]), color="k", label="0 / SN axis")
        plt.semilogy(freqs_mhz, np.abs(gal[1]), color="y", label="1 / EW axis")
        plt.semilogy(freqs_mhz, np.abs(gal[2]), color="b", label="2 / Up axis")
        plt.xlabel("MHz")
        plt.ylabel("$\mu$V")
        plt.xlim([0, 260])
        plt.ylim([1e-22, 1e-12])
        plt.grid()
        plt.legend()
