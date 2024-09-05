"""

"""

from logging import getLogger

import numpy as np
import scipy.fft as sf
from scipy import interpolate
import matplotlib.pyplot as plt

from .signal import interpol_at_new_x


logger = getLogger(__name__)


class WienerDeconvolutionWhiteNoise:
    def __init__(self, f_sample_hz=1):
        self.f_hz = f_sample_hz
        logger.info(f"f_sample_hz : {f_sample_hz}")
        self.f_ifftshift = False
        self.es_sig = None

    def set_flag_ifftshift(self, flag):
        self.f_ifftshift = flag

    def set_kernel(self, ker):
        self.ker = ker
        self.set_rfft_kernel(sf.rfft(ker))

    def set_rfft_kernel(self, rfft_ker):
        s_rfft = rfft_ker.shape[0]
        if s_rfft % 2 == 0:
            self.sig_size = 2 * (s_rfft - 1)
        else:
            self.sig_size = 2 * s_rfft - 1
        logger.debug(f"sig_size: {self.sig_size}, s_rfft: {s_rfft}")
        self.rfft_ker = rfft_ker
        self.rfft_ker_c = np.conj(self.rfft_ker)
        self.es_ker = (rfft_ker * self.rfft_ker_c).real
        self.a_freq_mhz = sf.rfftfreq(self.sig_size, 1 / self.f_hz) * 1e-6

    def set_spectrum_sig(self, es_sig):
        """
        Set energy spectrum of signal

        :param es_sig:
        :type es_sig:
        """
        self.es_sig = es_sig

    def get_spectrum_vec(self, vec):
        """
        Set energy spectrum of signal
        :param vec:
        :type vec:
        """
        rfft_m = sf.rfft(vec, n=self.sig_size)
        es_sig = (rfft_m * np.conj(rfft_m)).real / self.sig_size
        return es_sig

    def deconv_es_noise_fft_in(self, rfft_measure, es_noise):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        :param es_noise: energy spectrum
        :type es_noise: float (n_s,)
        """
        rfft_m = rfft_measure
        # coeff normalisation of se is sig_size
        if self.es_sig is None:
            es_sig = (rfft_m * np.conj(rfft_m)).real / self.sig_size
            # just remove variance from se of measure
            idx_neg = np.where(es_sig < 0)[0]
            logger.debug(f"find {idx_neg.shape[0]} freq with negative value.")
            es_sig[idx_neg] = 0
            # self.es_sig = es_sig
        else:
            es_sig = self.es_sig
        wiener = (self.rfft_ker_c * es_sig) / (self.es_ker * es_sig + es_noise)
        fft_sig = rfft_m * wiener
        sig = sf.irfft(fft_sig)
        # sig[:2] = 0
        if self.f_ifftshift:
            sig = sf.ifftshift(sig)
        self.wiener = wiener
        self.sig = sig
        self.es_sig_est = es_sig
        self.snr = es_sig / es_noise
        self.es_noise = es_noise
        return sig, fft_sig

    def deconv_es_noise(self, measure, es_noise):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        """
        rfft_m = sf.rfft(measure, n=self.sig_size)
        self.measure = measure
        return self.deconv_white_noise_fft_in(rfft_m, es_noise)

    def deconv_white_noise_fft_in(self, rfft_measure, sigma):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        :param sigma: white noise with standard deviation sigma > 0
        :type sigma: float
        """
        wh_var = sigma ** 2
        rfft_m = rfft_measure
        # coeff normalisation of se is sig_size
        if self.es_sig is None:
            es_sig = (rfft_m * np.conj(rfft_m)).real / self.sig_size
            # just remove variance from se of measure
            es_sig -= wh_var
            idx_neg = np.where(es_sig < 0)[0]
            logger.debug(f"find {idx_neg.shape[0]} freq with negative value.")
            es_sig[idx_neg] = 0
            # self.es_sig = es_sig
        else:
            es_sig = self.es_sig
        wiener = (self.rfft_ker_c * es_sig) / (self.es_ker * es_sig + wh_var)
        fft_sig = rfft_m * wiener
        sig = sf.irfft(fft_sig)
        # sig[:2] = 0
        if self.f_ifftshift:
            sig = sf.ifftshift(sig)
        self.wiener = wiener
        self.sig = sig
        self.es_sig_est = es_sig
        self.snr = es_sig / wh_var
        self.es_noise = wh_var * np.ones(rfft_m.shape[0])
        return sig, fft_sig

    def deconv_white_noise(self, measure, sigma):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        :param sigma: white noise with standard deviation sigma > 0
        :type sigma: float
        """
        rfft_m = sf.rfft(measure, n=self.sig_size)
        self.measure = measure
        return self.deconv_white_noise_fft_in(rfft_m, sigma)

    def plot_spectrum(self, loglog=True):
        freq_hz = self.a_freq_mhz
        print(self.sig_size, freq_hz.shape, 1 / self.f_hz)
        plt.figure()
        plt.title("Energy Spectrum (ES)")
        if loglog:
            my_plot = plt.loglog
        else:
            my_plot = plt.semilogy
        my_plot(freq_hz[1:], self.es_sig_est[1:], label="ES estimated signal")
        my_plot(freq_hz[1:], self.es_noise[1:], label="ES estimated noise")
        plt.grid()
        plt.legend()

    def plot_snr(self):
        freq_hz = self.a_freq_mhz
        plt.figure()
        plt.title("SNR")
        plt.semilogy(freq_hz[1:], self.snr[1:])
        plt.grid()

    def plot_measure_signal(self, title=""):
        plt.figure()
        plt.title("measure_signal" + title)
        plt.plot(self.sig, label="Wiener solution")
        plt.plot(self.measure, label="Measures")
        plt.grid()
        plt.legend()


class WienerDeconvolution:
    def __init__(self, f_sample_hz=1):
        assert isinstance(f_sample_hz, float)
        self.f_hz = f_sample_hz
        logger.info(f"f_sample_hz : {f_sample_hz}")
        self.f_ifftshift = False
        self.psd_sig = None
        self.idx_min = 0
        self.a_freq_mhz = None
        self.sig_size = None

    def set_flag_ifftshift(self, flag):
        self.f_ifftshift = flag

    def set_kernel(self, ker):
        self.ker = ker
        self.set_rfft_kernel(sf.rfft(ker))

    def set_rfft_kernel(self, rfft_ker):
        s_rfft = rfft_ker.shape[0]
        if s_rfft % 2 == 1:
            self.sig_size = 2 * (s_rfft - 1)
        else:
            self.sig_size = 2 * s_rfft - 1
        logger.debug(f"sig_size: {self.sig_size}, s_rfft: {s_rfft}")
        self.rfft_ker = rfft_ker
        self.rfft_ker_c = np.conj(self.rfft_ker)
        self.ker_pow2 = (rfft_ker * self.rfft_ker_c).real
        self.a_freq_mhz = sf.rfftfreq(self.sig_size, 1 / self.f_hz) * 1e-6
        self.idx_max = s_rfft

    def set_band(self, f_band_mhz):
        delta_f = self.a_freq_mhz[1]
        idx_min = int(f_band_mhz[0] / delta_f)
        idx_max = int(0.5 + f_band_mhz[1] / delta_f)
        self.r_freq = range(idx_min, idx_max)
        logger.info(f"Bandxitch {len(self.r_freq)} modes")

    def set_psd_noise(self, psd_noise):
        """
        Set energy spectrum of 

        :param psd_sig:
        :type psd_sig:
        """
        self.psd_noise = psd_noise

    def set_psd_sig(self, psd_sig):
        """
        Set energy spectrum of signal

        :param psd_sig:
        :type psd_sig:
        """
        self.psd_sig_est = psd_sig

    def get_interpol(self, freq_mhz, sig):
        return interpol_at_new_x(freq_mhz, sig, self.a_freq_mhz, "linear")

    def deconv_fft_measure(self, rfft_measure, psd_sig):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        :param es_noise: energy spectrum
        :type es_noise: float (n_s,)
        """
        rfft_m = rfft_measure
        # coeff normalisation of se is sig_size
        wiener = (self.rfft_ker_c * psd_sig) / (self.ker_pow2 * psd_sig + self.psd_noise)
        fft_sig = np.zeros_like(rfft_m)
        fft_sig[self.r_freq] = rfft_m[self.r_freq] * wiener[self.r_freq]
        sig = sf.irfft(fft_sig)
        if self.f_ifftshift:
            sig = sf.ifftshift(sig)
        self.wiener = wiener
        # between 0 and 1
        self.gain_weiner = psd_sig / (psd_sig + self.psd_noise)
        self.sig = sig
        self.psd_sig_est = psd_sig
        self.snr = psd_sig / self.psd_noise
        logger.debug(fft_sig.shape)
        return sig, fft_sig

    def deconv_measure(self, measure, psd_sig):
        """

        :param measure: measures from convolution operation
        :type measure: float (n_s,)
        """
        rfft_m = sf.rfft(measure, n=self.sig_size)
        self.measure = measure
        return self.deconv_fft_measure(rfft_m, psd_sig)

    def plot_psd(self, title="", loglog=False):
        freq_hz = self.a_freq_mhz
        print(self.sig_size, freq_hz.shape, 1 / self.f_hz)
        plt.figure()
        plt.title(f"Power Spectrum Density (PSD){title}")
        if loglog:
            my_plot = plt.loglog
        else:
            my_plot = plt.semilogy
        my_plot(freq_hz[1:], self.psd_sig_est[1:], label="PSD estimated signal")
        my_plot(freq_hz[1:], self.psd_noise[1:], label="PSD estimated noise")
        plt.grid()
        plt.legend()

    def plot_snr(self):
        freq_hz = self.a_freq_mhz
        plt.figure()
        plt.title("SNR")
        plt.semilogy(freq_hz[1:], self.snr[1:])
        plt.grid()

    def plot_measure_signal(self, title=""):
        plt.figure()
        plt.title("measure_signal" + title)
        plt.plot(self.sig, label="Wiener solution")
        plt.plot(self.measure, label="Measures")
        plt.grid()
        plt.legend()
    
    def plot_ker_pow2(self, title=""):
        plt.figure()
        plt.title("|Kernel|^2" + title)
        plt.plot(self.a_freq_mhz, self.ker_pow2, label="Kernel")
        plt.grid()
        plt.legend()
