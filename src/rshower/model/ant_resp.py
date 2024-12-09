"""
Created on 4 avr. 2023

@author: jcolley

Hypothesis: small network  (20-30km ) so => [XCS]~[DU] for vector/direction

see rshower.basis.frame for frame definition [XXX]
"""

import os.path
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np

import rshower.basis.coord as coord

# from rshower.io.leff_fmt import AntennaLeffStorage
# from rshower import get_path_model_du

logger = getLogger(__name__)


# def patch_leff(fft_leff_hc, f_hz):
#     """
#     ringing and causality
#
#     :param fft_leff_hc:
#     :type fft_leff_hc:
#     :param f_hz:
#     :type f_hz:
#     """
#     print(f_hz)
#     coeff_b, coeff_a = butter(6, [60 * 1e6, 220 * 1e6], btype="bandpass", fs=f_hz)
#     size_fc = 2 * (fft_leff_hc.shape[0] - 1)
#     print(fft_leff_hc.shape, size_fc)
#     w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=fft_leff_hc.shape[0], include_nyquist=True)
#     if False:
#         print("w : ", w, w.shape)
#         size_h = w.shape[0]
#         h_hc = h[: size_h // 2 + 1]
#         fft_leff_hc_cor = fft_leff_hc * h
#     else:
#         # add causality
#         fc_leff = np.concatenate((fft_leff_hc, np.flip(np.conj(fft_leff_hc[1:-1]))))
#         fc_leff.imag = -hilbert(np.real(fc_leff)).imag
#         size_l = fc_leff.shape[0]
#         fft_leff_hc_cor = fc_leff[: size_l // 2 + 1]
#     if True:
#         plt.figure()
#         plt.title("Leff correction")
#         plt.plot(w * 1e-6, abs(fft_leff_hc), label="leff")
#         plt.plot(w * 1e-6, abs(fft_leff_hc_cor), "-.", label="leff cor")
#         plt.plot(w * 1e-6, abs(h), label="Butterworth")
#         plt.grid()
#         plt.legend()
#     return fft_leff_hc_cor


class PreComputeInterpolFreq:
    """
    Precompute linear interpolation of frequency of Leff
    """

    def __init__(self):
        # index of freq in first in band 30-250MHz
        self.idx_first = None
        # index of freq in last plus one in band 30-250MHz
        self.idx_lastp1 = None
        # array of index where f_out are in f_in
        self.idx_itp = None
        # array of coefficient inf
        self.c_inf = None
        # array of coefficient sup
        # c_inf + c_sup = 1
        self.c_sup = None
        self.size_out = 0

    def get_idx_range(self, r_mhz=[55, 185]):
        d_freq_out = self.freq_out_mhz[1]
        idx_first = int(r_mhz[0] / d_freq_out) + 1
        # index of freq in last plus one, + 1 to have first out band
        idx_lastp1 = int(r_mhz[1] / d_freq_out) + 1
        return range(idx_first, idx_lastp1)

    def init_linear_interpol(self, freq_in_mhz, freq_out_mhz):
        """
        Precompute coefficient of linear interpolation for freq_out_mhz with reference defined at freq_in_mhz

        :param freq_in_mhz: regular array of frequency where function is defined
        :param freq_out_mhz: regular array frequency where we want interpol
        """
        # TODO: debbug when freq_in_mhz, freq_out_mhz are equal "index 221 is out of bounds "
        # logger.debug(f"freq_in_mhz: {freq_in_mhz}")
        # logger.debug(f"freq_out_mhz: {freq_out_mhz}")
        self.freq_out_mhz = freq_out_mhz
        self.size_out = freq_out_mhz.shape[0]
        # freq_out_mhz must start be 0 and first element is the delta frequency
        assert freq_out_mhz[0] == 0.0
        d_freq_out = freq_out_mhz[1]
        # index of freq in first in band, + 1 to have first in band
        idx_first = int(freq_in_mhz[0] / d_freq_out) + 1
        # index of freq in last plus one, + 1 to have first out band
        idx_lastp1 = int(freq_in_mhz[-1] / d_freq_out) + 1
        self.idx_first = idx_first
        self.idx_lastp1 = idx_lastp1
        d_freq_in = freq_in_mhz[1] - freq_in_mhz[0]
        freq_in_band = freq_out_mhz[idx_first:idx_lastp1]
        self.idx_itp = np.trunc((freq_in_band - freq_in_mhz[0]) / d_freq_in).astype(int)
        # logger.debug(f"freq_in_band freq_in_mhz[0]  d_freq_in")
        # logger.debug(f"{freq_in_band} {freq_in_mhz[0]} { d_freq_in}")
        # logger.debug(f"{self.idx_itp}")
        # define coefficient of linear interpolation
        self.c_sup = (freq_in_band - freq_in_mhz[self.idx_itp]) / d_freq_in
        if self.idx_itp[-1] + 1 == freq_in_mhz.shape[0]:
            # https://github.com/grand-mother/collaboration-issues/issues/30
            logger.info(f" ** Specfic processing when f_in = k * f_out else IndexError **")
            self.idx_itp[-1] -= 1
            # in this case last c_sup must be zero
            # check it !
            assert np.allclose(self.c_sup[-1], 0)
        self.c_inf = 1 - self.c_sup
        self.range_itp = range(self.idx_first, self.idx_lastp1)

    def get_linear_interpol(self, a_val):
        """
        Return f(freq_out_mhz) by linear interpolation of f defined by
        f(freq_in_mhz) = a_val

        :param a_val: defined value of function at freq_in_mhz
        """
        a_itp = self.c_inf * a_val[self.idx_itp] + self.c_sup * a_val[self.idx_itp + 1]
        return a_itp


class LengthEffectiveInterpolation:
    """
    From AntennaProcessing class of https://github.com/grand-mother/grand
    """

    def __init__(self):
        self.o_pre = PreComputeInterpolFreq()

    def _update_idx_interpol_sph(self):
        # logger.debug(f"New direction {self.dir_src_deg}")
        # delta theta in degree
        phi_efield = self.dir_src_deg[0]
        theta_efield = self.dir_src_deg[1]
        dtheta = self.theta_deg[1] - self.theta_deg[0]
        # theta_efield between index it0 and it1 in theta antenna response representation
        rt1 = (theta_efield - self.theta_deg[0]) / dtheta
        # prevent > 360 deg or >180 deg ?
        it0 = int(np.floor(rt1) % self.theta_deg.size)
        it1 = it0 + 1
        if it1 == self.theta_deg.size:  # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= np.floor(rt1)
        rt0 = 1 - rt1
        # phi_efield between index ip0 and ip1 in phi antenna response representation
        dphi = self.phi_deg[1] - self.phi_deg[0]  # deg
        rp1 = (phi_efield - self.phi_deg[0]) / dphi
        ip0 = int(np.floor(rp1) % self.phi_deg.size)
        ip1 = ip0 + 1
        if ip1 == self.phi_deg.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        self.weight = [rt0, rt1, rp0, rp1]
        self.idx_i = [it0, it1, ip0, ip1]
        logger.debug(self.idx_i)
        logger.debug(self.weight)
        return rt0, rt1, rp0, rp1, it0, it1, ip0, ip1
    
    #
    # SETTER
    #
    def set_sampling_angle(self, s_theta, s_phi):
        self.theta_deg = s_theta
        self.phi_deg = s_phi

    def set_dir_source(self, sph_du):
        self.dir_src_deg = np.rad2deg(sph_du)
        self.dir_src_rad = sph_du
        self._update_idx_interpol_sph()

    def set_angle_polar(self, a_pol_tan):
        """

        :param a_pol_tan: RAD
        """
        self.angle_pol = a_pol_tan
        self.cos_pol = np.cos(self.angle_pol)
        self.sin_pol = np.sin(self.angle_pol)

    #
    # Leff
    #
    def get_fft_leff_du(self, leff):
        l_tan = self.get_fft_leff_tan(leff)
        l_t, l_p = l_tan[0], l_tan[1]
        p_rad = self.dir_src_rad[0]
        t_rad = self.dir_src_rad[1]
        c_t, s_t = np.cos(t_rad), np.sin(t_rad)
        c_p, s_p = np.cos(p_rad), np.sin(p_rad)
        l_x = l_t * c_t * c_p - s_p * l_p
        l_y = l_t * c_t * s_p + c_p * l_p
        l_z = -s_t * l_t
        self.l_x = l_x
        self.l_y = l_y
        self.l_z = l_z
        return np.array([l_x, l_y, l_z])

    def get_fft_leff_tan(self, leff_tp):
        self.leff = leff_tp
        rt0, rt1, rp0, rp1 = self.weight
        it0, it1, ip0, ip1 = self.idx_i
        leff = leff_tp.leff_theta
        leff_itp_t = (
            rp0 * rt0 * leff[ip0, it0, :]
            + rp1 * rt0 * leff[ip1, it0, :]
            + rp0 * rt1 * leff[ip0, it1, :]
            + rp1 * rt1 * leff[ip1, it1, :]
        )
        leff = leff_tp.leff_phi
        leff_itp_p = (
            rp0 * rt0 * leff[ip0, it0, :]
            + rp1 * rt0 * leff[ip1, it0, :]
            + rp0 * rt1 * leff[ip0, it1, :]
            + rp1 * rt1 * leff[ip1, it1, :]
        )
        leff_itp_sph = np.array([leff_itp_t, leff_itp_p])
        # leff_itp_sph = np.array([leff_tp.leff_theta[ip0, it0], leff_tp.leff_phi[ip0, it0]])
        pre = self.o_pre
        leff_itp = (
            pre.c_inf * leff_itp_sph[:, pre.idx_itp] + pre.c_sup * leff_itp_sph[:, pre.idx_itp + 1]
        )

        # now add zeros outside leff frequency band and unpack leff theta , phi
        l_t = np.zeros(self.o_pre.size_out, dtype=np.complex64)
        l_t[pre.idx_first : pre.idx_lastp1] = leff_itp[0]
        l_p = np.zeros(self.o_pre.size_out, dtype=np.complex64)
        l_p[pre.idx_first : pre.idx_lastp1] = leff_itp[1]
        self.l_phi = l_p
        self.l_theta = l_t
        return np.array([l_t, l_p])

    def get_fft_leff_pol(self, leff):
        # logger.debug(f"{self.dir_src_deg} {np.rad2deg(self.angle_pol)}")
        l_tan = self.get_fft_leff_tan(leff)
        # TAN order is (e_theta, e_phi, e_normal_out)
        return self.cos_pol * l_tan[0] + self.sin_pol * l_tan[1]

    #
    # PLOT
    #
    def plot_leff_tan(self):
        plt.figure()
        plt.title(
            f"Interpolated Leff {self.leff.name} at (phi={self.dir_src_deg[0]:.1f}, theta={self.dir_src_deg[1]:.1f})"
        )
        plt.plot(self.o_pre.freq_out_mhz, self.l_phi.real, ".-.", label="Leff phi real")
        plt.plot(self.o_pre.freq_out_mhz, self.l_phi.imag, label="Leff phi imag")
        plt.plot(self.o_pre.freq_out_mhz, self.l_theta.real, ".-.", label="Leff theta real")
        plt.plot(self.o_pre.freq_out_mhz, self.l_theta.imag, label="Leff theta imag")
        idx_phi = int(self.dir_src_deg[0])
        idx_theta = int(self.dir_src_deg[1])
        plt.plot(
            self.leff.freq_mhz,
            self.leff.leff_phi.real[idx_phi, idx_theta],
            "*",
            label=f"RAW Leff phi real idx={idx_phi}",
        )
        idx_phi_p1 = (idx_phi + 1) % 360
        plt.plot(
            self.leff.freq_mhz,
            self.leff.leff_phi.real[idx_phi_p1, idx_theta],
            "*",
            label=f"RAW Leff phi real idx={idx_phi_p1}",
        )
        plt.plot(
            self.leff.freq_mhz,
            self.leff.leff_theta.real[idx_phi, idx_theta],
            "*",
            label=f"RAW Leff theta real idx={idx_theta}",
        )
        idx_theta_p1 = (idx_theta + 1) % 90
        plt.plot(
            self.leff.freq_mhz,
            self.leff.leff_theta.real[idx_phi, idx_theta_p1],
            "*",
            label=f"RAW Leff theta real idx={idx_theta_p1}",
        )
        plt.grid()
        plt.xlabel("MHz")
        plt.legend()

    def plot_leff_pol(self):
        leff_pol = self.get_fft_leff_pol(self.leff)
        plt.figure()
        plt.title(
            f"Interpolated Leff {self.leff.name} polar angle ({np.rad2deg(self.angle_pol):.1f} deg) at phi={self.dir_src_deg[0]:.1f}, theta={self.dir_src_deg[1]:.1f}"
        )
        plt.plot(self.o_pre.freq_out_mhz, leff_pol.real, label="Leff polar real")
        plt.plot(self.o_pre.freq_out_mhz, leff_pol.imag, label="Leff polar imag")
        plt.plot(self.o_pre.freq_out_mhz, np.abs(leff_pol), label="|Leff|")
        plt.grid()
        plt.xlabel("MHz")
        plt.ylim([-5, 5])
        plt.legend()

    def plot_leff_xyz(self):
        plt.figure()
        plt.title(
            f"Interpolated Leff {self.leff.name} x, y, z at phi={self.dir_src_deg[0]:.1f}, theta={self.dir_src_deg[1]:.1f}"
        )
        plt.plot(self.o_pre.freq_out_mhz, np.abs(self.l_x), label="abs(Leff_x)")
        plt.plot(self.o_pre.freq_out_mhz, np.abs(self.l_y), label="abs(Leff_y)")
        plt.plot(self.o_pre.freq_out_mhz, np.abs(self.l_z), label="abs(Leff_z)")
        plt.xlim([0, 300])
        plt.grid()
        plt.xlabel("MHz")
        plt.legend()


class DetectorUnitAntenna3Axis:
    """
    Compute DU response at efield

    """

    def __init__(self, d_leff):
        """

        :param name:
        """
        self.name = "TBD"
        self.pos_du_xcs = np.zeros(3, dtype=np.float32)
        self.interp_leff = LengthEffectiveInterpolation()
        self.freq_out_mhz = np.zeros(0)
        # Hypothesis : all leff storage have same array freq definition
        self.sn_leff = d_leff["sn"]
        self.ew_leff = d_leff["ew"]
        self.up_leff = d_leff["up"]
        self.leff = [d_leff["sn"], d_leff["ew"], d_leff["up"]]
        logger.debug("Hypothesis : all leff storage have same array angle phi, theta")
        self.interp_leff.set_sampling_angle(self.sn_leff.theta_deg, self.sn_leff.phi_deg)

    def set_name_pos(self, name, pos_xcs):
        """
        [XCS] is the frame associated to XCore of air shower
        :param name:
        :param pos_xcs: [m] (3,) in stations frame [XCS]
        """
        self.name = name
        self.pos_du_xcs = pos_xcs
        self._update_dir_source()

    def set_freq_out_mhz(self, out_freq):
        self.freq_out_mhz = out_freq
        freq_in_mhz = self.sn_leff.freq_mhz
        self.interp_leff.o_pre.init_linear_interpol(freq_in_mhz, out_freq)

    def set_pos_source(self, pos_xcs):
        """
        set of source mainly Xmax in [XCS]

        :param pos_xcs:
        :type pos_xcs:
        """
        self.pos_src_xcs = pos_xcs
        self._update_dir_source()

    def set_dir_source(self, dir_du):
        """
         in RAD
        :param dir_du: (azi, d_zen)
        """
        self.dir_src_du = dir_du
        self.interp_leff.set_dir_source(dir_du)

    def _update_dir_source(self):
        """
        return direction of source in [DU] frame
        :param self:
        :type self:
        """
        diff_n = self.pos_src_xcs - self.pos_du_xcs
        # Hypothesis: small network  (20-30km ) => [XCS]=[DU]+offset, so direction ar same
        self.cart_src_du = diff_n
        self.dir_src_du = coord.du_cart_to_dir(diff_n)
        self.interp_leff.set_dir_source(self.dir_src_du)

    def get_resp_3d_efield_du(self, fft_efield_du):
        """Return fft of antennas response for 3 axis with efield in [XCS] frame

        :param fft_efield_du: electric field at DU in [DU]
        :type fft_efield_du: float (3, n_s)
        """
        resp = np.empty_like(fft_efield_du)
        itp = self.interp_leff
        resp[0] = np.sum(itp.get_fft_leff_du(self.sn_leff) * fft_efield_du, axis=0)
        resp[1] = np.sum(itp.get_fft_leff_du(self.ew_leff) * fft_efield_du, axis=0)
        resp[2] = np.sum(itp.get_fft_leff_du(self.up_leff) * fft_efield_du, axis=0)
        return resp

    def get_resp_2d_efield_tan(self, fft_efield_tan):
        """Return fft of antennas response for 3 axis with efield in [TAN] tangential plan

        :param efield_tan: electric field in [TAN]
        :type efield_tan: float (2, n_s)
        """
        resp = np.empty((3,fft_efield_tan.shape[1]), dtype=fft_efield_tan.dtype)
        itp = self.interp_leff
        resp[0] = np.sum(itp.get_fft_leff_tan(self.sn_leff) * fft_efield_tan, axis=0)
        resp[1] = np.sum(itp.get_fft_leff_tan(self.ew_leff) * fft_efield_tan, axis=0)
        resp[2] = np.sum(itp.get_fft_leff_tan(self.up_leff) * fft_efield_tan, axis=0)
        return resp

    def get_resp_1d_efield_pol(self, fft_efield_pol):
        """Return fft of antennas response for 3 axis with efield in [POL] linear polarization

        :param efield_pol:electric field in [POL]
        :type efield_pol: float32 (n_s,)
        """
        resp = np.empty((3, fft_efield_pol.shape[0]), dtype=fft_efield_pol.dtype)
        itp = self.interp_leff
        resp[0] = itp.get_fft_leff_pol(self.sn_leff) * fft_efield_pol
        resp[1] = itp.get_fft_leff_pol(self.ew_leff) * fft_efield_pol
        resp[2] = itp.get_fft_leff_pol(self.up_leff) * fft_efield_pol
        return resp
