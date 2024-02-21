"""
Handling a set of 3D traces
"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

from rshower import Handling3dTraces
import rshower.num.signal as sns
from rshower.basis.frame import FrameDuFrameTan
import rshower.basis.coord as coord

logger = getLogger(__name__)


# def fit_vec_linear_polar(trace, threshold=20):
#     """
#
#     gere les changement de sens du E field
#
#     :param trace:
#     :type trace: float (3, n_s)
#     :param threshold:
#     :type threshold:
#     """
#     assert trace.shape[0] == 3
#     n_elec = np.linalg.norm(trace, axis=0)
#     idx_hb = np.where(n_elec > threshold)[0]
#     logger.debug(f"{trace.shape} {np.argmax(trace)}")
#     # unit
#     sple_ok = trace[:, idx_hb] / n_elec[idx_hb]
#     idx_neg = np.where(sple_ok[1] < 0)[0]
#     sple_ok[:, idx_neg] = -sple_ok[:, idx_neg]
#     n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
#     # weighted estimation
#     temp = sple_ok * n_elec_2
#     logger.debug(f"{sple_ok.shape} {n_elec_2.shape}")
#     # (3,)/()
#     pol_est = np.sum(temp, axis=1) / np.sum(n_elec_2)
#     logger.info(pol_est)
#     # unit vect
#     pol_est /= np.linalg.norm(pol_est)
#     logger.info(pol_est.shape)
#     logger.info(pol_est)
#     logger.info(np.linalg.norm(pol_est))
#     # TODO: check if lineat polarization is here with stats test
#     return pol_est


def fit_vec_linear_polar_with_max(trace):
    """

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    assert trace.shape[0] == 3
    n_elec = np.linalg.norm(trace, axis=0)
    idx_max = np.argmax(n_elec)
    v_pol = trace[:, idx_max] / n_elec[idx_max]
    logger.debug(v_pol)
    return v_pol


def fit_vec_linear_polar_l2_2(trace, threshold=20, plot=False):
    """Fit the unit linear polarization vec with samples out of noise (>threshold)

    We used weighted estimation with square l2 norm

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    n_elec = np.linalg.norm(trace, axis=0)
    idx_hb = np.where(n_elec > threshold)[0]
    logger.debug(f"{len(idx_hb)} samples out noise :\n{idx_hb}")
    # to unit vector for samples out noise
    # (3,ns)/(ns) => OK
    sple_ok = trace[:, idx_hb] / n_elec[idx_hb]
    logger.debug(sple_ok)
    n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
    # weighted estimation with norm^2
    # temp is (3,n_s)=(3,ns)*(ns)
    temp = sple_ok * n_elec_2
    pol_est = np.sum(temp, axis=1) / np.sum(n_elec_2)
    logger.info(pol_est)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.info(f"pol_est: {pol_est}, {np.linalg.norm(pol_est)}")
    return pol_est, idx_hb


def fit_vec_linear_polar_l2(trace, threshold=20):
    """Fit the unit linear pola vec with samples out of noise (>threshold)

    We used weighted estimation with norm l2

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    n_elec = np.linalg.norm(trace, axis=0)
    idx_hb = np.where(n_elec > threshold)[0]
    if len(idx_hb) == 0:
        # set to nan to be excluded by plot
        return np.array([[np.nan, np.nan, np.nan]]), np.array([])
    # logger.debug(f"{len(idx_hb)} samples out noise :\n{idx_hb}")
    # to unit vector for samples out noise
    # (3,ns)/(ns) => OK
    n_elec_hb = n_elec[idx_hb]
    sple_ok = trace[:, idx_hb]
    # logger.debug(sple_ok)
    # weighted estimation with norm
    pol_est = np.sum(sple_ok, axis=1) / np.sum(n_elec_hb)
    logger.info(pol_est)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.info(f"pol_est: {pol_est} with {len(idx_hb)} values out of noise.")
    return pol_est, idx_hb


def fit_vec_linear_polar_hls(trace):
    """Fit the unit linear pola vec with homogenous linear system


    :param trace:
    :type trace: float (3, n_s)

    """
    # TODO: add weigth
    n_sple = trace.shape[1]
    m_a = np.zeros((3 * n_sple, 3), dtype=np.float32)
    # p_x coeff
    m_a[:n_sple, 0] = -trace[1]
    m_a[:n_sple, 1] = trace[0]
    # p_y coeff
    m_a[n_sple : 2 * n_sple, 0] = -trace[2]
    m_a[n_sple : 2 * n_sple, 2] = trace[0]
    # p_z coeff
    m_a[2 * n_sple : 3 * n_sple, 1] = -trace[2]
    m_a[2 * n_sple : 3 * n_sple, 2] = trace[1]
    # solve
    m_ata = np.matmul(m_a.T, m_a)
    assert m_ata.shape == (3, 3)
    w_p, vec_p = np.linalg.eig(m_ata)
    # logger.debug(f"{w_p}")
    # logger.debug(f"{vec_p}")
    vec_pol = vec_p[:, 0]
    logger.debug(f"vec_pol eigen : {vec_pol}")
    res = np.matmul(m_a, vec_pol)
    # logger.debug(f"{res} ")
    # logger.debug(f"{np.linalg.norm(res)} {res.min()} {res.max()}")
    # assert np.allclose(np.linalg.norm(np.matmul(m_ata, vec_pol)), 0)
    return vec_pol


def check_vec_linear_polar_proto(trace, idx_on, vec_pol):
    """

    :param trace_on: sample of trace out noise
    :type trace_on: float (3,n) n number of sample
    :param vec_pol:
    :type vec_pol:float (3,)
    """
    if idx_on is not None:
        if len(idx_on) == 0:
            # set to nan to be excluded by plot
            return np.nan, np.nan
        trace_on = trace[:, idx_on]
    else:
        trace_on = trace
        idx_on = np.arange(trace.shape[1])
    # plt.figure()
    # plt.plot(idx_on, trace_on[0], label="x")
    # plt.plot(idx_on, trace_on[1], label="y")
    # plt.plot(idx_on, trace_on[2], label="z")
    norm_tr = np.linalg.norm(trace_on, axis=0)
    # logger.info(norm_tr)
    tr_u = trace_on / norm_tr
    # logger.info(tr_u)
    cos_angle = np.dot(vec_pol, tr_u)
    idx_pb = np.argwhere(cos_angle > 1)
    cos_angle[idx_pb] = 1.0
    # logger.info(cos_angle)
    assert cos_angle.shape[0] == idx_on.shape[0]
    angles = np.rad2deg(np.arccos(cos_angle))
    # logger.info(angles)
    # for measures in opposite direction
    idx_neg = np.where(angles > 180)[0]
    angles[idx_neg] -= 180
    idx_neg = np.where(angles > 90)[0]
    angles[idx_neg] = 180 - angles[idx_neg]
    # logger.info(angles)
    assert np.alltrue(angles >= 0)
    mean, std = angles.mean(), angles.std()
    # weighted estimation
    norm2_tr = np.sum(trace_on * trace_on, axis=0)
    prob = norm2_tr / np.sum(norm2_tr)
    mean_w = np.sum(angles * norm_tr * norm_tr) / np.sum(norm_tr * norm_tr)
    mean_w2 = np.sum(angles * prob)
    diff = angles - mean_w2
    std_w2 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error: {mean}, sigma {std} ")
    logger.debug(f"Angle error w2: {mean_w2} {std_w2}")
    prob = norm_tr / np.sum(norm_tr)
    assert np.isclose(prob.sum(), 1)
    mean_w1 = np.sum(angles * prob)
    diff = angles - mean_w2
    std_w1 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error w1: {mean_w1} {std_w1}")
    # plt.figure()
    # plt.hist(angles)
    return mean_w2, std_w2


def check_vec_linear_polar_l2(trace, idx_on, vec_pol):
    """

    :param trace_on: sample of trace out noise
    :type trace_on: float (3,n) n number of sample
    :param vec_pol:
    :type vec_pol:float (3,)
    """
    if idx_on is not None:
        if len(idx_on) == 0:
            # set to nan to be excluded by plot
            return np.nan, np.nan
        trace_on = trace[:, idx_on]
    else:
        trace_on = trace
        idx_on = np.arange(trace.shape[1])
    norm_tr = np.linalg.norm(trace_on, axis=0)
    # logger.info(norm_tr)
    tr_u = trace_on / norm_tr
    # logger.info(tr_u)
    cos_angle = np.dot(vec_pol, tr_u)
    idx_pb = np.argwhere(cos_angle > 1)
    cos_angle[idx_pb] = 1.0
    # logger.info(cos_angle)
    assert cos_angle.shape[0] == idx_on.shape[0]
    angles = np.rad2deg(np.arccos(cos_angle))
    # logger.info(angles)
    # for measures in opposite direction
    idx_neg = np.argwhere(angles > 180)
    angles[idx_neg] -= 180
    idx_neg = np.argwhere(angles > 90)
    angles[idx_neg] = 180 - angles[idx_neg]
    # logger.info(angles)
    assert np.alltrue(angles >= 0)
    prob = norm_tr / np.sum(norm_tr)
    assert np.isclose(prob.sum(), 1)
    mean_l2 = np.sum(angles * prob)
    diff = angles - mean_l2
    std_l2 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error l2: {mean_l2} {std_l2}")
    return mean_l2, std_l2


def efield_in_polar_frame(efield3d, threshold=40):
    """Return E field in linear polarization direction

    :param efield3d: [uV/m] Efield 3D
    :type efield3d: float (3, n_s)
    :param threshold: [uV/m] used to select sample to fit direction
    :type threshold: float (n_s,)
    """
    pol_est, idx_on = fit_vec_linear_polar_l2(efield3d, threshold)
    check_vec_linear_polar_l2(efield3d, idx_on, pol_est)
    efield1d = np.dot(efield3d.T, pol_est)
    return efield1d, pol_est


class HandlingEfield(Handling3dTraces):
    """
    Handling a set of E field traces associated to one event observed on Detector Unit network

    Goal: apply specific E field processing on all traces
    """

    def __init__(self, name="NotDefined"):
        super().__init__(name)
        self.xmax = -1

    ### INTERNAL

    ### SETTER

    def set_xmax(self, xmax):
        """Set Xmax position

        :param xmax: [m] cartesian position in same frame as DU position
        :type xmax: float (3,)
        """
        assert xmax.ndim == 1
        assert xmax.shape[0] == 3
        self.xmax = xmax

    ### INIT/SETTER

    def get_polar_vec(self, threshold=40):
        a_vec_pol = np.empty((self.get_nb_du(), 3), dtype=np.float32)
        for idx in range(self.get_nb_du()):
            a_vec_pol[idx, :], _ = fit_vec_linear_polar_l2(self.traces[idx], threshold)
        return a_vec_pol

    # def get_tmax_vmax(self, interpol="auto"):
    #     """Return time where norm is max
    #
    #     :return:  time of max and max
    #     :rtype: float(nb_du,) , float(nb_du,)
    #     """
    #     tr_norm = np.linalg.norm(self.traces, axis=1)
    #     idx_max = np.argmax(tr_norm, axis=1)
    #     if interpol != "no":
    #         if not interpol in ["parab", "auto"]:
    #             raise
    #         t_max = np.empty_like(idx_max, dtype=np.float32)
    #         e_max = np.empty_like(idx_max, dtype=np.float32)
    #         for idx in range(len(idx_max)):
    #             logger.debug(f"{idx} {self.idx2idt[idx]} {idx_max[idx]}")
    #             if interpol == "parab":
    #                 t_max[idx], e_max[idx] = sns.find_max_with_parabola_interp_3pt(
    #                     self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
    #                 )
    #             else:
    #                 t_max[idx], e_max[idx] = sns.find_max_with_parabola_interp(
    #                     self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
    #                 )
    #                 logger.debug(f"{t_max[idx]} ; {e_max[idx]}")
    #         self.t_max = t_max
    #         self.e_max = e_max
    #         return t_max, e_max
    #     idx_max = idx_max[:, np.newaxis]
    #     t_max = np.take_along_axis(self.t_samples, idx_max, axis=1)
    #     v_max = np.take_along_axis(tr_norm, idx_max, axis=1)
    #     # remove dimension (np.squeeze) to have ~vector ie shape is (n,) instead (n,1)
    #     self.t_max = np.squeeze(t_max)
    #     self.e_max = np.squeeze(e_max)
    #     return self.t_max, self.e_max

    def get_traces_passband(self, f_mhz=[30, 250], causal=False):
        """Return array traces with passband filter

        :param f_mhz: [MHz] border
        :type f_mhz: list of 2 number
        """
        if causal:
            return sns.filter_butter_band_lfilter(self.traces, f_mhz[0], f_mhz[1], self.f_samp_mhz)
        return sns.filter_butter_band(self.traces, f_mhz[0], f_mhz[1], self.f_samp_mhz)

    def get_polar_angle_efield(self, degree=False):
        """Return polar angle estimation with traces E field for all DUs

        :param degree: flag to set return angle in degree
        :type degree: bool

        :return: polar angle estimation with traces E field for all DUs
        :rtype: float (nb_du,)
        """
        assert isinstance(self.xmax, np.ndarray)
        # in DU frame
        vec_polar = self.get_polar_vec()
        polars = np.empty(self.get_nb_du(), dtype=np.float32)
        for idx in range(self.get_nb_du()):
            # direction toward source
            v_dux = self.xmax - self.network.du_pos[idx]
            vec_dir_du = coord.du_cart_to_dir(v_dux)
            t_dutan = FrameDuFrameTan(vec_dir_du)
            v_pol_tan = t_dutan.vec_to(vec_polar[idx], "TAN")
            polars[idx] = coord.tan_cart_to_polar_angle(v_pol_tan)
        if degree:
            return np.rad2deg(polars)
        return polars

    #
    # PLOTS
    #

    def plot_polar_check_fit(self, threshold=40):
        nb_du = self.get_nb_du()
        a_vec_pol = np.empty((nb_du, 3), dtype=np.float32)
        a_stat = np.empty((nb_du, 2), dtype=np.float32)
        a_nb_sple = np.empty(nb_du, dtype=np.float32)
        for idx in range(self.get_nb_du()):
            vec, idx_on = fit_vec_linear_polar_l2(self.traces[idx], threshold)
            a_vec_pol[idx, :] = vec.ravel()
            a_nb_sple[idx] = len(idx_on)
            a_stat[idx, 0], a_stat[idx, 1] = check_vec_linear_polar_l2(
                self.traces[idx], idx_on, vec
            )
        self.network.plot_footprint_4d(self, a_vec_pol, "Unit polar vector", False)
        self.network.plot_footprint_1d(
            a_stat[:, 0], "Mean of polar angle fit residu", self, scale="lin", unit="deg"
        )
        self.network.plot_footprint_1d(
            a_stat[:, 1], "Std of polar angle fit residu", self, scale="lin", unit="deg"
        )
        self.network.plot_footprint_1d(
            a_nb_sple, "Nunber of samples used to fit polar vector", self, scale="lin"
        )
