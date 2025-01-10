"""
Colley Jean-Marc, CNRS/IN2P3/LPNHE

Handling a set of 3D traces
"""
from logging import getLogger

import numpy as np
import scipy.linalg as splg
import matplotlib.pyplot as plt
from matplotlib import colors

from rshower.basis.traces_event import Handling3dTraces
import rshower.num.signal as sns
from rshower.basis.frame import FrameDuFrameTan
import rshower.basis.coord as coord

logger = getLogger(__name__)


def estimate_xmax_line(v_dir_src, du_pos):
    """

    :param v_dir_src:
    :param du_pos:
    """
    nb_trace = du_pos.shape[0]
    mat_xmax = np.empty((nb_trace * 2, 3), dtype=np.float64)
    sec_member = np.zeros((nb_trace * 2), dtype=np.float64)
    du_pos_t = np.copy(du_pos)
    temp = du_pos_t[:, 0]
    du_pos_t[:, 0] = -du_pos_t[:, 1]
    du_pos_t[:, 1] = temp
    for idx in range(nb_trace):
        idx2 = 2 * idx
        idx2_1 = idx2 + 1
        # normal vector of plan: p = dir ^ pos_i
        mat_xmax[idx2] = np.cross(v_dir_src[idx], du_pos[idx])
        mat_xmax[idx2_1] = np.cross(v_dir_src[idx], du_pos_t[idx])
        # <pos_i, p>
        # sec_member[idx2] = np.dot(mat_xmax[idx2], du_pos[idx])
        # 0 because mixte product
        sec_member[idx2_1] = np.dot(mat_xmax[idx2_1], du_pos[idx])
    xmax, res, rnk, sgl_val = splg.lstsq(mat_xmax, sec_member)
    # print(f"sec_member: {sec_member}\nRank: {rnk}")
    # print(mat_xmax)
    v_res = np.matmul(mat_xmax, xmax) - sec_member
    # print(v_res)
    v_res = np.sum(np.abs(v_res.reshape(nb_trace, 2)), axis=-1)
    return xmax, v_res


def fit_vec_linear_polar_l2(trace, threshold=25):
    """Fit the unit linear polarization vec with samples out of noise (>threshold)

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
    pol_est = np.sum(sple_ok * n_elec_hb, axis=1) / np.sum(n_elec_hb)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.debug(f"pol_est: {pol_est} w)th {len(idx_hb)} values out of noise.")
    # print(pol_est.shape, np.squeeze(pol_est).shape)
    return np.squeeze(pol_est), idx_hb


class HandlingEfield(Handling3dTraces):
    """
    Handling a set of E field traces associated to one event observed on Detector Unit network

    Goal: apply specific E field processing on all traces
    """

    def __init__(self, name="NotDefined"):
        super().__init__(name)
        self.xmax = -1
        self.polar_angle_rad = None

    #
    # SETTER
    #
    def set_xmax(self, xmax):
        """Set Xmax position

        :param xmax: [m] cartesian position in same frame as DU position
        :type xmax: float (3,)
        """
        assert xmax.ndim == 1
        assert xmax.shape[0] == 3
        self.xmax = xmax
        self.network.xmax_pos = xmax

    #
    # GETTER
    #
    def get_polar_normal_vec(self):
        """Return unit vector of polarization and normal of Efield
        with PCA method

        Principal component analysis (PCA)

        v_pol: (nb_trace,3) where EField extrenum is positif
        v_dir_src : (nb_trace,3) normal of plan wave, ~ direction Xmax, ie z > 0
        """
        # AAt, where A is trace (3, nb_sample)
        m_ata = np.matmul(self.traces, np.swapaxes(self.traces, 1, 2))
        eig_val, eig_vec = np.linalg.eig(m_ata)
        i_nor = np.argmin(eig_val, axis=1, keepdims=True)
        i_pol = np.argmax(eig_val, axis=1, keepdims=True)
        # print(i_nor[:, None].shape, i_pol[:, None].shape)
        v_pol = np.take_along_axis(eig_vec, i_pol[:, None], axis=2)
        v_dir_src = np.take_along_axis(eig_vec, i_nor[:, None], axis=2)
        v_pol = np.squeeze(v_pol)
        v_dir_src = np.squeeze(v_dir_src)
        # turn direction to sky, ie z > 0
        z_neg = np.argwhere(v_dir_src[:, 2] < 0)
        v_dir_src[z_neg] *= -1
        # turn pol toword max efield value
        norm = np.linalg.norm(self.traces, axis=1)
        i_max = np.argmax(norm, axis=1, keepdims=True)
        max_sample = np.take_along_axis(self.traces, i_max[:, None], axis=2)
        max_sample = np.squeeze(max_sample)
        assert np.allclose(
            np.linalg.norm(max_sample, axis=1), np.squeeze(np.take_along_axis(norm, i_max, axis=1))
        )
        # TODO: can be replace by np.vecdot with numpy 2.0
        for idx in range(self.get_nb_trace()):
            if np.dot(max_sample[idx], v_pol[idx]) < 0:
                # print(f"swap {idx}")
                v_pol[idx] *= -1
        self.v_pol = v_pol
        self.v_dir_src = v_dir_src
        return v_pol, v_dir_src

    def get_polar_vec(self, threshold=25):
        a_vec_pol = np.empty((self.get_nb_trace(), 3), dtype=np.float32)
        for idx in range(self.get_nb_trace()):
            a_vec_pol[idx, :], _ = fit_vec_linear_polar_l2(self.traces[idx], threshold)
        return a_vec_pol

    # def get_traces_passband(self, f_mhz=[30, 250], causal=False):
    #     """Return array traces with passband filter
    #
    #     :param f_mhz: [MHz] border
    #     :type f_mhz: list of 2 number
    #     """
    #     raise
    #     if causal:
    #         return sns.filter_butter_band_lfilter(self.traces, f_mhz[0], f_mhz[1], self.f_samp_mhz)
    #     return sns.filter_butter_band(self.traces, f_mhz[0], f_mhz[1], self.f_samp_mhz)

    def get_polar_angle(self, degree=False):
        """
        Return polar angle estimation with traces E field for all DUs

        attribut filled:
          * ef_pol: (nb_du, nb_sample) Efield polar !
          * polar_angle_rad : (nb_du,) polar angle 
          
        :param degree: flag to set return angle in degree
        :type degree: bool

        :return: polars, dir_angle, dir_vec
            polars (nb_du,) like polar_angle_rad
            dir_angle (nb_du,2) xmax direction angle (azi, d_zen)
            dir_vec (nb_du,3)   xmax direction cartesian unit
        :rtype: float (nb_du,)
        """
        assert isinstance(self.xmax, np.ndarray)
        # in DU frame
        vec_unit_polar = self.get_polar_vec()
        self.ef_pol = np.empty((self.get_nb_trace(), self.get_size_trace()), dtype=np.float32)
        polars = np.empty(self.get_nb_trace(), dtype=np.float32)
        dir_angle = np.empty((self.get_nb_trace(), 2), dtype=np.float32)
        # dir_vec : direction to Xmax, unit vector
        dir_vec = np.empty((self.get_nb_trace(), 3), dtype=np.float32)
        for idx in range(self.get_nb_trace()):
            # direction toward source
            v_dux = self.xmax - self.network.du_pos[idx]
            v_dux /= np.linalg.norm(v_dux)
            dir_vec[idx] = v_dux
            logger.debug(f"xmax  : {self.xmax}")
            logger.debug(f"pos du: {self.network.du_pos[idx]}, {v_dux}")
            vec_dir_du = coord.nwu_cart_to_dir_one(v_dux)
            dir_angle[idx] = vec_dir_du
            # print(idx, np.rad2deg(vec_dir_du))
            t_dutan = FrameDuFrameTan(vec_dir_du)
            v_pol_tan = t_dutan.vec_to(vec_unit_polar[idx], "TAN")
            polars[idx] = coord.tan_cart_to_polar_angle(v_pol_tan)
            self.ef_pol[idx] = np.dot(self.traces[idx].T, vec_unit_polar[idx])
        logger.info(f"mean dir {np.rad2deg(np.mean(dir_angle, 0))}")
        self.polar_angle_rad = polars
        if degree:
            return np.rad2deg(polars), np.rad2deg(dir_angle), dir_vec
        return polars, dir_angle, dir_vec

    #
    # Procesing
    #
    def estimate_xmax_with_wave_plan(self):
        _, v_dir_src = self.get_pca()
        return estimate_xmax_line(v_dir_src, self.network.du_pos)

    #
    # PLOTS
    #
    def plot_trace_pol_idx(self, idx):
        plt.figure()
        title = f"n{self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        plt.title(title)
        plt.plot(self.t_samples[idx], self.ef_pol[idx])
        plt.xlabel("ns")
        plt.ylabel(self.unit_trace)
        plt.grid()

    def plot_trace_tan_idx(self, idx):
        """Scatter plot tangentiel component and histogram for normal

        Normal defined by Xmax definition

        :param idx:
        """
        v_dux = self.xmax - self.network.du_pos[idx]
        vec_dir_du = coord.nwu_cart_to_dir_one(v_dux)
        t_dutan = FrameDuFrameTan(vec_dir_du)
        print(f"trace tan: {vec_dir_du}")
        traces_tan = np.empty((self.get_size_trace(), 3), dtype=np.float32)
        for i_s in range(self.get_size_trace()):
            traces_tan[i_s] = t_dutan.vec_to_b(self.traces[idx, :, i_s])
        # plt.figure()
        fig = plt.figure()
        title = "Trace in tangential frame"
        title += f"n{self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        plt.title(title)
        vmin = 2000
        vmax = 2200
        norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        max_tr = np.max(np.abs(traces_tan)) * 1.1
        sca = plt.scatter(
            traces_tan[:, 0],
            traces_tan[:, 1],
            s=100,
            norm=norm_user,
            c=1 + np.arange(self.get_size_trace()),
        )
        fig.colorbar(sca, label="")
        plt.grid()
        plt.xlim(-max_tr, max_tr)
        s_xlabel = r"$e_{\theta} => $, vertical polarization."
        if self.polar_angle_rad is not None:
            s_xlabel += f"\nPolar angle estimated {np.rad2deg(self.polar_angle_rad[idx]):.1f}°"
        plt.xlabel(s_xlabel)
        plt.ylabel(r"$e_{\phi}$ =>, horizontal polarization.")
        plt.ylim(-max_tr, max_tr)
        plt.figure()
        plt.title("Histogram normal composant, (must be close to zero)")
        plt.hist(traces_tan[:, 2])
        plt.yscale("log")
        plt.grid()

    def plot_polar_angle(self):
        polars, dir_angle, _ = self.get_polar_angle(True)
        self.network.plot_footprint_1d(polars, "Polar angle, degree", self, scale="lin", unit="deg")

    def plot_polar_check_fit(self, threshold=40):
        nb_du = len(self.idt2idx)
        a_vec_pol = np.empty((nb_du, 3), dtype=np.float32)
        a_stat = np.empty((nb_du, 2), dtype=np.float32)
        a_nb_sple = np.empty(nb_du, dtype=np.float32)
        for idx in range(nb_du):
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

    def plot_trace_idx(self, idx, to_draw="012"):
        super().plot_trace_idx(idx, to_draw)
        self.plot_trace_tan_idx(idx)
        self.plot_trace_pol_idx(idx)
