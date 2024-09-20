"""
Handling a set of 3D traces
"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from rshower.basis.traces_event import Handling3dTraces
import rshower.num.signal as sns
from rshower.basis.frame import FrameDuFrameTan
import rshower.basis.coord as coord

logger = getLogger(__name__)


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
    pol_est = np.sum(sple_ok, axis=1) / np.sum(n_elec_hb)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.debug(f"pol_est: {pol_est} with {len(idx_hb)} values out of noise.")
    return pol_est, idx_hb


class HandlingEfield(Handling3dTraces):
    """
    Handling a set of E field traces associated to one event observed on Detector Unit network

    Goal: apply specific E field processing on all traces
    """

    def __init__(self, name="NotDefined"):
        super().__init__(name)
        self.xmax = -1

    #
    # INTERNAL
    #
    def _efield_acp(self):
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
        for idx in range(self.get_nb_du()):
            if np.dot(max_sample[idx], v_pol[idx]) < 0:
                # print(f"swap {idx}")
                v_pol[idx] *= -1
        return v_pol, v_dir_src

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

    #
    # GETTER
    #
    def get_dir_source(self):
        pass

    def get_polar_vec(self, threshold=25):
        a_vec_pol = np.empty((self.get_nb_du(), 3), dtype=np.float32)
        for idx in range(self.get_nb_du()):
            a_vec_pol[idx, :], _ = fit_vec_linear_polar_l2(self.traces[idx], threshold)
        return a_vec_pol

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
        vec_unit_polar = self.get_polar_vec()
        polars = np.empty(self.get_nb_du(), dtype=np.float32)
        dir_angle = np.empty((self.get_nb_du(), 2), dtype=np.float32)
        for idx in range(self.get_nb_du()):
            # direction toward source
            v_dux = self.xmax - self.network.du_pos[idx]
            v_dux /= np.linalg.norm(v_dux)
            logger.info(f"xmax  : {self.xmax}")
            logger.info(f"pos du: {self.network.du_pos[idx]}, {v_dux}")
            vec_dir_du = coord.du_cart_to_dir(v_dux)
            dir_angle[idx] = vec_dir_du
            #print(idx, np.rad2deg(vec_dir_du))
            t_dutan = FrameDuFrameTan(vec_dir_du)
            v_pol_tan = t_dutan.vec_to(vec_unit_polar[idx], "TAN")
            logger.info(f"{idx} {self.idx2idt[idx]} {v_pol_tan}")
            polars[idx] = coord.tan_cart_to_polar_angle(v_pol_tan)
        logger.info(f"mean dir {np.rad2deg(np.mean(dir_angle, 0))}")
        if degree:
            return np.rad2deg(polars), dir_angle
        return polars, dir_angle

    #
    # PLOTS
    #
    def plot_trace_tan(self, idx):
        v_dux = self.xmax - self.network.du_pos[idx]
        vec_dir_du = coord.du_cart_to_dir(v_dux)
        t_dutan = FrameDuFrameTan(vec_dir_du)
        traces_tan = np.empty((self.get_size_trace(), 3), dtype=np.float32)
        for i_s in range(self.get_size_trace()):
            traces_tan[i_s] = t_dutan.vec_to_b(self.traces[idx, :, i_s])
        fig = plt.figure("Trace in tangential frame: e_theta, e_phi")
        vmin = 2000
        vmax = 2200
        norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        sca = plt.scatter(
            traces_tan[:, 0],
            traces_tan[:, 1],
            s=100,
            norm=norm_user,
            c=1 + np.arange(self.get_size_trace()),
        )
        fig.colorbar(sca, label="")
        plt.grid()
        plt.figure()
        plt.title("Histogram normal composant, (must be close to zero)")
        plt.hist(traces_tan[:, 2])
        plt.yscale("log")
        plt.grid()

    def plot_polar_angle(self):
        polars, dir_angle = self.get_polar_angle_efield(True)
        self.network.plot_footprint_1d(
           polars , "Polar angle, degree", self, scale="lin", unit="deg"
        )

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
