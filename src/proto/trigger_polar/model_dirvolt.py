#! /usr/bin/env python3
"""
Colley Jean-Marc
"""
import pathlib
from logging import getLogger
import logging
from datetime import datetime
import re

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from numba import guvectorize, float32, int32, njit
from scipy.interpolate import griddata, make_splrep, BSpline, splrep, splev, splint

import rshower.basis.coord as coord
from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
import proto.simu_dc2.asdf_traces as f_tr
from rshower.simu.gal_resp import GalacticRespDetectorGenerator


logger = getLogger(__name__)


@njit
# @guvectorize([(float32[:], int32[:])], '(n)->(2)', nopython=True)
#   ne fonctionne pas, problème de signature avec ->(2) à la compilation
# en version njit pas de gain énorme en appelant min et max de numpy ...
def argminmax_trace(tra, amm):
    nb_s = tra.shape[1]
    for axis in range(3):
        min_idx = 0
        max_idx = 0
        trax = tra[axis]
        min_val = trax[0]
        max_val = trax[0]
        for idx in range(nb_s):
            val = trax[idx]
            if val < min_val:
                min_val = val
                min_idx = idx
            elif val > max_val:
                max_val = val
                max_idx = idx
        amm[axis, 0] = min_idx
        amm[axis, 1] = max_idx


@njit
def get_max_interpolate(x_trace, y_trace, idx_max):
    """Parabolic interpolation of the maximum with 3 points
    trace : all values >=  0

    :param x_trace:
    :param y_trace:
    algo Mode pic, input 3 values and the middle one is max:
        parabola : ax^2 + bx + c
        offset of (x0, y0)
        solve coef a, b , interpolation of the maximum is
          x_m = x0 - b/2a
          y_m = y0 - b^2/4a
    :param idx_max: index of sample max, idx_max < nb_sample
    :type idx_max: int
    :return: x_max, y_maxidx_maxidx_maxidx_maxidx_max
    """
    # remove offset (x0, v0)
    y_pic = y_trace[idx_max : idx_max + 2] - y_trace[idx_max - 1]
    x_pic = x_trace[idx_max : idx_max + 2] - x_trace[idx_max - 1]
    # solve coef a, b : linear system
    r_pic = y_pic / x_pic
    c_a = (r_pic[1] - r_pic[0]) / (x_pic[1] - x_pic[0])
    c_b = r_pic[0] - c_a * x_pic[0]
    # compute x_max, y_max with offset
    x_m = -c_b / (2 * c_a)
    # x_max = x_trace[idx_max - 1] + x_m
    y_max = y_trace[idx_max - 1] + x_m * c_b / 2
    return y_max


def relative_voltage_evt(evt):
    """
    for each event:
        1) find argument of min max for each axis => 6 values
        2) parabola interpolation
        3) find the absolue max  from 6 optima : v_ref >0
        4) compute relative optimun for each axis
        5) store relative optimun and v_ref
    out format:
        * xmax_1_2_3, xmin_1_2_3, v_ref
    """
    # ===== 1)
    a_min = np.argmin(evt.traces, axis=2)
    a_max = np.argmax(evt.traces, axis=2)
    # a_min, a_max : (nb_du,3)
    # ===== 2)
    nb_tr = evt.get_nb_trace()
    rel_opti = np.zeros((nb_tr, 7), dtype=np.float32)
    # format : xmax_1_2_3, xmin_1_2_3, v_ref:
    for idx in range(nb_tr):
        for axis in range(3):
            traces = evt.traces[idx, axis]
            y_max = get_max_interpolate(evt.t_samples[idx], traces, a_max[idx, axis])
            rel_opti[idx, axis] = y_max
            y_min = get_max_interpolate(evt.t_samples[idx], -traces, a_min[idx, axis])
            rel_opti[idx, axis + 3] = -y_min
    # ===== 3)
    v_ref = np.max(np.abs(rel_opti), keepdims=True, axis=1)
    # ===== 4)
    rel_opti /= v_ref
    # assert np.all(rel_opti <= 1)
    # assert np.all(rel_opti >= -1)
    rel_opti[:, 6] = v_ref[:, 0]
    return rel_opti


class DirectionVoltageParameters:
    """Extract parameters for model direction voltage amplitude"""

    def __init__(self, pn_vash):
        self.pn_vash = pn_vash
        self.ie_endp1 = 0  #
        self.size_chk = 1
        self.out_dir = "/home/jcolley/projet/lucky/data/"

    def process_chunk_file(self, idx):
        """Process chunk file evt idx to idx+self.size_chk excluded"""
        f_ash = f_tr.AsdfReadTraces(self.pn_vash)
        l_events = []
        i_endp1 = min(self.ie_endp1, idx + self.size_chk)
        logger.info(f"process chunk event [{idx}, {i_endp1}]")
        for i_e in range(idx, i_endp1):
            evt = f_ash.get_event(i_e)
            rel_opti = relative_voltage_evt(evt)
            # collect direction
            idx_beg, idx_end = f_ash.get_event_interval(i_e)
            azi = f_ash.mtraces["azi"][idx_beg:idx_end]
            d_zen = f_ash.mtraces["d_zen"][idx_beg:idx_end]
            l_events.append([rel_opti, azi, d_zen])
        return l_events

    def process_events(self, ie_beg, ie_endp1):
        """
        numpy array format:
          0           ,     1,2,3,      4,5,6, 7    , 8            , 9
          index event, xmax_1_2_3, xmin_1_2_3, v_ref, azimuth [deg], dist_zenith [deg]
        """
        # manage index begin, end
        f_ef = f_tr.AsdfReadTraces(self.pn_vash)
        if ie_endp1 < 0:
            ie_endp1 = f_ef.get_nb_events()
        self.ie_endp1 = ie_endp1
        START = datetime.now()
        print("1 CPU")
        size_chk = ie_endp1 - ie_beg
        self.size_chk = size_chk
        l_events = self.process_chunk_file(ie_beg)
        # collect result in unique numpy array
        idx_beg, idx_end = f_ef.get_chunk_event_interval(ie_beg, ie_endp1 - 1)
        nb_tr = idx_end - idx_beg
        dataset = np.zeros((nb_tr, 10), dtype=np.float32)
        i_beg = 0
        i_evt = 0
        for res in l_events:
            nb_du = res[0].shape[0]
            i_end = i_beg + nb_du
            dataset[i_beg:i_end, 0] = i_evt * np.ones(nb_du)
            dataset[i_beg:i_end, 1:8] = res[0]
            dataset[i_beg:i_end, 8] = res[1]
            dataset[i_beg:i_end, 9] = res[2]
            i_beg = i_end
            i_evt += 1
        print(nb_tr, i_evt, i_end)
        print(dataset[:, 7].mean(), dataset[:, 7].std())
        in_f = pathlib.Path(self.pn_vash)
        out_name = str(in_f.name).replace("volt", "dirvolt")
        out_name = out_name.replace(".asdf", "")
        np.save(self.out_dir + out_name, dataset)
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        print(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")


class ModelDirectionVoltage:
    """Estimation amplitude voltage and validation"""

    def __init__(self, nside=16):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.ds_prefix = r"^dirvolt-ash"

    def _define_healpix_pixel(self):
        rad_dir = np.deg2rad(self.ds_diramp[:, 8:])
        self.hpix = hp.ang2pix(self.nside, rad_dir[:, 1], rad_dir[:, 0])

    def _define_hit(self, hpix):
        hit = np.zeros(self.npix, dtype=np.int32)
        for pix in hpix:
            hit[pix] += 1
        self.hit = hit

    def init_collect_dataset(self, ds_dir):
        self.ds_dir = ds_dir
        pattern = re.compile(r"^dirvolt-ash")
        rep = pathlib.Path(ds_dir)
        ds_diramp = None
        cpt_file = 0
        for m_f in rep.iterdir():
            if m_f.is_file() and pattern.search(m_f.name):
                print(m_f)
                f_ds = str(m_f.absolute())
                m_ds = np.load(f_ds)
                if ds_diramp is None:
                    ds_diramp = m_ds
                else:
                    ds_diramp = np.vstack((ds_diramp, m_ds))
                cpt_file += 1
                print(ds_diramp.shape)
                if cpt_file == 9:
                    break
        # rng = np.random.default_rng()
        print(ds_diramp.shape)
        # np.random.shuffle(ds_diramp)
        print(ds_diramp.shape)
        self.ds_diramp_ref = ds_diramp.copy()
        self.check = np.sort(self.ds_diramp_ref.sum(axis=1))
        self.ds_diramp = ds_diramp
        self.ds_tra = self.ds_diramp
        self._define_healpix_pixel()
        self._define_hit(self.hpix)

    def partitioning_dataset(self, ash_faction, f_shuffle=False, f_roll=False):
        assert ash_faction <= 1.0
        fidx_vld = int(self.ds_diramp.shape[0] * ash_faction)
        print("fidx_vld:", fidx_vld, self.ds_diramp.shape)
        if f_shuffle:
            rng = np.random.default_rng()
            size = int(self.ds_diramp.shape[0])
            perm = np.arange(size)
            rng.shuffle(perm)
            print(perm[:10])
            # print(self.ds_diramp.mean(),self.ds_diramp.std(), np.median(self.ds_diramp))
            self.ds_diramp = self.ds_diramp[perm]
            # print(self.ds_diramp.mean(),self.ds_diramp.std(), np.median(self.ds_diramp))
        elif f_roll:
            offset = int((1 - ash_faction) * self.ds_diramp.shape[0])
            self.ds_diramp = np.roll(self.ds_diramp, offset, axis=0)
        check = np.sort(self.ds_diramp_ref.sum(axis=1))
        assert np.allclose(check, self.check)
        print("check val")
        print(check[10], self.check[10])
        self._define_healpix_pixel()
        # redefine hit
        self.ds_tra = self.ds_diramp[:fidx_vld]
        self._define_hit(self.hpix[:fidx_vld])
        # data set validation
        self.ds_vld = self.ds_diramp[fidx_vld:]
        self.fidx_vld = fidx_vld

    def estimate_relvolt(self, ampdir, method="ngp"):
        """From max volt and direction xmax estimate relative voltage expected

        :param diramp: v_ref, azimuth [deg], dist_zenith [deg]
        :param method:
        """
        #  check if pix associated to dir isn't empty
        #  use NGP method
        hpix = hp.ang2pix(self.nside, np.deg2rad(ampdir[:, 2]), np.deg2rad(ampdir[:, 1]))
        if method == "ngp":
            relvolt = griddata(
                self.ds_tra[:, 7:10], self.ds_tra[:, 1:7], ampdir, method="nearest", rescale=True
            )
        else:
            raise
        print(ampdir.shape, relvolt.shape)
        ds_hit = self.hit[hpix]
        print(ds_hit)
        return relvolt, ds_hit

    def get_residu_distrib(self, dataset):
        """For dataset with format "dirvolt-xx" file return distribution of residu"""
        relvolt, hit = self.estimate_relvolt(dataset[:, 7:10])
        idx_ok = np.argwhere(hit >= 4)
        nb_vlt = dataset.shape[0]
        if len(idx_ok) != nb_vlt:
            print(f"{nb_vlt-len(idx_ok)}/{nb_vlt} traces don't well defined with model")
            dataset_ok = np.squeeze(dataset[idx_ok])
            relvolt = np.squeeze(relvolt[idx_ok])
            print(relvolt.shape)
            hit = np.squeeze(hit[idx_ok])
        else:
            dataset_ok = dataset
        norm_dif = np.linalg.norm(relvolt - dataset_ok[:, 1:7], axis=-1)
        print(relvolt[:5])
        print(dataset_ok[:5, 1:7])
        print(norm_dif.shape)
        print(norm_dif.shape, norm_dif.mean(), norm_dif.max(), norm_dif.std())
        nb_bin = 250
        hist, bin_edges = np.histogram(norm_dif, nb_bin)
        dist_vld = hist / (bin_edges[1] - bin_edges[0])
        dist_vld /= dist_vld.sum()
        return norm_dif, bin_edges, dist_vld

    def get_true_error_distrib_validation(self):
        """For dataset validation return distribution of residu"""
        # loop on ds_val
        #   estimate_amplvolt
        #   compute true error and norm and store it
        # plot histogram normalize
        relvolt, hit = self.estimate_relvolt(self.ds_vld[:, 7:10])
        idx_ok = np.argwhere(hit >= 4)
        nb_vlt = self.ds_vld.shape[0]
        if len(idx_ok) != nb_vlt:
            print(f"validation {nb_vlt-len(idx_ok)}/{nb_vlt} traces don't well defined with model")
            self.ds_vld = np.squeeze(self.ds_vld[idx_ok])
            relvolt = np.squeeze(relvolt[idx_ok])
            print(relvolt.shape)
            hit = np.squeeze(hit[idx_ok])
        norm_dif = np.linalg.norm(relvolt - self.ds_vld[:, 1:7], axis=-1)
        print(relvolt[:5])
        print(self.ds_vld[:5, 1:7])
        print(norm_dif.shape)
        print(norm_dif.shape, norm_dif.mean(), norm_dif.max(), norm_dif.std())
        self.valid_done = True
        nb_bin = 250
        hist, bin_edges = np.histogram(norm_dif, nb_bin, (0, 0.6))
        dist_vld = hist / (bin_edges[1] - bin_edges[0])
        dist_vld /= dist_vld.sum()
        if False:
            plt.figure()
            size_bin = bin_edges[1]
            plt.plot(bin_edges[-2:] + size_bin / 2, dist_vld)
            # plt.ylim(0, 0.2)
            plt.xlim(0, 1.2)
            plt.yscale("log")
            plt.grid()
        self.valid_done = True
        return norm_dif, bin_edges, dist_vld

    def set_spline_model_validation(self, coef, surf_tot, emax):
        self.cspl = coef
        self.dist_surf_tot = surf_tot
        self.dist_emax = emax

    def compute_spline_model_validation(self, nb_rand=10):
        """From 'nb_rand' distributions of true error return the interpolate spline cubic

        Define :
            self.cspl
            self.dist_surf_tot
            self.dist_emax
        """
        norm_dif, bin_edges, dist_vld = self.get_true_error_distrib_validation()
        size_bin = bin_edges[1]
        nb_val = dist_vld.shape[0]
        print(nb_val, dist_vld.shape)
        bin_middle = bin_edges[:-1] + size_bin / 2
        assert bin_middle.shape[0] == nb_val
        print(bin_middle[-5:])
        print(nb_val, dist_vld.shape, bin_middle.shape)
        x_vec = np.zeros(bin_middle.shape[0] * nb_rand + 1, dtype=np.float64)
        y_vec = np.zeros_like(x_vec)
        w_vec = np.ones_like(x_vec)
        x_vec[:nb_val] = bin_middle
        print(x_vec[-5:])
        y_vec[:nb_val] = dist_vld
        plt.figure()
        plt.title("Histogram normalized and fit\ndataset validation with ~Jack Knife")
        plt.plot(
            bin_middle,
            dist_vld,
            label=f"rand 0 {dist_vld.sum()} , {dist_vld.shape[0]} {self.ds_vld.shape}",
        )
        for ite in range(1, nb_rand):
            self.partitioning_dataset(0.90, f_shuffle=True)
            _, _, dist_vld = self.get_true_error_distrib_validation()
            x_vec[ite * nb_val : (ite + 1) * nb_val] = bin_middle
            dist_vld[-2:] = 0
            y_vec[ite * nb_val : (ite + 1) * nb_val] = dist_vld
            plt.plot(
                bin_middle,
                dist_vld,
                label=f"rand {ite} {dist_vld.sum()}, {dist_vld.shape[0]} {self.ds_vld.shape}",
            )
        plt.xlabel("Norm of true error")
        # spline representation
        idx_sort = np.argsort(x_vec)
        x_vec = x_vec[idx_sort]
        y_vec = y_vec[idx_sort]
        # constraint origin to 0 and end of distribution to 0 too.
        w_vec[0] = 1000
        w_vec[-2] = 1000
        w_vec[-1] = 1000
        print(x_vec[:20])
        print(y_vec[:20])
        print(x_vec[-5:])
        print(y_vec[-5:])
        x_node = np.array(
            [0.003, 0.008, 0.01, 0.012, 0.015, 0.017, 0.02, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
        # x_node = np.array(
        #     [
        #         0.003,
        #         0.008,
        #         0.01,
        #         0.015,
        #         0.02,
        #         0.03,
        #         0.07,
        #         0.1,
        #         0.2,
        #         0.35,
        #         0.4,
        #         0.55
        #     ]
        # )
        self.dist_emax = 0.56
        idx_end = np.argwhere(x_vec > self.dist_emax)
        y_vec[idx_end] = 0
        w_vec[idx_end] = 1000
        self.cspl = splrep(x_vec, y_vec, w_vec, k=3, s=nb_val * ite, t=x_node)
        self.dist_surf_tot = splint(0, self.dist_emax, self.cspl)
        print("integral spline:", self.dist_surf_tot)
        p_01 = splint(0.01, self.dist_emax, self.cspl)
        print("proba 0.1 [%]: ", 100 * p_01 / self.dist_surf_tot)
        x_new = np.linspace(0, x_vec[-1], 1000)
        print(x_new[:5], x_new[-5:])
        s_y = splev(x_new, self.cspl)
        print(s_y[:5], s_y[-5:])
        plt.plot(x_new, s_y, "-*", color="k", label="Fit with cubic spline")
        plt.legend()
        plt.grid()
        #
        plt.figure()
        plt.title("Probability polar angle compatibility")
        cunul_y = self.estimate_proba_nresidu(x_new)
        # cunul_y = np.empty_like(x_new)
        # for idx, val in enumerate(x_new):
        #     cunul_y[idx] = splint(val, self.dist_emax, self.cspl)
        plt.plot(x_new, cunul_y)
        plt.ylabel("proba")
        plt.xlabel("Norm residu")
        plt.legend()
        plt.grid()
        # self.cspl = make_splrep(x_vec[idx_sort], y_vec[idx_sort], k=3, s=0.1)
        # ne fonctionne pas correctement même en définissant le nombre de noeuds nest
        # ou la liste des noeuds 't' ... ?

    def estimate_proba_evt(self, evt, f_plot=False):
        assert isinstance(evt, Handling3dTraces)
        # relative volatge
        rvolt = relative_voltage_evt(evt)
        # direction of Xmax
        print()
        assert evt.network.xmax_pos.shape == (3,)
        nb_tr = evt.get_nb_trace()
        dir_angle = np.empty((nb_tr, 2), dtype=np.float32)
        for idx in range(nb_tr):
            # direction toward source
            v_dux = evt.network.xmax_pos - evt.network.du_pos[idx]
            v_dux /= np.linalg.norm(v_dux)
            # azimuth, distance zenithal
            dir_angle[idx] = coord.nwu_cart_to_dir_one(v_dux)
        # to format dataset "dirvolt"
        dir_angle = np.rad2deg(dir_angle)
        print("dir_angle", dir_angle)
        dataset = np.zeros((nb_tr, 10), dtype=np.float32)
        dataset[:, 1:8] = rvolt
        dataset[:, 8] = dir_angle[:, 0]
        dataset[:, 9] = dir_angle[:, 1]
        model_relvolt, hit = self.estimate_relvolt(dataset[:, 7:10])
        print("HIT:", hit)
        idx_nok = np.argwhere(hit < 4)
        print(model_relvolt.shape)
        residu = model_relvolt - dataset[:, 1:7]
        norm_residu = np.linalg.norm(residu, axis=-1)
        proba_ash = self.estimate_proba_nresidu(norm_residu)
        proba_ash[idx_nok] = -10
        print("JMC", proba_ash.shape)
        print("norm_residu", norm_residu)
        print("proba_ash", proba_ash)
        if f_plot:
            evtc = evt.copy()
            evtc.network.plot_footprint_1d(
                proba_ash,
                f"Proba polar air shower compatibility",
                evt,
                "lin",
                "Proba",
            )
        return norm_residu, proba_ash

    def estimate_proba_nresidu(self, norm_res):
        if isinstance(norm_res, np.ndarray):
            out = np.zeros_like(norm_res)
            for idx in range(norm_res.shape[0]):
                out[idx] = splint(norm_res[idx], self.dist_emax, self.cspl)
                if out[idx] < 0:
                    out[idx] = 0
        else:
            out = max(0, splint(norm_res, self.dist_emax, self.cspl))

        return out / self.dist_surf_tot


def test_ModelDirectionVoltage():
    mdv = ModelDirectionVoltage()
    mdv.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    plt.figure()
    # plt.ylim(0, 0.2)
    nb_ite = 4
    for ite in range(nb_ite):
        mdv.partitioning_dataset(0.8, True)
        norm_dif, bin_edges, dist_vld = mdv.get_true_error_distrib_validation()
        size_bin = bin_edges[1]
        plt.plot(bin_edges[:-1] + size_bin / 2, dist_vld, label=f"random {ite}")
    plt.legend()
    plt.xlim(0, 1.2)
    # plt.yscale("log")
    plt.grid()
    plt.title(
        f"Relative voltage model along the Xmax direction\nTrue error distribution for {nb_ite} random validation dataset"
    )
    plt.xlabel(f"Norm true error, dataset validation is {mdv.ds_vld.shape[0]} traces")
    mdv.compute_spline_model_validation(10)
    x_ar = np.linspace(0, 0.6, 1000)
    plt.plot(x_ar, splev(x_ar, mdv.cspl), label="Spline model")
    plt.grid()


def dist_dataset90(pn_ds90):
    m_ds = np.load(pn_ds90)
    assert m_ds.shape[1] == 10
    mdv = ModelDirectionVoltage()
    mdv.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    _, bin_edges90, dist90 = mdv.get_residu_distrib(m_ds)
    delta90 = bin_edges90[1] - bin_edges90[0]
    mdv.partitioning_dataset(0.9, True)
    _, bin_edges, dist_vld = mdv.get_true_error_distrib_validation()
    deltadist = bin_edges[1] - bin_edges[0]
    plt.figure()
    plt.plot(
        bin_edges[1:],
        dist_vld,
        label=f"Air shower dataset validation, {mdv.ds_vld.shape[0]} traces, {dist_vld.sum():.1f}",
    )
    plt.plot(
        bin_edges90[1:],
        dist90,
        label=f"+90° to the polar angle of air shower, {m_ds.shape[0]} traces, {dist90.sum():.1f}",
    )
    plt.legend()
    # plt.xlim(0, 1.2)
    # plt.yscale("log")
    plt.grid()
    plt.title(f"Relative voltage model along the Xmax direction\nTrue error distribution")
    plt.xlabel("Norm true error")
    plt.ylabel("Normalized histogram (integral is 1)")


def test_spline():
    import numpy as np
    from scipy.interpolate import make_splrep

    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 16)
    rng = np.random.default_rng()
    y = np.sin(x) + 0.4 * rng.standard_normal(size=len(x))
    import matplotlib.pyplot as plt

    xnew = np.arange(0, 9 / 4, 1 / 50) * np.pi
    plt.plot(xnew, np.sin(xnew), "-.", label="sin(x)")
    plt.plot(xnew, make_splrep(x, y, s=0)(xnew), "-", label="s=0")
    plt.plot(xnew, make_splrep(x, y, s=len(x), nest=16)(xnew), "-", label=f"s={len(x)}")
    print(make_splrep(x, y, s=len(x), nest=16).t)
    plt.plot(x, y, "o")
    plt.legend()
    plt.show()


def check_estimate_proba_evt():
    import scipy.fft as sf

    PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"
    # PN_fmodel = "/sps/grand/colley/data/du_model/"
    # fix file model
    pn_tf_detector = PN_fmodel + "TF_RF_Chain_DC2.1rc.npy"
    pn_asd_galactic = PN_fmodel + "ASD_galaxy_ant_HFSS.npy"
    gresp = GalacticRespDetectorGenerator(pn_tf_detector, pn_asd_galactic)
    # Model proba
    p_model = ModelDirectionVoltage()
    p_model.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    p_model.partitioning_dataset(0.9, f_shuffle=True)
    p_model.compute_spline_model_validation(20)
    # Dataset training witestimate_proba_evthout shuffle
    rv_model = ModelDirectionVoltage()
    rv_model.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    rv_model.partitioning_dataset(0.9, f_shuffle=True)
    rv_model.set_spline_model_validation(p_model.cspl, p_model.dist_surf_tot, p_model.dist_emax)
    #
    path_asdf = "/home/jcolley/projet/lucky/data/v2/"
    # f_ash = path_asdf + "volt-ash_39-24951.asdf"
    # f_ash = path_asdf + "volt-bgk-rnd_0-24984.asdf"
    #f_ash = path_asdf + "volt-bgk-90_0-24984.asdf"
    f_ash = path_asdf + "volt-ash_46-24976.asdf"
    events = f_tr.AsdfReadTraces(f_ash)
    # test filter
    evt = events.get_event(120)
    assert isinstance(evt, Handling3dTraces)
    gresp.set_paramters_simu(evt.f_samp_mhz[0], evt.get_size_trace())
    gresp.add_galactic_component(evt)
    evt.plot_footprint_val_max()
    evt_filter = evt.copy()
    assert isinstance(evt_filter, Handling3dTraces)
    evt_filter.apply_bandpass(20, 100, True)
    evt_filter.plot_footprint_val_max()
    return 

    nb_evt = 1
    for idx in range(nb_evt):
        evt = events.get_event(120 + idx)
        assert isinstance(evt, Handling3dTraces)
        rv_model.estimate_proba_evt(evt, True)
    evt1 = evt.copy()
    evt1.get_tmax_vmax()
    evt1.plot_footprint_val_max()
    #
    f_s = evt.f_samp_mhz[0] / 1e-6
    a_freq = sf.rfftfreq(evt.get_size_trace(), 1 / f_s) * 1e-6
    for idx in range(nb_evt):
        evtn = events.get_event(120 + idx)
        gresp.set_paramters_simu(a_freq, evtn.get_size_trace(), evtn.get_nb_trace())
        noise = gresp.get_galactic_traces(18)
        print(noise.shape)
        print(evtn.traces.shape)
        evtn.traces += noise
        rv_model.estimate_proba_evt(evtn, True)
    evtn.get_tmax_vmax()
    evtn.plot_footprint_val_max()


if __name__ == "__main__":
    # path_asdf = "/home/jcolley/projet/lucky/data/"
    # f_ash = "volt-ash_39-24951.asdf"
    # pn_ash = path_asdf + f_ash
    # dirv = DirectionVoltageParameters(pn_ash)
    # dirv.process_events(0, -1)
    # test_spline()
    np.random.seed(10)
    check_estimate_proba_evt()
    # test_ModelDirectionVoltage()
    # dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bkg-90_0-24984.npy")
    # dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bkg-90_5-24938.npy")
    # dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bkg-90_9-24930.npy")
    plt.show()

    # def process_all_events_parallel_chunk(self, ie_beg, ie_endp1, size_chk=10):
    #     from joblib import Parallel, delayed, parallel_config
    #
    #     # manage index begin, end
    #     self.size_chk = size_chk
    #     f_ef = f_tr.AsdfReadTraces(self.pn_vash, False)
    #     if ie_endp1 < 0:
    #         ie_endp1 = f_ef.get_nb_events()
    #     self.ie_endp1 = ie_endp1
    #
    #     START = datetime.now()
    #     # load balancing: nb chunk to process
    #     nb_evt = ie_endp1 - ie_beg
    #     print(nb_evt, ie_endp1)
    #     n_chk = nb_evt // size_chk
    #     if nb_evt % size_chk:
    #         n_chk += 1  # add 1 for the rest
    #     # broadcast with joblib
    #     n_jobs = 1
    #     if n_jobs > 1:
    #         parallel_config(n_jobs=4, backend="loky", inner_max_num_threads=1, return_as="list")
    #         func_process = self.process_chunk_file
    #         results = Parallel()(delayed(func_process)(ie_beg + i * size_chk) for i in range(n_chk))
    #     else:
    #         print("1 CPU")
    #         size_chk = ie_endp1 - ie_beg
    #         self.size_chk = size_chk
    #         l_events = self.process_chunk_file(ie_beg)
    #         results = [l_events]
    #     # collect result
    #     idx_beg, idx_end = f_ef.get_chunk_event_interval(ie_beg, ie_endp1-1)
    #     nb_tr = idx_end - idx_beg
    #     dataset = np.zeros((nb_tr, 10), dtype=np.float32)
    #     i_beg = 0
    #     i_evt = 0
    #     for ret in results:
    #         for res in ret:
    #             nb_du = res[0].shape[0]
    #             i_end = i_beg + nb_du
    #             dataset[i_beg:i_end, 0] = i_evt * np.ones(nb_du)
    #             dataset[i_beg:i_end, 1:8] = res[0]
    #             dataset[i_beg:i_end, 8] = res[1]
    #             dataset[i_beg:i_end, 9] = res[2]
    #             i_beg = i_end
    #             i_evt += 1
    #     print(nb_tr, i_evt, i_end)
    #     print(dataset[:,7].mean(), dataset[:,7].std())
    #     in_f = pathlib.Path(self.pn_vash)
    #     out_name = str(in_f.name).replace("volt", "dirvolt")
    #     out_name = out_name.replace(".asdf", "")
    #     np.save(self.out_dir + out_name, dataset)
    #     logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
    #     print(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
