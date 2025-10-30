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
from scipy.interpolate import griddata

from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
import proto.simu_dc2.asdf_traces as f_tr


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
    # assert np.all(rel_opti >= -1) See you soon!
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
        self.valid_done = False

    def _define_healpix_pixel(self):
        rad_dir = np.deg2rad(self.ds_diramp[:, 8:]).T
        self.hpix = hp.ang2pix(self.nside, rad_dir[1], rad_dir[0])

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
            print(m_f)
            if m_f.is_file() and pattern.search(m_f.name):
                f_ds = str(m_f.absolute())
                m_ds = np.load(f_ds)
                if ds_diramp is None:
                    ds_diramp = m_ds
                else:
                    ds_diramp = np.vstack((ds_diramp, m_ds))
                cpt_file += 1
                print(ds_diramp.shape)
                # if cpt_file == 9:
                #     break
        self.ds_diramp = ds_diramp
        self.ds_tra = self.ds_diramp
        self._define_healpix_pixel()
        self._define_hit(self.hpix)

    def partitioning_dataset(self, ash_faction, f_shuffle=False):
        assert ash_faction <= 1.0
        fidx_vld = int(self.ds_diramp.shape[0] * ash_faction)
        if f_shuffle:
            perm = np.arange(self.ds_diramp.shape[0])
            np.random.shuffle(perm)
            print(perm[:10])
            #print(self.ds_diramp.mean(),self.ds_diramp.std(), np.median(self.ds_diramp))
            self.ds_diramp = self.ds_diramp[perm]
            #print(self.ds_diramp.mean(),self.ds_diramp.std(), np.median(self.ds_diramp))
        self._define_healpix_pixel()
        # redefine hit
        self.ds_tra = self.ds_diramp[:fidx_vld]
        self._define_hit(self.hpix[:fidx_vld])
        # data set validation
        self.ds_vld = self.ds_diramp[fidx_vld:]
        self.fidx_vld = fidx_vld

    def estimate_relvolt(self, ampdir, method="ngp"):
        """
        :param diramp: v_ref, azimuth [deg], dist_zenith [deg]
        :param method:
        """
        #  check if pix associated to dir isn't empty
        #  use NGP method
        print()
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

    def estimate_dist(self, dataset):
        relvolt, hit = self.estimate_relvolt(dataset[:, 7:10])
        idx_ok = np.argwhere(hit >= 4)
        nb_vlt = dataset.shape[0]
        if len(idx_ok) != nb_vlt:
            print(f"{nb_vlt-len(idx_ok)} traces don't well defined with model")
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
        self.valid_done = True
        nb_bin = 100
        hist, bin_edges = np.histogram(norm_dif, nb_bin)
        dist_vld = hist / hist.sum()
        return norm_dif, bin_edges, dist_vld

    def validation_dist(self):
        # loop on ds_val
        #   estimate_amplvolt
        #   compute true error and norm and store it
        # plot histogram normalize
        relvolt, hit = self.estimate_relvolt(self.ds_vld[:, 7:10])
        idx_ok = np.argwhere(hit >= 4)
        nb_vlt = self.ds_vld.shape[0]
        if len(idx_ok) != nb_vlt:
            print(f"{nb_vlt-len(idx_ok)} traces don't well defined with model")
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
        nb_bin = 100
        hist, bin_edges = np.histogram(norm_dif, nb_bin)
        dist_vld = hist / hist.sum()
        if False:
            plt.figure()
            plt.plot(bin_edges[1:], dist_vld)
            # plt.ylim(0, 0.2)
            plt.xlim(0, 1.2)
            plt.yscale("log")
            plt.grid()
        return norm_dif, bin_edges, dist_vld

    def estimate_proba(self, dir_xmax, relvolt):
        assert self.valid_done


def test_ModelDirectionVoltage():
    mdv = ModelDirectionVoltage()
    mdv.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    plt.figure()
    # plt.ylim(0, 0.2)
    nb_ite = 10
    for ite in range(nb_ite):
        mdv.partitioning_dataset(0.8, True)
        norm_dif, bin_edges, dist_vld = mdv.validation_dist()
        plt.plot(bin_edges[1:], dist_vld, label=f"random {ite}")
    plt.legend()
    plt.xlim(0, 1.2)
    #plt.yscale("log")
    plt.grid()
    plt.title(
        f"Relative voltage model along the Xmax direction\nTrue error distribution for {nb_ite} random validation dataset"
    )
    plt.xlabel(f"Norm true error, dataset validation is {mdv.ds_vld.shape[0]} traces")


def dist_dataset90(pn_ds90):
    m_ds = np.load(pn_ds90)
    assert m_ds.shape[1] == 10
    mdv = ModelDirectionVoltage()
    mdv.init_collect_dataset("/home/jcolley/projet/lucky/data/v2/")
    _, bin_edges90, dist90 = mdv.estimate_dist(m_ds)
    mdv.partitioning_dataset(0.8, True)
    _, bin_edges, dist_vld = mdv.validation_dist()
    plt.figure()
    plt.plot(bin_edges[1:], dist_vld, label=f"Air shower dataset validation, {mdv.ds_vld.shape[0]} traces")
    plt.plot(bin_edges90[1:], dist90, label=f"+90° to the polar angle of air shower, {m_ds.shape[0]} traces")
    plt.legend()
    #plt.xlim(0, 1.2)
    #plt.yscale("log")
    plt.grid()
    plt.title(
        f"Relative voltage model along the Xmax direction\nTrue error distribution"
    )
    plt.xlabel("Norm true error")
      
    


if __name__ == "__main__":
    # path_asdf = "/home/jcolley/projet/lucky/data/"
    # f_ash = "volt-ash_39-24951.asdf"
    # pn_ash = path_asdf + f_ash
    # dirv = DirectionVoltageParameters(pn_ash)
    # dirv.process_events(0, -1)
    np.random.seed(10)
    test_ModelDirectionVoltage()
    dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bgk-90_0-24984.npy")
    dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bgk-90_5-24938.npy")
    dist_dataset90("/home/jcolley/projet/lucky/data/v2/dirvolt-bgk-90_9-24930.npy")
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
