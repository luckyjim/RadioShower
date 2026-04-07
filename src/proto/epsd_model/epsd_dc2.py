"""
Created on 25 nov. 2025

@author: jcolley
"""

from pathlib import Path
from logging import getLogger
import sys
import re

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

from rshower.basis.traces_event import Handling3dTraces
import rshower.io.events.asdf_traces as f_tr
from rshower.simu.gal_resp import GalacticRespDetectorGenerator
from rshower.model.psd_efield import AirShowerEfieldPSDmodel, modelPSD_4params
from rshower.basis.traces_event import Handling3dTraces

logger = getLogger

PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"
# PN_fmodel = "/sps/grand/colley/data/du_model/"
# fix file model
pn_tf_detector = PN_fmodel + "TF_RF_Chain_DC2.1rc.npy"
pn_asd_galactic = PN_fmodel + "ASD_galaxy_ant_HFSS.npy"


s_epsd = [
    ("azi", "f4"),
    ("d_zen", "f4"),
    ("vmax", "f4"),
    ("d_XC", "f4"),  # distance Xmax Core
    ("p_psd", "f4", (5)),  # M, a, alpha, sigma, residu
]


def proto_vmax():
    f_volt = f_tr.AsdfReadTraces("/home/jcolley/projet/lucky/data/volt-ash_39-24951.asdf")
    i_e = 301
    volt = f_volt.get_event(i_e)
    volt.plot_footprint_val_max()
    volt_n = volt.copy()
    assert isinstance(volt_n, Handling3dTraces)
    volt_n.name += "with noise"
    gresp = GalacticRespDetectorGenerator(pn_tf_detector, pn_asd_galactic)
    gresp.set_paramters_simu(500, 1024)
    gresp.add_galactic_component(volt_n, 18)
    volt_n.plot_footprint_val_max()
    volt_n.get_tmaxd_XC_vmax()
    idx = 18
    volt.plot_trace_idx(idx, to_draw="0")
    a3_trace = volt_n.traces[idx]
    hilbert_amp = np.abs(hilbert(a3_trace, axis=-1))
    print(hilbert_amp.shape)
    volt_n.plot_trace_idx(idx, to_draw="012")
    plt.plot(volt_n.t_samples[idx], hilbert_amp[0], label="Hilbert 0")
    norm_hilbert_amp = np.linalg.norm(hilbert_amp, axis=0)
    plt.plot(volt_n.t_samples[idx], norm_hilbert_amp, "-.", label="Hilbert all")
    plt.legend()
    m_h = np.mean(norm_hilbert_amp[:200])
    c_sum = (norm_hilbert_amp - m_h).cumsum()
    a_x = np.arange(1, 200)
    m_p = np.mean(c_sum[1:200] / a_x)
    droite = m_p * np.arange(1, 1025)
    plt.figure()
    plt.plot(volt_n.t_samples[idx], c_sum)
    plt.plot(volt_n.t_samples[idx], droite)
    plt.grid()
    plt.figure()
    plt.plot(volt_n.t_samples[idx], c_sum - droite)
    plt.grid()


def proto_duration():
    """duration trop  bruite"""
    f_volt = f_tr.AsdfReadTraces("/home/jcolley/projet/lucky/data/volt-ash_39-24951.asdf")
    i_e = 303
    volt = f_volt.get_event(i_e)
    # volt.plot_footprint_val_max()
    volt_n = volt.copy()
    assert isinstance(volt_n, Handling3dTraces)
    volt_n.name += "with noise"
    gresp = GalacticRespDetectorGenerator(pn_tf_detector, pn_asd_galactic)
    gresp.set_paramters_simu(500, 1024)
    gresp.add_galactic_component(volt_n, 18)
    # volt_n.apply_lowpass(150)
    volt_n.plot_footprint_val_max()
    volt_n.get_tmax_vmax()
    idx = 0
    volt.plot_trace_idx(idx, to_draw="0")
    a3_trace = volt_n.traces[idx]
    hilbert_amp = np.abs(hilbert(a3_trace, axis=-1))
    print(hilbert_amp.shape)
    volt_n.plot_trace_idx(idx, to_draw="012")
    plt.plot(volt_n.t_samples[idx], hilbert_amp[0], label="Hilbert 0")
    norm_hilbert_amp = np.linalg.norm(hilbert_amp, axis=0)
    plt.plot(volt_n.t_samples[idx], norm_hilbert_amp, "-.", label="Hilbert all")
    plt.legend()
    # noise
    noise = np.std(a3_trace[:150], axis=1)
    print(noise)
    assert noise.shape == (3,)
    max_sig = np.max(noise)
    # max Hilbert
    i_max = np.argmax(norm_hilbert_amp)
    print(i_max)
    # search before
    threshold = 2 * max_sig
    print(threshold)
    i_left = i_max - 1
    while True:
        print(norm_hilbert_amp[i_left])
        if norm_hilbert_amp[i_left] < threshold:
            break
        i_left -= 1
    i_right = i_max + 1
    while True:
        if norm_hilbert_amp[i_right] < threshold:
            break
        i_right += 1
    # result
    nb_sple = i_right - i_left
    delta_ns = 1e3 / volt_n.f_samp_mhz[0]
    tau_ns = nb_sple * delta_ns
    print("duration [ns]:", tau_ns)
    acorel = np.correlate(norm_hilbert_amp, norm_hilbert_amp, "same")
    plt.figure()
    plt.plot(volt_n.t_samples[idx], acorel)
    plt.grid()


def model_epsf_test():
    model_epsf("/home/jcolley/projet/lucky/data/efield_39-24951.asdf")


def model_epsf(pn_efield):
    f_ef = f_tr.AsdfReadTraces(pn_efield)
    pnv = pn_efield.replace("efield", "volt-ash")
    f_volt = f_tr.AsdfReadTraces(pnv)
    d_xc = np.linalg.norm(f_volt.events["xmax_nwu"] - f_volt.events["core_nwu"], axis=1)
    d_xc /= 1000
    print(d_xc.shape)
    nb_traces_all = f_ef.meta["nb_traces"]
    dataset = np.zeros(nb_traces_all, dtype=s_epsd)
    print(dataset.shape)
    i_tr = 0
    epsd = AirShowerEfieldPSDmodel()
    nb_evts = f_ef.meta["nb_evts"]
    for i_e in range(nb_evts):
        evt = f_ef.get_event(i_e)
        volt = f_volt.get_event(i_e)
        epsd.set_efield(evt)
        epsd.fit_all()
        nb_traces = evt.get_nb_trace()
        print(f"{i_e}/{nb_evts}, nb trace : {nb_traces}")
        idx_r = range(i_tr, i_tr + nb_traces)
        dataset["p_psd"][idx_r] = epsd.fit_pars
        v_max = volt.get_max_norm()
        dataset["vmax"][idx_r] = v_max
        dataset["d_XC"][idx_r] = d_xc[i_e]
        i_tr += nb_traces
    #
    i_nok = np.argwhere(dataset["p_psd"] == -1).squeeze()
    print(f"Nb nok: {len(i_nok)}/{i_tr}")
    pnv = pn_efield.replace("efield", "volt-ash")
    f_volt = f_tr.AsdfReadTraces(pnv)
    print(f_volt.mtraces.shape)
    dataset["azi"] = f_volt.mtraces["azi"]
    dataset["d_zen"] = f_volt.mtraces["d_zen"]
    # i_ok = np.argwhere(dataset[:, 2] != -1).squeeze()
    pn_in = Path(pn_efield)
    pn_out = pn_in.parent / pn_in.stem.replace("efield", "epsd")
    print(pn_out)
    np.save(pn_out, dataset)


def script_epsd():
    pn_efield = sys.argv[1]
    model_epsf(pn_efield)


def check_EfieldModelDataset():
    emodel = EfieldModelDataset()
    emodel.init_collect_dataset("/home/jcolley/projet/lucky/data/v2")
    emodel.partitioning_dataset(0.9)
    idx = 1010
    evt = emodel.get_traces(idx + emodel.fidx_vld)
    evt.plot_trace_idx(0)
    psdmodel = AirShowerEfieldPSDmodel()
    psdmodel.set_efield(evt)
    psdmodel.fit_idx(0, True)
    print(emodel.ds_vld[idx])
    print(emodel.ds_vld["p_psd"][idx] ** 2)
    params = np.asarray(emodel.ds_vld[idx].tolist()[:4])
    ind = emodel.estimate_idx_epsd_vec(params)
    print(ind)
    freqs = np.arange(1, 500)
    plt.figure()
    m_p, a_p, alpha_p, sigma, _ = emodel.ds_vld["p_psd"][idx]
    plt.plot(freqs, modelPSD_4params(freqs, m_p, a_p, alpha_p, sigma), label="True")
    m_p, a_p, alpha_p, sigma, _ = emodel.ds_tra["p_psd"][ind[0]]
    plt.plot(freqs, modelPSD_4params(freqs, m_p, a_p, alpha_p, sigma), label="Guess 1")
    m_p, a_p, alpha_p, sigma, _ = emodel.ds_tra["p_psd"][ind[1]]
    plt.plot(freqs, modelPSD_4params(freqs, m_p, a_p, alpha_p, sigma), "-.", label="Guess 2")
    plt.grid()
    plt.semilogy()
    plt.legend()
    print(emodel.ds_tra[ind])


class EfieldModelDataset:

    def __init__(self):
        pass

    def init_collect_dataset(self, ds_dir, exclude_file=""):
        self.ds_dir = ds_dir
        pattern = re.compile(r"^epsd")
        rep = Path(ds_dir)
        ds_epsd = None
        cpt_file = 0
        l_file2idx = [0]
        l_name = []
        for m_f in rep.iterdir():
            if m_f.is_file() and pattern.search(m_f.name):
                if exclude_file:
                    if exclude_file in m_f.stem:
                        print(f"Skip {m_f}")
                        continue
                print(m_f)
                f_ds = str(m_f.absolute())
                m_ds = np.load(f_ds)
                pn_ef = str(m_f).replace("epsd", "efield")
                pn_ef = pn_ef.replace(".npy", ".asdf")
                l_name.append(pn_ef)
                if ds_epsd is None:
                    ds_epsd = m_ds
                    l_file2idx.append(len(m_ds))
                else:
                    l_file2idx.append(l_file2idx[-1] + len(m_ds))
                    ds_epsd = np.hstack((ds_epsd, m_ds))
                cpt_file += 1

        self.true_idx = np.arange(ds_epsd.shape[0])
        # remove fit failed
        print(len(self.true_idx))
        i_ok = np.argwhere(ds_epsd["p_psd"][:, 0] != -1).squeeze()
        print(len(i_ok))
        ds_epsd = ds_epsd[i_ok]
        self.true_idx = self.true_idx[i_ok]
        # remove fit with hight residuparams
        i_ok = np.argwhere(ds_epsd["p_psd"][:, 4] < 1.6).squeeze()
        print(len(i_ok))
        ds_epsd = ds_epsd[i_ok]
        self.true_idx = self.true_idx[i_ok]
        print(l_name)
        print(l_file2idx)
        self.nf_efield = l_name
        self.l_file2idx = l_file2idx
        self.ds_epsd = ds_epsd

    def partitioning_dataset(self, trainnig_frac):
        assert trainnig_frac <= 1.0
        fidx_vld = int(self.ds_epsd.shape[0] * trainnig_frac)
        self.ds_tra = self.ds_epsd[:fidx_vld]
        self.ds_vld = self.ds_epsd[fidx_vld:]
        self.fidx_vld = fidx_vld

    def estimate_psd(self, freqs_mhz, azi, d_zen, vmax, d_XC):
        """

        :param azi: deg
        :param d_zen: deg
        :param vmax:
        :param d_XC: kma_idx
        """
        a_idx = self.estimate_idx_psd(azi, d_zen, vmax, d_XC)
        print(a_idx.shape)
        m_p, a_p, alpha_p, sigma, _ = self.ds_tra["p_psd"][a_idx[0]]
        return modelPSD_4params(freqs_mhz, m_p, a_p, alpha_p, sigma)

    def estimate_idx_psd(self, azi, d_zen, vmax, d_XC):
        """

        :param azi:
        :param d_zen:
        :param vmax:
        :param d_XC:
        """
        params = np.array([azi, d_zen, vmax, d_XC])
        return self.estimate_idx_epsd_vec(params)

    def estimate_idx_epsd_vec(self, params):
        """
        :param params:
        return: index
        """
        print(f"par in epsd : {params}")
        dataset = np.column_stack(
            [
                self.ds_tra["azi"],
                self.ds_tra["d_zen"],
                self.ds_tra["vmax"] / 100,
                self.ds_tra["d_XC"],
                np.arange(len(self.ds_tra["d_XC"]), dtype=np.int64),
            ]
        )
        print(dataset[:, :2].shape)
        tree = BallTree(dataset[:, :2], leaf_size=10)
        ppars = params[:2][None, :]
        print(ppars.shape)
        aind = tree.query_radius(ppars, r=4.0)
        print(aind[0].shape)
        dataset = dataset[aind[0]]
        tree = BallTree(dataset[:, :4], leaf_size=10)
        ppars = params
        ppars[2] /= 100
        dist, ind = tree.query(params[None, :], k=4)
        print(ind)
        print(dataset[ind[0]])
        return dataset[ind[0]][:, 4].astype(np.int64)

    def get_ref_pulse(self, idx):
        t_idx = self.true_idx[idx]
        idx_infile = t_idx - self.l_file2idx[-1]
        for idx, i_beg in enumerate(self.l_file2idx):
            if i_beg > t_idx:
                idx_infile = t_idx - self.l_file2idx[idx - 1]
                break
        return self.nf_efield[idx - 1], idx_infile

    def get_traces(self, idx):
        nf_efield, idx_trace = self.get_ref_pulse(idx)
        f_ef = f_tr.AsdfReadTraces(nf_efield)
        print(f_ef.traces.shape)
        traces = f_ef.traces[idx_trace, 0]
        evt = Handling3dTraces("Efield trace")
        tr = np.zeros((1, 3, len(traces)), dtype=np.float32)
        print(tr.shape)
        tr[0, 0] = traces
        evt.init_traces(tr, f_samp_mhz=2000)
        evt.set_unit_axis(r"$\mu V/m$", "pol", f"Efield ")
        return evt


if __name__ == "__main__":
    # proto_duration()
    # model_epsf_test()
    # script_epsd()
    check_EfieldModelDataset()
    plt.show()
