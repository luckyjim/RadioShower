"""
Created on 25 mai 2025

@author: jcolley
"""
from logging import getLogger
import logging
import pprint
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import asdf
from astropy.time import Time

import grand.dataio.root_files as froot

from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
from proto.simu_dc2.du_resp import SimuDetectorUnitResponse

from datetime import datetime


logger = getLogger(__name__)


def get_efield_ref_values(tref):
    assert isinstance(tref, HandlingEfield)
    # tref.remove_trace_low_signal(60)
    tref_gd = tref.copy()
    assert isinstance(tref_gd, HandlingEfield)
    tref_gd.apply_bandpass(40, 210, causal=False)
    # l_ok = tref_gd.remove_trace_low_signal(60)
    # tref.keep_only_trace_with_index(l_ok)
    t_max, v_max = tref.get_tmax_vmax(hilbert=False, interpol="parab")
    t_gd_max, v_gd_max = tref_gd.get_tmax_vmax(hilbert=True, interpol="parab")
    polars, _, _ = tref.get_polar_angle()
    # logger.info(tref.traces.shape)
    # logger.info(tref.f_samp_mhz.shape)
    freq, psd = get_psd(tref.ef_pol[0], tref.f_samp_mhz[0])
    psd_all = np.empty((tref.get_nb_trace(), len(psd)), dtype=psd.dtype)
    for i_du in range(tref.get_nb_trace()):
        freq, psd_all[i_du] = get_psd(tref.ef_pol[i_du], tref.f_samp_mhz[0])
    # logger.info(freq.shape)
    # logger.info(psd_all.shape)
    delta_f = float(freq[1])
    i_2pt = (np.array([110, 190]) / delta_f).astype(np.int32)
    freq_2pt = freq[i_2pt]
    psd_2pt = psd_all[:, i_2pt]
    # logger.info(freq_2pt)
    # logger.info(psd_2pt[:2])
    #logger.info(np.rad2deg(polars))
    return [t_max, v_max, t_gd_max, v_gd_max, polars, freq_2pt, psd_2pt]


class DataSimuFile:
    def __init__(self):
        pass

    def init_type(self, s_trace):
        self.d_asdf = {}
        d_meta = {}
        d_meta["description"] = "PSF study"
        d_meta["version"] = "0.1"
        d_meta["date"] = Time.now().to_value("isot", subfmt="date_hm")
        d_meta["author"] = "Jean-Marc Colley"
        d_meta["comment"] = ""
        d_meta["history"] = ""
        self.d_asdf["meta"] = d_meta

        self.dtype_traces = [
            ("du_id", "i4"),
            ("tr_sig", "f4", (3, s_trace)),
            ("tr_noise", "f4", (3, s_trace)),
            ("start_s", "i8"),
            ("start_ns", "f8"),
        ]
        self.dtype_efref = [
            ("polar_angle", "f4"),
            ("tmax_ns", "i8"),
            ("emax", "f4"),
            ("tmax_ns_band", "i8"),
            ("emax_band", "f4"),
            ("freq_psd", "f4", (2)),
            ("psd", "f4", (2)),
        ]
        self.dtype_events = [
            ("idx_end", "i4"),
            ("run_nb", "i4"),
            ("event_nb", "i4"),
            ("idx", "i4"),
            ("energy", "f4"),
            ("xmax_nwu", "f4", (3)),
            ("core_nwu", "f4", (3)),
        ]
        self.dtype_network = [
            ("du_id", "i4"),
            ("pos_nwu", "f4", (3)),
        ]

    def _create_arrays(self, nb_traces, nb_evts, nb_du=400):
        self.traces = np.zeros(nb_traces, dtype=self.dtype_traces)
        self.efield = np.zeros(nb_traces, dtype=self.dtype_efref)
        self.events = np.zeros(nb_evts, dtype=self.dtype_events)
        self.network = np.zeros(nb_du, dtype=self.dtype_network)

    def read_asdf(self, pn_simu):
        self.d_asdf = asdf.open(pn_simu)
        self.traces = self.d_asdf["data"]["traces"]
        self.events = self.d_asdf["data"]["events"]
        self.network = self.d_asdf["data"]["network"]
        #self.efield = self.d_asdf["data"]["efield"]
        self.efield = self.d_asdf["data"]["ef_ref"]
        self.idt2idx = {idt: idx for idx, idt in enumerate(self.network["du_id"])}

    def get_event(self, idx_evt, kind="measure"):
        if idx_evt == 0:
            idx_beg = 0
        else:
            idx_beg = self.events["idx_end"][idx_evt - 1]
        idx_end = self.events["idx_end"][idx_evt]
        logger.info(idx_beg)
        logger.info(idx_end)
        if kind == "noise":
            traces = self.traces["tr_noise"][idx_beg:idx_end]
        elif kind == "sig":
            traces = self.traces["tr_sig"][idx_beg:idx_end]
        elif kind == "measure":
            traces = self.traces["tr_sig"][idx_beg:idx_end]
            traces += self.traces["tr_noise"][idx_beg:idx_end]
        #
        evt_id = f"IDX={self.events['idx'][idx_evt]}, EVT_NB={self.events['event_nb'][idx_evt]}, RUN_NB={self.events['run_nb'][idx_evt]}"
        event = Handling3dTraces(evt_id)
        event.init_traces(
            traces,
            self.traces["du_id"][idx_beg:idx_end],
            self.traces["start_ns"][idx_beg:idx_end],
            self.d_asdf["info"]["f_samp_mhz"],
        )
        dist_xmax = np.linalg.norm(self.events["xmax_nwu"][idx_evt]) / 1000
        event.info_shower = f"||xmax_pos_shc||={dist_xmax:.1f} km;"
        unit = self.d_asdf["info"]["unit"]
        l_idt = list(self.traces["du_id"][idx_beg:idx_end])
        l_idx = [self.idt2idx[idt] for idt in l_idt]
        pos_du = self.network["pos_nwu"][l_idx]
        event.init_network(pos_du)
        event.network.name = self.d_asdf["info"]["network"]
        event.set_unit_axis(unit, "dir", "Trace")
        event.network.xmax_pos = self.events["xmax_nwu"][idx_evt]
        event.network.core_pos = self.events["core_nwu"][idx_evt]
        efield = self.efield[idx_beg:idx_end]
        logger.info(np.rad2deg(efield["polar_angle"]))
        return event, efield

    def upload_all_events(self, l_events, nb_trace):
        """
        l_events = [[event_signal, event_noise, dict_params_simu] ]
        """
        self._create_arrays(nb_trace, len(l_events))
        i_beg = 0
        idt2idx = {}
        idx_du = 0
        # info
        self.d_info = {}
        for idx, evt in enumerate(l_events):
            logger.info(evt[2])
            sig = evt[0]
            noise = evt[1]
            eref = evt[3]
            assert isinstance(sig, Handling3dTraces)
            assert isinstance(noise, Handling3dTraces)
            i_end = i_beg + sig.get_nb_trace()
            # traces
            self.traces["tr_sig"][i_beg:i_end] = sig.traces
            self.traces["tr_noise"][i_beg:i_end] = noise.traces
            self.traces["start_ns"][i_beg:i_end] = sig.t_start_ns
            self.traces["du_id"][i_beg:i_end] = sig.idx2idt
            # Efield
            self.efield["polar_angle"][i_beg:i_end] = eref[4]
            self.efield["tmax_ns"][i_beg:i_end] = eref[0]
            self.efield["emax"][i_beg:i_end] = eref[1]
            self.efield["tmax_ns_band"][i_beg:i_end] = eref[2]
            self.efield["emax_band"][i_beg:i_end] = eref[3]
            self.efield["freq_psd"][i_beg:i_end] = eref[5]
            self.efield["psd"][i_beg:i_end] = eref[6]
            # events
            info = evt[2]
            self.events["idx_end"][idx] = i_end
            self.events["run_nb"][idx] = info["run_nb"]
            self.events["event_nb"][idx] = info["event_nb"]
            self.events["idx"][idx] = info["idx"]
            self.events["xmax_nwu"][idx] = sig.network.xmax_pos
            self.events["core_nwu"][idx] = sig.network.core_pos
            self.events["energy"][idx] = info["energy"]
            # network
            for idt, pos in zip(sig.network.idx2idt, sig.network.du_pos):
                if idt in idt2idx.keys():
                    i_du = idt2idx[idt]
                    assert np.allclose(self.network["pos_nwu"][i_du], pos)
                else:
                    idt2idx[idt] = idx_du
                    self.network["du_id"][idx_du] = idt
                    self.network["pos_nwu"][idx_du] = pos
                    idx_du += 1
            i_beg = i_end
        self.idt2idx = idt2idx
        self.d_info = {}
        self.d_info["f_samp_mhz"] = sig.f_samp_mhz[0]
        self.d_info["unit"] = sig.unit_trace
        self.d_info["network"] = sig.network.name

    def save_asdf(self, pn_simu):
        self.d_asdf["info"] = self.d_info
        self.d_asdf["data"] = {}
        self.d_asdf["data"]["traces"] = self.traces
        self.d_asdf["data"]["events"] = self.events
        self.d_asdf["data"]["network"] = self.network
        self.d_asdf["data"]["efield"] = self.efield
        file_simu = asdf.AsdfFile(self.d_asdf)
        # file_simu.write_to(pn_simu, all_array_compression="zlib")
        file_simu.write_to(pn_simu)


# class SimuGrand_DC2_1:
#     def __init__(self):
#         self.voltage = DataSimuToFile()
#         self.simu = SimuDetectorUnitResponse()
#
#     def to_voltage(self, pn_efield, nb_evt = 1000):
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         f_ef = froot.get_file_event(pn_efield)
#         for i_e in range(nb_evt):
#             i_e +=200
#             f_ef.load_event_idx(i_e)
#             tref_gd = f_ef.get_obj_handling3dtraces()
#             d_simu = f_ef.get_simu_parameters()
#             logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#             tref = convert_3dtrace_grandlib(tref_gd, True)
#             tref.network.xmax_pos = d_simu["FIX_xmax_pos"]
#             tref.network.core_pos = d_simu["shower_core_pos"]
#             # check nb DU ok
#             tref.remove_trace_low_signal(70)
#             if tref.get_nb_trace() < 4:
#                 logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#                 continue
#             tref.plot_footprint_val_max()
#             self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#             self.simu.set_data_efield(tref)
#             self.simu.compute_du_all()
#             # Check in voltage
#             trsig = tref.copy(self.simu.v_out)
#             trsig.name = "signal no noise"
#             trsig.set_unit_axis( "Volt", "dir", "Voltage")
#             trsig.plot_footprint_val_max()
#             trnoise = tref.copy(self.simu.v_noise)
#             trnoise.name = "galactic 'noise'"
#             trnoise.set_unit_axis( "Volt", "dir", "Voltage")
#             trnoise.plot_footprint_val_max()
#             # like a trigger
#             idx_ok = trsig.remove_trace_low_signal(3000)
#             if trsig.get_nb_trace() < 4:
#                 logger.info(f"Skip event {i_e} voltage too low, nb trace < 4")
#                 plt.show()
#                 continue
#             trsig.name = "signal no noise trigger"
#             trnoise.keep_only_trace_with_index(idx_ok)
#             trsig.traces += trnoise.traces
#             trsig.get_tmax_vmax()
#             trsig.downsize_sampling(4)
#             trsig.traces *= (8192.0/9e5)
#             trsig.traces = trsig.traces[:,:,:1024]
#             trsig.t_samples = trsig.t_samples[:,:1024]
#             trsig.get_tmax_vmax()
#             trsig.plot_footprint_val_max()
#             plt.show()


class SimuGrand:
    def __init__(self):
        # self.data_file = DataSimuFile(8192)
        self.simu = SimuDetectorUnitResponse()
        self.fact_downsample = 0
        self.size_trace = 0

    def set_sampling_size(self, fact_downsample=0, size_trace=0):
        self.fact_downsample = fact_downsample
        self.size_trace = size_trace

    def reduce_traces(self, evt):
        assert isinstance(evt, Handling3dTraces)
        if self.fact_downsample != 0:
            evt.downsize_sampling(self.fact_downsample)
        if self.size_trace != 0:
            evt.reduce_nb_sample(self.size_trace)

    def to_voltage_debug(self, pn_efield, nb_evt=1000):
        START = datetime.now()
        logger.info(f"Simu DU voltage from {pn_efield}")
        self.pn_efield = pn_efield
        f_ef = froot.get_file_event(pn_efield)
        cpt_nok = 0
        cpt_du = 0
        l_events = []
        for i_e in range(nb_evt):
            d_info = {}
            i_e += 400
            f_ef.load_event_idx(i_e)
            tref_gd = f_ef.get_obj_handling3dtraces()
            d_simu = f_ef.get_simu_parameters()
            logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
            tref = convert_3dtrace_grandlib(tref_gd, True)
            assert isinstance(tref, HandlingEfield)
            tref.network.xmax_pos = d_simu["FIX_xmax_pos"]
            tref.network.core_pos = d_simu["shower_core_pos"]
            tref.network.name = f_ef.tt_run.site
            # tref.apply_bandpass(50, 250)
            # check nb DU ok
            tref.remove_trace_low_signal(80)
            if tref.get_nb_trace() < 4:
                logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
                cpt_nok += 1
                continue
            # tref.plot_footprint_val_max()
            tref_bd = tref.copy()
            tref_bd.apply_bandpass(40, 200)
            tref_bd.plot_footprint_val_max()
            tref.plot_footprint_val_max()

            self.simu.set_xmax(d_simu["FIX_xmax_pos"])
            self.simu.set_data_efield(tref)
            self.simu.compute_du_all()
            # Check in voltage
            trsig = tref.copy(self.simu.v_out)
            trsig.name = "signal no noise"
            trsig.set_unit_axis("Volt", "dir", "Voltage")
            trsig.plot_footprint_val_max()
            plt.show()
            trnoise = trsig.copy(self.simu.v_noise)
            trnoise.name = "galactic 'noise'"
            # like a trigger
            # logger.info(trnoise.get_std_noise().mean(axis=1))
            # logger.info(trsig.get_tmax_vmax())
            idx_ok = np.argwhere(
                trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1)
            )
            idx_ok = np.ravel(idx_ok)
            if len(idx_ok) <= 4:
                logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
                # trsig.get_tmax_vmax()
                # trsig.traces += trnoise.traces
                # trsig.plot_footprint_val_max()
                # plt.show()
                cpt_nok += 1
                continue
            # Selected !!
            d_info["run_nb"] = f_ef.run_number
            d_info["event_nb"] = f_ef.event_number
            d_info["energy"] = d_simu["energy_primary"]
            # trsig_copy = trsig.copy()
            trsig.keep_only_trace_with_index(idx_ok)
            trsig.name = "signal no noise trigger"
            trnoise.keep_only_trace_with_index(idx_ok)
            # tref.keep_only_trace_with_index(idx_ok)
            # trnoise.plot_footprint_val_max()
            # plt.show()
            # logger.info(f"{trnoise.traces.shape}")
            # trsig_trig.traces += trnoise.traces
            trsig.downsize_sampling(4)
            trsig.reduce_nb_sample(1024)
            # trsig_trig.traces *= (8192.0/9e5)d_simu
            trnoise.downsize_sampling(4)
            trnoise.reduce_nb_sample(1024)
            # trsig_trig.traces *= (8192.0/9e5)d_simu
            # trsig_trig.get_tmax_vmax()
            # trsig_trig.traces += trnoise.traces
            # trsig_trig.plot_footprint_val_max()
            # 
            self.reduce_traces(trsig)
            self.reduce_traces(trnoise)
            cpt_du += trsig.get_nb_trace()
            l_events.append([trsig, trnoise, d_info])
            # if len(l_events) >= 2:
            #     break
        #
        self.data_file = DataSimuFile(1024)
        self.data_file.upload_all_events(l_events, cpt_du)
        self.data_file.save_asdf("test_05.asdf")
        logger.info(f"Event nok: {cpt_nok}")
        logger.info(f"Nb DU    : {cpt_du}")
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")

    def to_voltage(self, pn_efield, nb_evt=1000):
        START = datetime.now()
        logger.info(f"Simu DU voltage from {pn_efield}")
        self.pn_efield = pn_efield
        f_ef = froot.get_file_event(pn_efield)
        cpt_nok = 0
        cpt_du = 0
        cpt_du_all = 0
        l_events = []
        for idx in range(1000):
            d_info = {}
            i_e = 400+idx
            #i_e = idx 
            f_ef.load_event_idx(i_e)
            tref_gd = f_ef.get_obj_handling3dtraces()
            d_simu = f_ef.get_simu_parameters()
            logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
            tref = convert_3dtrace_grandlib(tref_gd, True)
            assert isinstance(tref, HandlingEfield)
            cpt_du_all += tref.get_nb_trace()
            tref.set_xmax(d_simu["FIX_xmax_pos"])
            tref.network.core_pos = d_simu["shower_core_pos"]
            tref.network.name = f_ef.tt_run.site
            # tref.apply_bandpass(50, 250)
            # check nb DU ok
            tref.remove_trace_low_signal(60)
            if tref.get_nb_trace() < 4:
                logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
                cpt_nok += 1
                continue
            # tref.plot_footprint_val_max()
            self.simu.set_xmax(d_simu["FIX_xmax_pos"])
            self.simu.set_data_efield(tref)
            self.simu.compute_du_all()
            # Check in voltage
            trsig = tref.copy(self.simu.v_out)
            assert isinstance(trsig, Handling3dTraces)
            trsig.name = "signal no noise"
            trsig.set_unit_axis("$\mu V$", "dir", "Voltage")
            # trsig.plot_footprint_val_max()
            trnoise = trsig.copy(self.simu.v_noise)
            trnoise.name = "galactic 'noise'"
            # like a trigger
            # logger.info(trnoise.get_std_noise().mean(axis=1))
            # logger.info(trsig.get_tmax_vmax())
            idx_ok = np.argwhere(
                trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1)
            )
            idx_ok = np.ravel(idx_ok)
            if len(idx_ok) <= 4:
                logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
                # trsig.get_tmax_vmax()
                # trsig.traces += trnoise.traces
                # trsig.plot_footprint_val_max()
                # plt.show()
                cpt_nok += 1
                continue
            # Selected !!
            d_info["run_nb"] = f_ef.run_number
            d_info["event_nb"] = f_ef.event_number
            d_info["idx"] = i_e
            d_info["energy"] = d_simu["energy_primary"]
            # trsig_copy = trsig.copy()
            trsig.keep_only_trace_with_index(idx_ok)
            trsig.name = "signal no noise trigger"
            trnoise.keep_only_trace_with_index(idx_ok)
            # trsig.traces += trnoise.traces
            # trsig.plot_footprint_val_max()
            # trnoise.plot_footprint_val_max()
            # plt.show()
            tref.keep_only_trace_with_index(idx_ok)
            # trnoise.plot_footprint_val_max()
            # plt.show()
            # logger.info(f"{trnoise.traces.shape}")
            # trsig_trig.traces += trnoise.traces
            # trsig.downsize_sampling(4)
            # trsig.reduce_nb_sample(1024)
            # trsig_trig.traces *= (8192.0/9e5)d_simu
            # trnoise.downsize_sampling(4)
            # trnoise.reduce_nb_sample(1024)
            # trsig_trig.traces *= (8192.0/9e5)d_simu
            # trsig_trig.get_tmax_vmax()
            self.reduce_traces(trsig)
            self.reduce_traces(trnoise)
            # do ref value Efield
            ref = get_efield_ref_values(tref)
            cpt_du += trsig.get_nb_trace()
            l_events.append([trsig, trnoise, d_info, ref])
            if len(l_events) >= nb_evt:
                break
        #
        f_trsig = l_events[0][0]
        self.data_file = DataSimuFile()
        self.data_file.init_type(f_trsig.get_size_trace())
        self.data_file.upload_all_events(l_events, cpt_du)
        # name file
        n_asdf = pn_efield.split("/")[-1]
        n_asdf = n_asdf.replace("efield", "volt")
        n_asdf = n_asdf.replace(".root", "")
        f_sample = int(trsig.f_samp_mhz[0])
        n_asdf += f"_fs{f_sample}_st{trsig.get_size_trace()}_ne{len(l_events)}.asdf"
        logger.info(f'{n_asdf}')
        self.data_file.save_asdf(n_asdf)
        logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
        logger.info(f"Event select: {len(l_events)}/{idx}")
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")


if __name__ == "__main__":
    logger = getLogger(__name__)
    TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")

    path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
    path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
    f_adc = "adc_29-24992_L1_0000.root"
    f_ef = "efield_29-24992_L0_0000.root"

    def do_simu():
        simu = SimuGrand()
        # simu.to_voltage(path_dc2 + f_ef, 200)
        simu.set_sampling_size(size_trace=4096)
        simu.to_voltage(path_dc2 + f_ef, 100)

    def check_asdf():
        df = DataSimuFile()
        df.read_asdf("volt_29-24992_L0_0000_fs2000_st8192_ne405.asdf")
        print(df.d_asdf.keys())
        event,_ = df.get_event(23)
        assert isinstance(event, Handling3dTraces)
        event.get_tmax_vmax()
        event.plot_footprint_val_max()
        event1,_ = df.get_event(402)
        event1.get_tmax_vmax()
        event1.plot_footprint_val_max()
        plt.show()

    #
    # MAIN
    #
    np.random.seed(10)
    do_simu()
    #check_asdf()
