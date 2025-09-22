#! /usr/bin/env python3
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
import proto.simu_dc2.asdf_traces as f_tr

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
    # logger.info(np.rad2deg(polars))
    return [t_max, v_max, t_gd_max, v_gd_max, polars, freq_2pt, psd_2pt]


# class SimuGrandProto:
#     def __init__(self):
#         # self.f_voc = DataFileSimu(8192)
#         self.simu = SimuDetectorUnitResponse()
#         self.fact_downsample = 0
#         self.size_trace = 0
#
#     def set_out_sampling_size(self, fact_downsample=0, size_trace=0):
#         self.fact_downsample = fact_downsample
#         self.size_trace = size_trace
#
#     def download_resize(self, evt):
#         assert isinstance(evt, Handling3dTraces)
#         if self.fact_downsample != 0:
#             evt.downsize_sampling(self.fact_downsample)
#         if self.size_trace != 0:
#             evt.reduce_nb_sample(self.size_trace)
#
#     def to_voltage_debug(self, pn_efield, nb_evt=1000):
#         START = datetime.now()
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         f_ef = froot.get_file_event(pn_efield)
#         cpt_nok = 0
#         cpt_du = 0
#         l_events = []
#         for i_e in range(nb_evt):
#             d_info = {}
#             i_e += 400
#             f_ef.load_event_idx(i_e)
#             tref_gd = f_ef.get_obj_handling3dtraces()
#             d_simu = f_ef.get_simu_parameters()
#             logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#             tref = convert_3dtrace_grandlib(tref_gd, True)
#             assert isinstance(tref, HandlingEfield)
#             tref.network.xmax_pos = d_simu["FIX_xmax_pos"]
#             tref.network.core_pos = d_simu["shower_core_pos"]
#             tref.network.name = f_ef.tt_run.site
#             # tref.apply_bandpass(50, 250)
#             # check nb DU ok
#             tref.remove_trace_low_signal(80)
#             if tref.get_nb_trace() < 4:
#                 logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#                 cpt_nok += 1
#                 continue
#             # tref.plot_footprint_val_max()
#             tref_bd = tref.copy()
#             tref_bd.apply_bandpass(40, 200)
#             tref_bd.plot_footprint_val_max()
#             tref.plot_footprint_val_max()
#
#             self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#             self.simu.set_data_efield(tref)
#             self.simu.compute_du_all()
#             # Check in voltage
#             trsig = tref.copy(self.simu.v_out)
#             trsig.name = "signal no noise"
#             trsig.set_unit_axis("Volt", "dir", "Voltage")
#             trsig.plot_footprint_val_max()
#             plt.show()
#             trnoise = trsig.copy(self.simu.v_noise)
#             trnoise.name = "galactic 'noise'"
#             # like a trigger
#             # logger.info(trnoise.get_std_noise().mean(axis=1))
#             # logger.info(trsig.get_tmax_vmax())
#             idx_ok = np.argwhere(
#                 trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1)
#             )
#             idx_ok = np.ravel(idx_ok)
#             if len(idx_ok) <= 4:
#                 logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
#                 # trsig.get_tmax_vmax()
#                 # trsig.traces += trnoise.traces
#                 # trsig.plot_footprint_val_max()
#                 # plt.show()
#                 cpt_nok += 1
#                 continue
#             # Selected !!
#             d_info["run_nb"] = f_ef.run_number
#             d_info["event_nb"] = f_ef.event_number
#             d_info["energy"] = d_simu["energy_primary"]
#             # trsig_copy = trsig.copy()
#             trsig.keep_only_trace_with_index(idx_ok)
#             trsig.name = "signal no noise trigger"
#             trnoise.keep_only_trace_with_index(idx_ok)
#             # tref.keep_only_trace_with_index(idx_ok)
#             # trnoise.plot_footprint_val_max()
#             # plt.show()
#             # logger.info(f"{trnoise.traces.shape}")
#             # trsig_trig.traces += trnoise.traces
#             trsig.downsize_sampling(4)
#             trsig.reduce_nb_sample(1024)
#             # trsig_trig.traces *= (8192.0/9e5)d_simu
#             trnoise.downsize_sampling(4)
#             trnoise.reduce_nb_sample(1024)
#             # trsig_trig.traces *= (8192.0/9e5)d_simu
#             # trsig_trig.get_tmax_vmax()
#             # trsig_trig.traces += trnoise.traces
#             # trsig_trig.plot_footprint_val_max()
#             #
#             self.download_resize(trsig)
#             self.download_resize(trnoise)
#             cpt_du += trsig.get_nb_trace()
#             l_events.append([trsig, trnoise, d_info])
#             # if len(l_events) >= 2:
#             #     break
#         #
#         self.f_voc = DataFileSimu(1024)
#         self.f_voc.upload_all_events(l_events, cpt_du)
#         self.f_voc.save_asdf("test_05.asdf")
#         logger.info(f"Event nok: {cpt_nok}")
#         logger.info(f"Nb DU    : {cpt_du}")
#         logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
#
#     def process_event(self, f_ef, idx):
#         d_info = {}
#         i_e = idx
#         f_ef.load_event_idx(i_e)
#         tref_gd = f_ef.get_obj_handling3dtraces()
#         d_simu = f_ef.get_simu_parameters()
#         logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#         tref = convert_3dtrace_grandlib(tref_gd, True)
#         assert isinstance(tref, HandlingEfield)
#         nb_du = tref.get_nb_trace()
#         tref.set_xmax(d_simu["FIX_xmax_pos"])
#         tref.network.core_pos = d_simu["shower_core_pos"]
#         tref.network.name = f_ef.tt_run.site
#         # tref.apply_bandpass(50, 250)
#         # check nb DU ok
#         tref.remove_trace_low_signal(60)
#         if tref.get_nb_trace() < 4:
#             logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#             return [None, nb_du]
#         # tref.plot_footprint_val_max()
#         self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#         self.simu.set_data_efield(tref)
#         self.simu.compute_du_all()
#         # Check in voltage
#         trsig = tref.copy(self.simu.v_out)
#         assert isinstance(trsig, Handling3dTraces)
#         trsig.name = "signal no noise"
#         trsig.set_unit_axis("$\mu V$", "dir", "Voltage")
#         # trsig.plot_footprint_val_max()
#         trnoise = trsig.copy(self.simu.v_noise)
#         trnoise.name = "galactic 'noise'"
#         # like a trigger
#         # logger.info(trnoise.get_std_noise().mean(axis=1))
#         # logger.info(trsig.get_tmax_vmax())
#         idx_ok = np.argwhere(trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1))
#         idx_ok = np.ravel(idx_ok)
#         if len(idx_ok) <= 4:
#             logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
#             # trsig.get_tmax_vmax()
#             # trsig.traces += trnoise.traces
#             # trsig.plot_footprint_val_max()
#             # plt.show()
#             return [None, nb_du]
#         # Selected !!
#         d_info["run_nb"] = f_ef.run_number
#         d_info["event_nb"] = f_ef.event_number
#         d_info["idx"] = i_e
#         d_info["energy"] = d_simu["energy_primary"]
#         trsig.keep_only_trace_with_index(idx_ok)
#         trsig.name = "signal no noise trigger"
#         trnoise.keep_only_trace_with_index(idx_ok)
#         tref.keep_only_trace_with_index(idx_ok)
#         self.download_resize(trsig)
#         self.download_resize(trnoise)
#         # do ref value Efield
#         ref = get_efield_ref_values(tref)
#         return [[trsig, trnoise, d_info, ref], nb_du]
#
#     def process_event_in_file(self, pn_ef, idx):
#         f_ef = froot.get_file_event(pn_ef)
#         d_info = {}
#         i_e = idx
#         f_ef.load_event_idx(i_e)
#         tref_gd = f_ef.get_obj_handling3dtraces()
#         d_simu = f_ef.get_simu_parameters()
#         logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#         tref = convert_3dtrace_grandlib(tref_gd, True)
#         assert isinstance(tref, HandlingEfield)
#         nb_du = tref.get_nb_trace()
#         tref.set_xmax(d_simu["FIX_xmax_pos"])
#         tref.network.core_pos = d_simu["shower_core_pos"]
#         tref.network.name = f_ef.tt_run.site
#         # tref.apply_bandpass(50, 250)
#         # check nb DU ok
#         tref.remove_trace_low_signal(60)
#         if tref.get_nb_trace() < 4:
#             logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#             return [None, nb_du]
#         # tref.plot_footprint_val_max()
#         self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#         self.simu.set_data_efield(tref)
#         self.simu.compute_du_all()
#         # Check in voltage
#         trsig = tref.copy(self.simu.v_out)
#         assert isinstance(trsig, Handling3dTraces)
#         trsig.name = "signal no noise"
#         trsig.set_unit_axis("$\mu V$", "dir", "Voltage")
#         # trsig.plot_footprint_val_max()
#         trnoise = trsig.copy(self.simu.v_noise)
#         trnoise.name = "galactic 'noise'"
#         # like a trigger
#         # logger.info(trnoise.get_std_noise().mean(axis=1))
#         # logger.info(trsig.get_tmax_vmax())
#         idx_ok = np.argwhere(trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1))
#         idx_ok = np.ravel(idx_ok)
#         if len(idx_ok) <= 4:
#             logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
#             # trsig.get_tmax_vmax()
#             # trsig.traces += trnoise.traces
#             # trsig.plot_footprint_val_max()
#             # plt.show()
#             return [None, nb_du]
#         # Selected !!
#         d_info["run_nb"] = f_ef.run_number
#         d_info["event_nb"] = f_ef.event_number
#         d_info["idx"] = i_e
#         d_info["energy"] = d_simu["energy_primary"]
#         trsig.keep_only_trace_with_index(idx_ok)
#         trsig.name = "signal no noise trigger"
#         trnoise.keep_only_trace_with_index(idx_ok)
#         tref.keep_only_trace_with_index(idx_ok)
#         self.download_resize(trsig)
#         self.download_resize(trnoise)
#         # do ref value Efield
#         ref = get_efield_ref_values(tref)
#         return [[trsig, trnoise, d_info, ref], nb_du]
#
#     def process_event_in_file_chunk(self, pn_efield, idx, n_chunk=1):
#         self.pn_efield = pn_efield
#         f_ef = froot.get_file_event(pn_efield)
#         cpt_nok = 0
#         cpt_du = 0
#         cpt_du_all = 0
#         l_events = []
#         for idx in range(idx, idx + n_chunk):
#             d_info = {}
#             i_e = idx
#             # i_e = idx
#             f_ef.load_event_idx(i_e)
#             tref_gd = f_ef.get_obj_handling3dtraces()
#             d_simu = f_ef.get_simu_parameters()
#             logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#             tref = convert_3dtrace_grandlib(tref_gd, True)
#             assert isinstance(tref, HandlingEfield)
#             cpt_du_all += tref.get_nb_trace()
#             tref.set_xmax(d_simu["FIX_xmax_pos"])
#             tref.network.core_pos = d_simu["shower_core_pos"]
#             tref.network.name = f_ef.tt_run.site
#             # tref.apply_bandpass(50, 250)
#             # check nb DU ok
#             tref.remove_trace_low_signal(60)
#             if tref.get_nb_trace() < 4:
#                 logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#                 cpt_nok += 1
#                 continue
#             # tref.plot_footprint_val_max()
#             self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#             self.simu.set_data_efield(tref)
#             self.simu.compute_du_all()
#             # Check in voltage
#             trsig = tref.copy(self.simu.v_out)
#             assert isinstance(trsig, Handling3dTraces)
#             trsig.name = "signal no noise"
#             trsig.set_unit_axis("$\mu V$", "dir", "Voltage")
#             # trsig.plot_footprint_val_max()
#             trnoise = trsig.copy(self.simu.v_noise)
#             trnoise.name = "galactic 'noise'"
#             # like a trigger
#             # logger.info(trnoise.get_std_noise().mean(axis=1))
#             # logger.info(trsig.get_tmax_vmax())
#             threshold = 5 * trnoise.get_std_noise().mean(axis=1)
#             idx_ok = np.argwhere(trsig.get_tmax_vmax()[1] > threshold)
#             idx_ok = np.ravel(idx_ok)
#             if len(idx_ok) <= 4:
#                 logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
#                 cpt_nok += 1
#                 continue
#             # Selected !!
#             d_info["run_nb"] = f_ef.run_number
#             d_info["event_nb"] = f_ef.event_number
#             d_info["idx"] = i_e
#             d_info["energy"] = d_simu["energy_primary"]
#             trsig.keep_only_trace_with_index(idx_ok)
#             trsig.name = "signal no noise trigger"
#             trnoise.keep_only_trace_with_index(idx_ok)
#             tref.keep_only_trace_with_index(idx_ok)
#             self.download_resize(trsig)
#             self.download_resize(trnoise)
#             # do ref value Efield
#             ref = get_efield_ref_values(tref)
#             cpt_du += trsig.get_nb_trace()
#             l_events.append([trsig, trnoise, d_info, ref])
#         logger.info(l_events)
#
#         return l_events, cpt_du_all, cpt_du
#
#     def process_all_events_parallel_chunk(self, pn_efield):
#         from joblib import Parallel, delayed
#
#         def process_results_chunk(results):
#             l_events = []
#             cpt_du = 0
#             cpt_du_all = 0
#
#             for ret in results:
#                 l_events += ret[0]
#                 cpt_du_all += ret[1]
#                 cpt_du += ret[2]
#                 print("\nprocess_results_chunk:")
#                 print(cpt_du, cpt_du_all, len(l_events))
#             return l_events, cpt_du, cpt_du_all
#
#         START = datetime.now()
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         # f_ef = froot.get_file_event(pn_efield)
#         # i_b, i_e = 400, 500
#         size_chk = 10
#         nb_evt = 100
#         n_chk = int(nb_evt / size_chk)
#         results = Parallel(n_jobs=4, backend="loky", inner_max_num_threads=1, return_as="list")(
#             delayed(self.process_event_in_file_chunk)(pn_efield, i * size_chk, size_chk)
#             for i in range(n_chk)
#         )
#         l_events, cpt_du, cpt_du_all = process_results_chunk(results)
#         # Create output file
#         f_trsig = l_events[0][0]
#         self.f_voc = DataFileSimu()
#         self.f_voc.set_size_trace(f_trsig.get_size_trace())
#         self.f_voc.upload_all_events(l_events, cpt_du)
#         # name file
#         n_asdf = pn_efield.split("/")[-1]
#         n_asdf = n_asdf.replace("efield", "volt")
#         n_asdf = n_asdf.replace(".root", "")
#         f_sample = int(f_trsig.f_samp_mhz[0])
#         n_asdf += f"_fs{f_sample}_st{f_trsig.get_size_trace()}_ne{len(l_events)}.asdf"
#         logger.info(f"{n_asdf}")
#         self.f_voc.save_asdf(n_asdf)
#         logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
#         logger.info(f"Event select: {len(l_events)}/{nb_evt}")
#         logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
#
#     def process_all_events(self, pn_efield):
#         cpt_nok = 0
#         cpt_du = 0
#         cpt_du_all = 0
#         START = datetime.now()
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         f_ef = froot.get_file_event(pn_efield)
#         l_events = []
#         i_b, i_e = 400, 500
#         for idx in range(i_b, i_e):
#             res = self.process_event(f_ef, idx)
#             ret = res[0]
#             cpt_du_all += res[1]
#             if ret:
#                 l_events.append(ret)
#                 trsig = ret[0]
#                 cpt_du += trsig.get_nb_trace()
#             else:
#                 cpt_nok += 1
#         # Create output file
#         f_trsig = l_events[0][0]
#         self.f_voc = DataFileSimu()
#         self.f_voc.set_size_trace(f_trsig.get_size_trace())
#         self.f_voc.upload_all_events(l_events, cpt_du)
#         # name file
#         n_asdf = pn_efield.split("/")[-1]
#         n_asdf = n_asdf.replace("efield", "volt")
#         n_asdf = n_asdf.replace(".root", "")
#         f_sample = int(trsig.f_samp_mhz[0])
#         n_asdf += f"_fs{f_sample}_st{trsig.get_size_trace()}_ne{len(l_events)}.asdf"
#         logger.info(f"{n_asdf}")
#         self.f_voc.save_asdf(n_asdf)
#         logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
#         logger.info(f"Event select: {len(l_events)}/{ i_e-i_b}")
#         logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
#
#     def process_all_events_parallel(self, pn_efield):
#         from joblib import Parallel, delayed
#
#         def process_results(results):
#             l_events = []
#             cpt_nok = 0
#             cpt_du = 0
#             cpt_du_all = 0
#
#             for ret in results:
#                 res = ret[0]
#                 cpt_du_all += ret[1]
#                 print(res)
#                 if res:
#                     l_events.append(res)
#                     trsig = res[0]
#                     cpt_du += trsig.get_nb_trace()
#                 else:
#                     cpt_nok += 1
#                 print(cpt_du_all, cpt_du, cpt_nok)
#             return l_events, cpt_nok, cpt_du, cpt_du_all
#
#         START = datetime.now()
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         # f_ef = froot.get_file_event(pn_efield)
#         i_b, i_e = 0, 240
#         results = Parallel(n_jobs=4, inner_max_num_threads=1, return_as="generator_unordered")(
#             delayed(self.process_event_in_file)(pn_efield, i) for i in range(i_b, i_e)
#         )
#
#         l_events, cpt_nok, cpt_du, cpt_du_all = process_results(results)
#         # Create output file
#         f_trsig = l_events[0][0]
#         self.f_voc = DataFileSimu()
#         self.f_voc.set_size_trace(f_trsig.get_size_trace())
#         self.f_voc.upload_all_events(l_events, cpt_du)
#         # name file
#         n_asdf = pn_efield.split("/")[-1]
#         n_asdf = n_asdf.replace("efield", "volt")
#         n_asdf = n_asdf.replace(".root", "")
#         f_sample = int(f_trsig.f_samp_mhz[0])
#         n_asdf += f"_fs{f_sample}_st{f_trsig.get_size_trace()}_ne{len(l_events)}.asdf"
#         logger.info(f"{n_asdf}")
#         self.f_voc.save_asdf(n_asdf)
#         logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
#         logger.info(f"Event select: {len(l_events)}/{i_e-i_b}")
#         logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
#
#     def to_voltage(self, pn_efield, i_beg, i_end):
#         START = datetime.now()
#         logger.info(f"Simu DU voltage from {pn_efield}")
#         self.pn_efield = pn_efield
#         f_ef = froot.get_file_event(pn_efield)
#         cpt_nok = 0
#         cpt_du = 0
#         cpt_du_all = 0
#         l_events = []
#         for idx in range(i_beg, i_end):
#             d_info = {}
#             i_e = idx
#             f_ef.load_event_idx(i_e)
#             tref_gd = f_ef.get_obj_handling3dtraces()
#             d_simu = f_ef.get_simu_parameters()
#             logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
#             tref = convert_3dtrace_grandlib(tref_gd, True)
#             assert isinstance(tref, HandlingEfield)
#             cpt_du_all += tref.get_nb_trace()
#             tref.set_xmax(d_simu["FIX_xmax_pos"])
#             tref.network.core_pos = d_simu["shower_core_pos"]
#             tref.network.name = f_ef.tt_run.site
#             # tref.apply_bandpass(50, 250)
#             # check nb DU ok
#             tref.remove_trace_low_signal(60)
#             if tref.get_nb_trace() < 4:
#                 logger.info(f"Skip event {i_e} efield to low,  nb trace < 4")
#                 cpt_nok += 1
#                 continue
#             # tref.plot_footprint_val_max()
#             self.simu.set_xmax(d_simu["FIX_xmax_pos"])
#             self.simu.set_data_efield(tref)
#             self.simu.compute_du_all()
#             # Check in voltage
#             trsig = tref.copy(self.simu.v_out)
#             assert isinstance(trsig, Handling3dTraces)
#             trsig.name = "signal no noise"
#             trsig.set_unit_axis("$\mu V$", "dir", "Voltage")
#             # trsig.plot_footprint_val_max()
#             trnoise = trsig.copy(self.simu.v_noise)
#             trnoise.name = "galactic 'noise'"
#             # trnoise.plot_footprint_val_max()
#             # like a trigger
#             # logger.info(trnoise.get_std_noise().mean(axis=1))
#             # logger.info(trsig.get_tmax_vmax())
#             idx_ok = np.argwhere(
#                 trsig.get_tmax_vmax()[1] > 5 * trnoise.get_std_noise().mean(axis=1)
#             )
#             idx_ok = np.ravel(idx_ok)
#             if len(idx_ok) <= 4:
#                 logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
#                 # trsig.get_tmax_vmax()
#                 # trsig.traces += trnoise.traces
#                 # trsig.plot_footprint_val_max()
#                 # plt.show()
#                 cpt_nok += 1
#                 continue
#             # Selected !!
#             d_info["run_nb"] = f_ef.run_number
#             d_info["event_nb"] = f_ef.event_number
#             d_info["idx"] = i_e
#             d_info["energy"] = d_simu["energy_primary"]
#             # trsig_copy = trsig.copy()
#             trsig.keep_only_trace_with_index(idx_ok)
#             trsig.name = "signal no noise trigger"
#             trnoise.keep_only_trace_with_index(idx_ok)
#             # trsig.traces += trnoise.traces
#             # trsig.plot_footprint_val_max()
#             # trnoise.plot_footprint_val_max()
#             # plt.show()
#             tref.keep_only_trace_with_index(idx_ok)
#             trnoise.traces += trsig.traces
#             trnoise.plot_footprint_val_max()
#             # plt.show()
#             # logger.info(f"{trnoise.traces.shape}")
#             # trsig_trig.traces += trnoise.traces
#             # trsig.downsize_sampling(4)
#             # trsig.reduce_nb_sample(1024)
#             # trsig_trig.traces *= (8192.0/9e5)d_simu
#             # trnoise.downsize_sampling(4)l_events.append([trsig, trnoise, d_info, ref])
#             # trnoise.reduce_nb_sample(1024)
#             # trsig_trig.traces *= (8192.0/9e5)d_simu
#             # trsig_trig.get_tmax_vmax()
#             self.download_resize(trsig)
#             self.download_resize(trnoise)
#             # do ref value Efield
#             ref = get_efield_ref_values(tref)
#             cpt_du += trsig.get_nb_trace()
#             l_events.append([trsig, trnoise, d_info, ref])
#             # if len(l_events) >= nb_evt:
#             #     break
#         #
#         if len(l_events) == 0:
#             logger.info("No Events trigger !!!")
#             return
#         f_trsig = l_events[0][0]
#         self.f_voc = DataFileSimu()
#         self.f_voc.set_size_trace(f_trsig.get_size_trace())
#         self.f_voc.upload_all_events(l_events, cpt_du)
#         # name file
#         n_asdf = pn_efield.split("/")[-1]
#         n_asdf = n_asdf.replace("efield", "volt")
#         n_asdf = n_asdf.replace(".root", f"_{i_beg}_{i_end}")
#         f_sample = int(trsig.f_samp_mhz[0])
#         n_asdf += f"_fs{f_sample}_st{trsig.get_size_trace()}_ne{len(l_events)}.asdf"
#         logger.info(f"{n_asdf}")
#         self.f_voc.save_asdf(n_asdf)
#         logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
#         logger.info(f"Event select: {len(l_events)}/{i_end-i_beg-1}")
#         logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")


class SimuGrand:
    def __init__(self, pn_efield):
        # self.f_voc = DataFileSimu(8192)
        self.simu = SimuDetectorUnitResponse()
        self.simu.params["flag_noise"] = False
        self.fact_downsample = 0
        self.size_trace = 0
        self.pn_efield = pn_efield
        logger.info(f"Simu DU voltage from {pn_efield}")
        self.ie_endp1 = 0
        self.size_chk = 1
        self.out_dir = "/home/jcolley/projet/lucky/data/"

    def set_out_sampling_size(self, fact_downsample, size_trace=0):
        self.fact_downsample = fact_downsample
        self.size_trace = size_trace

    def download_resize(self, evt):
        assert isinstance(evt, Handling3dTraces)
        if self.fact_downsample != 0:
            evt.downsize_sampling(self.fact_downsample)
        if self.size_trace != 0:
            evt.reduce_nb_sample(self.size_trace)

    def process_event_in_file_chunk(self, idx):
        f_ef = froot.get_file_event(self.pn_efield)
        cpt_nok = 0
        cpt_du = 0
        cpt_du_all = 0
        l_events = []
        i_endp1 = min(self.ie_endp1, idx + self.size_chk)
        logger.info(f"[{idx}, {i_endp1}]")
        for i_e in range(idx, i_endp1):
            d_info = {}
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
            tref.remove_trace_low_signal(80)
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
            trsig.set_unit_axis(r"$\mu V$", "dir", "Voltage")
            # Like trigger
            threshold = 5000.0 * np.ones(trsig.get_nb_trace())
            idx_ok = np.argwhere(trsig.get_tmax_vmax()[1] > threshold)
            # logger.info(f"{threshold}")
            # print(threshold)
            idx_ok = np.ravel(idx_ok)
            if len(idx_ok) <= 4:
                logger.info(f"Skip event {i_e} voltage too low, nb trace  < 4")
                cpt_nok += 1
                continue
            # Selected !!
            d_info["run_nb"] = f_ef.run_number
            d_info["event_nb"] = f_ef.event_number
            d_info["idx"] = i_e
            d_info["energy"] = d_simu["energy_primary"]
            trsig.keep_only_trace_with_index(idx_ok)
            trsig.name = "Signal no noise trigger"
            # tref.keep_only_trace_with_index(idx_ok)
            self.download_resize(trsig)
            # do ref value Efield
            # ref = get_efield_ref_values(tref)
            # tref.reduce_nb_sample(2048)
            # self.set_out_sampling_size(2, 1024)
            # self.download_resize(tref)
            cpt_du += trsig.get_nb_trace()
            # l_events.append([trsig, d_info, ref, tref])
            l_events.append([trsig, d_info])
        logger.info(l_events)

        return l_events, cpt_du_all, cpt_du


    def process_all_events_parallel_chunk(self, ie_beg, ie_endp1, size_chk=10):
        from joblib import Parallel, delayed, parallel_config

        def process_results_chunk(results):
            l_events = []
            cpt_du = 0
            cpt_du_all = 0

            for ret in results:
                l_events += ret[0]
                cpt_du_all += ret[1]
                cpt_du += ret[2]
            return l_events, cpt_du, cpt_du_all

        START = datetime.now()
        self.ie_endp1 = ie_endp1
        self.size_chk = size_chk
        nb_evt = ie_endp1 - ie_beg
        n_chk = nb_evt // size_chk
        if nb_evt % size_chk:
            n_chk += 1  # add 1 for the rest
        parallel_config(n_jobs=4, backend="loky", inner_max_num_threads=1, return_as="list")
        func_process = self.process_event_in_file_chunk
        results = Parallel()(delayed(func_process)(ie_beg + i * size_chk) for i in range(n_chk))
        l_events, cpt_du, cpt_du_all = process_results_chunk(results)
        # Create output file
        f_trsig = l_events[0][0]
        self.f_voc = f_tr.AsdfWriteVolt()
        self.f_voc.set_kind("Voc")
        self.f_voc.meta["infile"] = self.pn_efield
        self.f_voc.upload_all_events(l_events, cpt_du)
        # name file
        n_asdf = self.pn_efield.split("/")[-1]
        n_asdf = n_asdf.replace("efield", "volt")
        n_asdf = n_asdf.replace(".root", "")
        f_sample = int(f_trsig.f_samp_mhz[0])
        n_asdf = (
            self.out_dir
            + f"{n_asdf}_fs{f_sample}_st{f_trsig.get_size_trace()}_ne{len(l_events)}.asdf"
        )
        self.n_volt = n_asdf.split("/")[-1]
        logger.info(f"{n_asdf}")
        self.f_voc.pn_file = n_asdf
        self.f_voc.save_asdf(n_asdf)
        logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
        logger.info(f"Event select: {len(l_events)}/{nb_evt}")
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        # remove all traces
        self.save_efield(n_asdf)
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        print(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")

    def save_efield(self, f_name):
        f_ef = froot.get_file_event(self.pn_efield)
        nb_tr = self.f_voc.events["evt2ftr"][-1]
        nb_evt = len(self.f_voc.events)
        new_size = 4096
        efield_tr = np.empty((nb_tr, 3, new_size), dtype=np.float32)
        for i_e in range(nb_evt):
            f_ef.load_event_idx(self.f_voc.events["idx"][i_e])
            evt = f_ef.get_obj_handling3dtraces()
            # assert isinstance(evt, Handling3dTraces)
            idx_beg, idx_end = self.f_voc.get_event_interval(i_e)
            l_id = self.f_voc.mtraces["du_id"][idx_beg:idx_end].tolist()
            evt.keep_only_trace_with_ident(l_id)
            evt.reduce_nb_sample(new_size)
            efield_tr[idx_beg:idx_end] = evt.traces
        f_ef = f_tr.AsdfWriteEfield()
        f_ef.set_with_volt(self.f_voc, efield_tr, 2000)
        ef_name = f_name.replace("volt", "efield")
        f_ef.save_asdf(ef_name)

if __name__ == "__main__":
    import sys

    #
    logger = getLogger(__name__)
    TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")
    #
    path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
    path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
    f_adc = "adc_29-24992_L1_0000.root"
    f_ef = "efield_29-24992_L0_0000.root"
    path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001/"
    f_adc = "adc_29-24992_L1_0000.root"
    f_ef = "efield_39-24951_L0_0000.root"

    def do_simu():
        pn_efield = path_dc2 + f_ef
        simu = SimuGrand(pn_efield)
        # simu.to_voltage(path_dc2 + f_ef, 200)
        simu.set_out_sampling_size(4, 1024)
        # simu.to_voltage(path_dc2 + f_ef, 400, 402)
        # simu.process_all_events_parallel(pn_efield)
        simu.process_all_events_parallel_chunk(0, 200, 10)
        # simu.process_all_events(pn_efield)
        plt.show()

    def check_asdf():
        df = DataFileEvents()
        df.read_asdf("volt_29-24992_L0_0000_fs2000_st8192_ne405.asdf")
        print(df.d_asdf.keys())
        event, _ = df.get_event(23)
        assert isinstance(event, Handling3dTraces)
        event.get_tmax_vmax()
        event.plot_footprint_val_max()
        event1, _ = df.get_event(402)
        event1.get_tmax_vmax()
        event1.plot_footprint_val_max()
        plt.show()

    #
    # MAIN
    #
    if len(sys.argv) > 1:
        i_beg = int(sys.argv[1])
        i_end = int(sys.argv[2])
        simu = SimuGrand()
        simu.set_out_sampling_size()
        pn_efield = path_dc2 + f_ef
        simu.to_voltage(pn_efield, i_beg, i_end)
    else:
        np.random.seed(10)
        do_simu()
        # check_asdf()
