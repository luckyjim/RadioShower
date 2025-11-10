#! /usr/bin/env python3
"""
Created on 25 mai 2025

@author: jcolley


"""
from logging import getLogger
import logging


import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time


import grand.dataio.root_files as froot

from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
from proto.simu_dc2.du_resp import SimuDetectorUnitResponse
import rshower.io.events.asdf_traces as f_tr

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
        self.out_dir = "/sps/grand/colley/data/dc2/"
        self.prefix = "volt-ash_"
        
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
            self.download_resize(trsig)
            #
            tref.keep_only_trace_with_index(idx_ok)
            tref.reduce_nb_sample(4096)
            tref.get_polar_angle()
            d_ef = {
                "ef_pol": tref.ef_pol,
                "angle_pol": tref.polar_angle_rad,
                "dir_xmax": np.rad2deg(tref.dir_angle.T),
            }
            cpt_du += trsig.get_nb_trace()
            l_events.append([trsig, d_info, d_ef])
        logger.info(l_events)

        return l_events, cpt_du_all, cpt_du

    def process_all_events_parallel_chunk(self, ie_beg, ie_endp1, size_chk=10):
        from joblib import Parallel, delayed, parallel_config
        
        f_ef = froot.get_file_event(self.pn_efield)
        if ie_endp1 < 0:
            ie_endp1 = f_ef.get_nb_events()

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
        self.cpt_du = cpt_du
        # Create output file
        # f_trsig = l_events[0][0]
        self.f_voc = f_tr.AsdfWriteVolt()
        self.f_voc.set_kind("Voc")
        self.f_voc.meta["infile"] = self.pn_efield
        #
        self.f_voc.upload_all_voltage(l_events, cpt_du)
        # add magnetic field
        d_simu = f_ef.get_simu_parameters()
        self.f_voc.set_magnetic_field(d_simu["magnetic_field"])
        # name file
        n_ef = self.pn_efield.split("/")[-1]
        ln_asdf = n_ef.split("_")
        n_asdf = self.prefix + ln_asdf[1] + ".asdf"
        n_asdf = self.out_dir + n_asdf
        logger.info(f"Save voltage in {n_asdf}")
        self.f_voc.save_asdf(n_asdf, False)
        logger.info(f"DU select   : {cpt_du}/{cpt_du_all}")
        logger.info(f"Event select: {len(l_events)}/{nb_evt}")
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        # remove all traces
        self.save_efield_pol(l_events, n_asdf)
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        

    def save_efield_pol(self, l_events, f_name):
        f_ef = f_tr.AsdfWriteEfield()
        f_ef.upload_all_efield(l_events, self.cpt_du)
        f_ef.set_with_volt(self.f_voc, 2000)
        ef_name = f_name.replace(self.prefix, "efield_")
        f_ef.save_asdf(ef_name, True)

    def save_efield_3d(self, f_name):
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
        ef_name = f_name.replace("volt", "efield3")
        f_ef.save_asdf(ef_name, True)


if __name__ == "__main__":
    import sys

    #
    logger = getLogger(__name__)
    TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")
    #
    path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
    path_data = "/sps/grand/DC2Training/"
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
        simu.process_all_events_parallel_chunk(0, -1, 10)
        # simu.process_all_events(pn_efield)
        plt.show()

    def check_asdf():
        df = f_tr.AsdfReadTraces("volt_29-24992.asdf")
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
