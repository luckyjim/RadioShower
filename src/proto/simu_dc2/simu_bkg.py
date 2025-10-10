#! /usr/bin/env python3
"""
Created on 25 mai 2025

@author: jcolley


"""
from logging import getLogger
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time


from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
from proto.simu_dc2.du_resp import SimuDetectorUnitResponse
import proto.simu_dc2.asdf_traces as f_tr


logger = getLogger(__name__)


def efield_remove_cherenkov(tref):
    """
    use the same trace for all DU with amplitude defined by distance to Xmax
    """
    assert isinstance(tref, Handling3dTraces)
    idx_rd = np.random.randint(0, tref.get_nb_trace(), 1)
    shape_cp = tref.traces.shape
    tref.traces = np.repeat(tref.traces[idx_rd],tref.get_nb_trace(),axis=0)
    assert shape_cp == tref.traces.shape
    #print(tref.traces.shape)
    a_ref = np.max(np.linalg.norm(tref.traces[0], axis=1))
    diff2 = tref.network.xmax_pos - tref.network.du_pos
    diff2 *= diff2
    a_r2 = diff2.sum(axis=1)
    r2_0 = a_r2[idx_rd]
    new_ampl = a_ref * (1 + r2_0) / (1 + a_r2)
    fact_cor = new_ampl / a_ref
    tref.traces[:, 0] *= fact_cor[:, None]
    return idx_rd


class SimuBackground:
    def __init__(self, pn_efield):
        # self.f_voc = DataFileSimu(8192)
        self.simu = SimuDetectorUnitResponse()
        self.simu.params["flag_noise"] = False
        self.fact_downsample = 0
        self.size_trace = 0
        self.pn_efield = pn_efield
        print(f"process {pn_efield}")
        logger.info(f"Simu DU voltage from {pn_efield}")
        self.ie_endp1 = 0
        self.size_chk = 1
        self.out_dir = "/home/jcolley/projet/lucky/data/"
        self.gen_polar_angle = 0

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
        f_ef = f_tr.AsdfReadTraces(self.pn_efield)
        cpt_du = 0
        cpt_du_all = 0
        l_events = []
        i_endp1 = min(self.ie_endp1, idx + self.size_chk)
        logger.info(f"[{idx}, {i_endp1}]")
        for i_e in range(idx, i_endp1):
            d_info = {}
            tref = f_ef.get_event(i_e)
            idx_cp = efield_remove_cherenkov(tref)
            # same polar angle for all DU
            if self.gen_polar_angle == "rand":
                new_polar = np.random.uniform(0, 2 * np.pi)
                d_info["new_polar"] = new_polar
            else:
                new_polar = tref.d_simu["angle_polar"] + self.gen_polar_angle
            assert isinstance(tref, Handling3dTraces)
            cpt_du_all += tref.get_nb_trace()
            #tref.plot_footprint_val_max()
            self.simu.set_xmax(tref.d_simu["xmax_nwu"])
            self.simu.set_data_efield(tref)
            self.simu.set_polar(new_polar)
            self.simu.compute_du_all()
            # Check in voltage
            trsig = tref.copy(self.simu.v_out)
            assert isinstance(trsig, Handling3dTraces)
            trsig.name = "signal no noise"
            trsig.set_unit_axis(r"$\mu V$", "dir", "Voltage")
            trsig.name = "Signal no noise"
            self.download_resize(trsig)
            cpt_du += trsig.get_nb_trace()
            l_events.append([trsig, new_polar, idx_cp])
        # logger.info(l_events)
        return l_events, cpt_du_all, cpt_du

    def process_all_events_parallel_chunk(self, ie_beg, ie_endp1, size_chk=10):
        from joblib import Parallel, delayed, parallel_config
        
        f_ef = f_tr.AsdfReadTraces(self.pn_efield, False)
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
        # name file
        n_ef = self.pn_efield.split("/")[-1]
        ln_asdf = n_ef.split("_")
        n_asdf = self.prefix + ln_asdf[1]
        n_asdf = self.out_dir + n_asdf
        f_bkg = f_tr.AsdfWriteVoltBackground()
        f_bkg.upload_all_voltage(l_events, self.cpt_du)
        d_data = {}
        f_trsig = l_events[0][0]
        d_data["f_s_mhz"] = f_trsig.f_samp_mhz[0]
        f_bkg.set_with_efield(f_ef, d_data)
        f_bkg.save_asdf(n_asdf, False)
        
        logger.info(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")
        print(f"-----> Chrono duration (h:m:s): {datetime.now()-START}")


if __name__ == "__main__":
    import sys

    #
    logger = getLogger(__name__)
    TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")
    #
    path_asdf = "/home/jcolley/projet/lucky/data/"
    f_ef = "efield_39-24951.asdf"

    def do_simu():
        pn_efield = path_asdf + f_ef
        sbkg = SimuBackground(pn_efield)
        sbkg.prefix = "volt-bgk-0_"
        sbkg.gen_polar_angle = np.deg2rad(0)
        #sbkg.gen_polar_angle = "rand"
        sbkg.simu.params["fact_padding"] = 2.0
        sbkg.set_out_sampling_size(4, 1024)
        sbkg.ie_endp1 = 435
        sbkg.size_chk = 2
        #ret = sbkg.process_event_in_file_chunk(10)
        sbkg.process_all_events_parallel_chunk(0, -1, 10)
        plt.show()


    #
    # MAIN
    #
    if len(sys.argv) > 1:
        i_beg = int(sys.argv[1])
        i_end = int(sys.argv[2])
        print(sys.argv)
    else:
        do_simu()
