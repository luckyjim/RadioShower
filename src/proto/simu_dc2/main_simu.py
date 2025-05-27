"""
Created on 25 mai 2025

@author: jcolley
"""
from logging import getLogger
import logging
import pprint

import numpy as np
import matplotlib.pyplot as plt
import asdf
from astropy.time import Time

import grand.dataio.root_files as froot

from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.traces_event import Handling3dTraces
from du_resp import SimuDetectorUnitResponse



logger = getLogger(__name__)


class DataSimuFile:
    def __init__(self, s_sample):
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
            ("tr_sig", "f4", (3, s_sample)),
            ("tr_noise", "f4", (3, s_sample)),
            ("start_s", "i8"),
            ("start_ns", "f8"),
            ("tmax_efield_ns", "i8"),
            ("tmax_efield_ns_band", "i8"),
            ("vmax_efield", "f4"),
            ("vmax_efield_band", "f4"),
            ("polar_angle", "f4"),
            ("du_id", "i4"),
        ]
        self.dtype_events = [
            ("idx_end", "i8"),
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
        self.events = np.zeros(nb_evts, dtype=self.dtype_events)
        self.network = np.zeros(nb_du, dtype=self.dtype_network)

    def read_asdf(self, pn_simu):
        self.d_asdf = asdf.open(pn_simu)
        self.traces = self.d_asdf["data"]["traces"]
        self.events = self.d_asdf["data"]["events"]
        self.network = self.d_asdf["data"]["network"]
        self.idt2idx = {idt: idx for idx, idt in enumerate(self.network["du_id"])}

    def get_event(self, idx_evt):
        if idx_evt == 0:
            idx_beg = 0
        else:
            idx_beg = self.events["idx_end"][idx_evt - 1]
        idx_end = self.events["idx_end"][idx_evt]
        logger.info(idx_beg)
        logger.info(idx_end)
        traces = self.traces["tr_sig"][idx_beg:idx_end]
        #traces += self.traces["tr_noise"][idx_beg:idx_end]
        event = Handling3dTraces(f"Event {idx_evt}")
        event.init_traces(
            traces,
            self.traces["du_id"][idx_beg:idx_end],
            self.traces["start_ns"][idx_beg:idx_end],
            self.d_asdf["info"]["f_samp_mhz"],
        )
        unit = self.d_asdf["info"]["unit"]
        l_idt = list(self.traces["du_id"][idx_beg:idx_end])
        l_idx = [self.idt2idx[idt] for idt in l_idt]
        pos_du = self.network["pos_nwu"][l_idx]
        event.init_network(pos_du)
        event.network.name = self.d_asdf["info"]["network"]
        event.set_unit_axis(unit, "dir", "Trace")
        event.network.xmax_pos = self.events["xmax_nwu"][idx_evt]
        event.network.core_pos = self.events["core_nwu"][idx_evt]
        return event

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
            assert isinstance(sig, Handling3dTraces)
            assert isinstance(noise, Handling3dTraces)
            i_end = i_beg + sig.get_nb_trace()
            # traces
            self.traces["tr_sig"][i_beg:i_end] = sig.traces
            self.traces["tr_noise"][i_beg:i_end] = noise.traces
            self.traces["start_ns"][i_beg:i_end] = sig.t_start_ns
            self.traces["du_id"][i_beg:i_end] = sig.idx2idt
            # events
            info = evt[2]
            self.events["idx_end"][idx] = i_end
            self.events["run_nb"][idx] = info["run_nb"]
            self.events["event_nb"][idx] = info["event_nb"]
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
        file_simu = asdf.AsdfFile(self.d_asdf)
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
        self.data_file = DataSimuFile(8192)
        self.simu = SimuDetectorUnitResponse()

    def to_voltage(self, pn_efield, nb_evt=1000):
        logger.info(f"Simu DU voltage from {pn_efield}")
        self.pn_efield = pn_efield
        f_ef = froot.get_file_event(pn_efield)
        cpt_nok = 0
        cpt_du = 0
        l_events = []
        for i_e in range(nb_evt):
            d_info = {}
            i_e += 200
            f_ef.load_event_idx(i_e)
            d_info["run_nb"] = f_ef.run_number
            d_info["event_nb"] = f_ef.event_number
            tref_gd = f_ef.get_obj_handling3dtraces()
            d_simu = f_ef.get_simu_parameters()
            d_info["energy"] = d_simu["energy_primary"]
            logger.info(f"Load {i_e} => run/evt : {f_ef.run_number}/{f_ef.event_number}")
            tref = convert_3dtrace_grandlib(tref_gd, True)
            tref.network.xmax_pos = d_simu["FIX_xmax_pos"]
            tref.network.core_pos = d_simu["shower_core_pos"]
            tref.network.name="Chine GP300"
            # check nb DU ok
            tref.remove_trace_low_signal(50)
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
            trsig.name = "signal no noise"
            trsig.set_unit_axis("Volt", "dir", "Voltage")
            # trsig.plot_footprint_val_max()
            trnoise = tref.copy(self.simu.v_noise)
            trnoise.name = "galactic 'noise'"
            trnoise.set_unit_axis("Volt", "dir", "Voltage")
            # like a trigger
            trsig_trig = trsig.copy()
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
            trsig_trig.keep_only_trace_with_index(idx_ok)
            tref.keep_only_trace_with_index(idx_ok)
            trsig_trig.name = "signal no noise trigger"
            trnoise.keep_only_trace_with_index(idx_ok)
            # trnoise.plot_footprint_val_max()
            # logger.info(f"{trnoise.traces.shape}")
            # trsig_trig.traces += trnoise.traces
            # trsig_trig.downsize_sampling(4)
            # trsig_trig.traces *= (8192.0/9e5)d_simu
            # trsig_trig.traces = trsig_trig.traces[:,:,:1024]
            # trsig_trig.t_samples = trsig_trig.t_samples[:,:1024]
            trsig_trig.get_tmax_vmax()
            #trsig_trig.traces += trnoise.traces
            trsig_trig.plot_footprint_val_max()
            #
            cpt_du += trsig_trig.get_nb_trace()
            l_events.append([trsig_trig, trnoise, d_info])
            if len(l_events) >= 2:
                break
        #
        self.data_file.upload_all_events(l_events, cpt_du)
        self.data_file.save_asdf("test.asdf")
        logger.info(f"Event nok: {cpt_nok}")
        logger.info(f"nb DU: {cpt_du}")


logger = getLogger(__name__)
TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")


path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
f_adc = "adc_29-24992_L1_0000.root"
f_ef = "efield_29-24992_L0_0000.root"

def do_simu():
    simu = SimuGrand()
    simu.to_voltage(path_dc2 + f_ef, 10)

def check_asdf():
    df =DataSimuFile(10)
    df.read_asdf("test.asdf")
    print(df.d_asdf.keys())
    event = df.get_event(0)
    assert isinstance(event, Handling3dTraces)
    event.plot_footprint_val_max()
    event1 = df.get_event(1)
    event1.plot_footprint_val_max()
    plt.show()
    
#
# MAIN
#
do_simu()
check_asdf()