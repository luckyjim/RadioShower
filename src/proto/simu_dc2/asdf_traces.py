from logging import getLogger
import logging

import numpy as np
import matplotlib.pyplot as plt
import asdf
from astropy.time import Time

from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.coord import nwu_cart_to_dir_one, nwu_cart_to_dir


logger = getLogger(__name__)


S_dtype_efref = [
    ("polar_angle", "f4"),
    ("tmax_ns", "i8"),
    ("emax", "f4"),
    ("tmax_ns_band", "i8"),
    ("emax_band", "f4"),
    ("coef_logpsd", "f4", (2)),
]

#
# evt2ftr indirection table event to first trace in array dtype_traces
# evt2ftr[0] is for event index 1, by default for event 0 first DU is 0
#
S_dtype_events = [
    ("evt2ftr", "i4"),
    ("run_nb", "i4"),
    ("event_nb", "i4"),
    ("idx", "i4"),  # index of event in raw file Efield
    ("energy", "f4"),
    ("xmax_nwu", "f4", (3)),
    ("core_nwu", "f4", (3)),
]

# Network parameters
S_dtype_network = [
    ("du_id", "i4"),
    ("pos_nwu", "f4", (3)),
]

# metadata of traces
S_dtype_mtraces = [
    ("du_id", "i4"),  # in table dtype_network
    ("start_s", "i8"),
    ("start_ns", "f8"),
    ("azi", "f4"), # Azimuth
    ("d_zen", "f4"), # distance zemithal
]


class AsdfTraces:

    def get_nb_du_in_event_sort(self):
        nb_du_in_event = np.empty_like(self.events["evt2ftr"])
        nb_du_in_event[1:] = np.diff(self.events["evt2ftr"])
        nb_du_in_event[0] = self.events["evt2ftr"][0]
        idx_du = np.argsort(nb_du_in_event)
        return idx_du, nb_du_in_event[idx_du]

    def get_event_interval(self, idx_evt):
        if idx_evt == 0:
            idx_beg = 0
        else:
            idx_beg = self.events["evt2ftr"][idx_evt - 1]
        # end is the begin of next
        idx_end = self.events["evt2ftr"][idx_evt]
        return idx_beg, idx_end

    # PLOTS
    def plot_hist_du(self):
        plt.figure()
        nb_du = np.diff(self.events["evt2ftr"])
        plt.hist(nb_du)
        plt.xlabel("Number of DU in event")
        plt.title(f"Set of {self.nb_events} events")

    def plot_hist_xcore(self):
        plt.figure()
        print(self.events["core_nwu"].shape)
        print(self.events["core_nwu"][0:10])
        dist = np.linalg.norm(self.events["core_nwu"], axis=-1) / 1000
        print(dist.shape, dist[0:10])
        plt.hist(dist)
        plt.grid()
        plt.xlabel("Distance Core [Km]")
        plt.title(f"Set of {self.nb_events} events")

    def plot_hist_xmax(self):
        plt.figure()
        print(self.events["core_nwu"].shape)
        print(self.events["core_nwu"][0:10])
        vec_x = self.events["xmax_nwu"] - self.events["core_nwu"]
        dist = np.linalg.norm(vec_x, axis=-1) / 1000
        print(dist.shape, dist[0:10])
        plt.hist(dist)
        plt.grid()
        plt.xlabel("Distance Xmax-Core [Km]")
        plt.title(f"Set of {self.nb_events} events")

    def plot_dir(self):
        vec_cx = self.events["xmax_nwu"] - self.events["core_nwu"]
        print(vec_cx.shape)
        azi, zenith = np.rad2deg(nwu_cart_to_dir(vec_cx.T))
        plt.figure()
        plt.hist(azi)
        plt.grid()
        plt.xlabel("Azimuth")
        plt.figure()
        plt.hist(zenith)
        plt.grid()
        plt.xlabel("Zenith")


class AsdfWriteTraces(AsdfTraces):
    def __init__(self):
        self.pn_file = ""
        # Master dictionary of ASDF file
        self.d_asdf = {}
        # Infos
        d_infos = {}
        d_infos["description"] = "Files events/traces from RadioShower library"
        d_infos["version"] = "0.1"
        d_infos["date"] = Time.now().to_value("isot", subfmt="date_hm")
        d_infos["laboratory"] = "LPNHE/IN2P3/CNRS France"
        d_infos["author"] = "Jean-Marc Colley"
        d_infos["comment"] = ""
        d_infos["history"] = ""
        d_infos["project"] = "GRAND RadioShower"
        d_infos["repository"] = "https://github.com/luckyjim/RadioShower"
        self.d_infos = d_infos
        # Metadata
        self.meta = {}
        self.meta["type_trace"] = "TBD"
        self.meta["f_samp_mhz"] = 0
        self.meta["unit"] = "TBD"
        self.meta["site"] = "TBD"
        self.meta["nb_evt"] = 0
        self.meta["nb_trace"] = 0

    def set_kind(self, kind):
        if not kind in ["Voc", "Efield"]:
            raise
        self.meta["type_trace"] = kind
        
    def set_magnetic_field(self, m_field):
        d_mag = {}
        d_mag["inc_deg"] = m_field[0]
        d_mag["dec_deg"] = m_field[1]
        d_mag["modul_uT"] = m_field[2]
        self.meta["mag_field"] = d_mag

    def allocate_arrays(self, nb_evts, nb_traces, s_trace, nb_du=400):
        self.traces = np.empty((nb_traces, 3, s_trace), dtype=np.float32)
        self.mtraces = np.zeros(nb_traces, dtype=S_dtype_mtraces)
        self.events = np.zeros(nb_evts, dtype=S_dtype_events)
        self.network = np.zeros(nb_du, dtype=S_dtype_network)

    def save_asdf(self, pn_traces, f_zip=False):
        self.pn_file = pn_traces
        self.d_asdf["infos_file"] = self.d_infos
        self.d_asdf["meta"] = self.meta
        self.d_asdf["traces"] = self.traces
        self.d_asdf["mtraces"] = self.mtraces
        self.d_asdf["events"] = self.events
        self.d_asdf["network"] = self.network
        file_simu = asdf.AsdfFile(self.d_asdf)
        if f_zip:
            file_simu.write_to(pn_traces, all_array_compression="zlib")
        else:
            file_simu.write_to(pn_traces)


class AsdfWriteVolt(AsdfWriteTraces):
    def __init__(self):
        super().__init__()

    # CREATE file
    def upload_all_voltage(self, l_events, nb_trace):
        """
        l_events = [[event_signal, dict_params_simu] ]
        """
        nb_evt = len(l_events)
        sig = l_events[0][0]
        assert isinstance(sig, Handling3dTraces)
        self.allocate_arrays(nb_evt, nb_trace, sig.get_size_trace())
        i_beg = 0
        idt2idx = {}
        idx_du = 0
        # info
        self.d_info = {}
        for idx, evt in enumerate(l_events):
            sig = evt[0]
            info = evt[1]
            efi = evt[2]
            assert isinstance(sig, Handling3dTraces)
            # assert isinstance(noise, Handling3dTraces)
            i_end = i_beg + sig.get_nb_trace()
            # Traces
            self.traces[i_beg:i_end] = sig.traces
            self.mtraces["start_ns"][i_beg:i_end] = sig.t_start_ns
            self.mtraces["du_id"][i_beg:i_end] = sig.idx2idt
            self.mtraces["azi"][i_beg:i_end] = efi["dir_xmax"][0]
            self.mtraces["d_zen"][i_beg:i_end] = efi["dir_xmax"][1]
            # Events
            self.events["evt2ftr"][idx] = i_end
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
            # free memory ?
            evt[0] = None
        self.idt2idx = idt2idx
        self.meta["f_samp_mhz"] = sig.f_samp_mhz[0]
        self.meta["unit"] = sig.unit_trace
        self.meta["site"] = sig.network.name
        self.meta["nb_evt"] = nb_evt
        self.meta["nb_trace"] = nb_trace


class AsdfWriteEfield(AsdfWriteTraces):
    def __init__(self):
        super().__init__()

    def upload_all_efield(self, l_events, nb_traces):
        """
        l_events = [[event_signal, dict_params_simu] ]
        """
        i_beg = 0
        self.traces = np.empty((nb_traces, 1, 4096), dtype=np.float32)
        self.pol_angle = np.empty(nb_traces, dtype=np.float32)
        for evt in l_events:
            d_ef = evt[2]
            ef_pol = d_ef["ef_pol"]
            i_end = i_beg + ef_pol.shape[0]
            # Traces
            self.traces[i_beg:i_end,0] = ef_pol
            self.pol_angle[i_beg:i_end] = d_ef["angle_pol"]
            i_beg = i_end
            
    def set_with_volt(self, f_volt, f_s_mhz):
        self.meta = f_volt.d_asdf["meta"].copy()
        self.meta["unit"] = r"$\mu V/m$"
        self.meta["f_samp_mhz"] = f_s_mhz
        self.set_kind("Efield")
        self.events = {"$ref": f"{f_volt.pn_file}#events"}
        self.network = {"$ref": f"{f_volt.pn_file}#network"}
        self.mtraces = {"$ref": f"{f_volt.pn_file}#mtraces"}
        self.d_asdf["pol_angle"] = self.pol_angle


class AsdfReadTraces(AsdfTraces):

    def __init__(self, pn_traces):
        self.d_asdf = asdf.open(pn_traces)
        self.d_asdf.find_references()
        self.traces = self.d_asdf["traces"]
        self.mtraces = self.d_asdf["mtraces"]
        self.events = self.d_asdf["events"]
        self.network = self.d_asdf["network"]
        self.meta = self.d_asdf["meta"]
        self.idt2idx = {idt: idx for idx, idt in enumerate(self.network["du_id"])}
        self.nb_events = len(self.events)

    def get_event(self, idx_evt):
        idx_beg, idx_end = self.get_event_interval(idx_evt)
        if self.traces.shape[1] == 1:
            traces = np.zeros((idx_end-idx_beg,3,self.traces.shape[2]),dtype= self.traces.dtype)
            traces[:,0] = np.squeeze(self.traces[idx_beg:idx_end])
        elif  self.traces.shape[1] == 3:
            traces = self.traces[idx_beg:idx_end]        
        evt_id = f"IDX={self.events['idx'][idx_evt]}, EVT_NB={self.events['event_nb'][idx_evt]}, RUN_NB={self.events['run_nb'][idx_evt]}"
        event = Handling3dTraces(evt_id)
        event.init_traces(
            traces,
            self.mtraces["du_id"][idx_beg:idx_end],
            self.mtraces["start_ns"][idx_beg:idx_end],
            self.meta["f_samp_mhz"],
        )
        vec_cx = self.events["xmax_nwu"][idx_evt] - self.events["core_nwu"][idx_evt]
        dist_xmax = np.linalg.norm(vec_cx) / 1000
        info_shower = f"||xmax_pos_shc||={dist_xmax:.1f} km;"
        azi, zenith = np.rad2deg(nwu_cart_to_dir_one(vec_cx))
        info_shower += f" (azi, zenith)=({azi:.0f}, {zenith:.0f}) deg;"
        nrj = self.events["energy"][idx_evt]
        info_shower += f" energy_primary={nrj:.1e} GeV"
        event.info_shower = info_shower
        l_idt = list(self.mtraces["du_id"][idx_beg:idx_end])
        l_idx = [self.idt2idx[idt] for idt in l_idt]
        pos_du = self.network["pos_nwu"][l_idx]
        event.init_network(pos_du)
        event.network.name = self.meta["site"]
        if self.meta["type_trace"] == "Voc":
            event.set_unit_axis(r"$\mu V$", "dir", "Voc")
        elif self.meta["type_trace"] == "Efield":
            if self.traces.shape[1] == 1 :
                event.set_unit_axis(r"$\mu V/m$", "pol", "Efield")
            else:
                event.set_unit_axis(r"$\mu V/m$", "dir", "Efield")
        else:
            print(self.meta)
            raise
        event.network.xmax_pos = self.events["xmax_nwu"][idx_evt]
        event.network.core_pos = self.events["core_nwu"][idx_evt]
        return event


if __name__ == "__main__":
    #
    logger = getLogger(__name__)
    TPL_FMT_LOGGER = "%(asctime)s.%(msecs)03d %(levelname)5s [%(name)s %(lineno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER, datefmt="%d %H:%M:%S")
    #
    path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
    path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
    f_adc = "adc_29-24992_L1_0000.root"
    self = "efield_29-24992_L0_0000.root"

    def check_asdf():
        df = AsdfReadTraces("volt_29-24992_L0_0000_fs2000_st8192_ne405.asdf")
        print(df.d_asdf.keys())
        event, _ = df.get_event(23)
        assert isinstance(event, Handling3dTraces)
        event.get_tmax_vmax()
        event.plot_footprint_val_max()
        event1, _ = df.get_event(402)
        event1.get_tmax_vmax()
        event1.plot_footprint_val_max()
        plt.show()

    check_asdf()
