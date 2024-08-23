"""
Created on 20 août 2024

@author: jcolley
"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

from rshower.basis.traces_event import Handling3dTraces

logger = getLogger(__name__)


class GrandEventsSelectedFmt01:
    """
    Format done by Xishui Tian

    In [3]: tr3d.files
    Out[3]: ['du_id', 'du_coord', 'traces', 'zenith', 'azimuth', 'event_id']

    In [5]: tr3d["du_id"].shape
    Out[5]: (157,)

    In [6]: tr3d["du_coord"].shape
    Out[6]: (157, 4)

    In [7]: tr3d["du_coord"][0]
    Out[7]: array([  54.        , -764.02856801,  441.36822321, 1215.84407084])

    In [10]: tr3d["traces"].shape
    Out[10]: (157, 4, 1024)

    In [11]: tr3d["zenith"].shape
    Out[11]: (25,)

    In [12]: tr3d["azimuth"].shape
    Out[12]: (25,)

    In [13]: tr3d["event_id"].shape
    Out[13]: (157,)

    In [14]: tr3d["event_id"]
    Out[14]:
    array([ 2260.,  2260.,  2260.,  2260.,  2260.,  2260.,  6272.,  6272.,
        6272.,  6272.,  6272.,  6272.,  6272.,  7336.,  7336.,  7336.,
        7336.,  7336.,  7336., 44371., 44371., 44371., 44371., 44371.,
       44371., 45499., 45499., 45499., 45499., 45499., 45499., 45502.,
       45502., 45502., 45502., 45502., 45502., 45503., 45503., 45503.,
       45503., 45503., 45503., 45503., 45504., 45504., 45504., 45504.,
       45504., 45504., 45966., 45966., 45966., 45966., 45966., 45966.,
       45966., 51121., 51121., 51121., 51121., 51121., 51121., 51122.,

    """

    def __init__(self, np_file):
        try:
            d_file = np.load(np_file)
        except:
            print(f"Can't open {np_file}")
            return None
        self.np_file = np_file
        self.traces = d_file["traces"]
        self.event_id = d_file["event_id"]
        self.azimuth = d_file["azimuth"]
        self.zenith = d_file["zenith"]
        self.traces = d_file["traces"]
        self.du_id = d_file["du_id"]
        self.du_coord = d_file["du_coord"]
        try:
            self.trigger_time = d_file["trigger_time"]
        except:
            self.trigger_time = None
        assert self.zenith.size == self.azimuth.size
        idx_start_events = [0]
        nb_du_events = []
        id_crt = self.event_id[0]
        crt = 0
        for idx, id_ev in enumerate(self.event_id):
            if id_crt != id_ev:
                nb_du_events.append(crt)
                idx_start_events.append(idx)
                id_crt = id_ev
                crt = 1
            else:
                crt += 1
        nb_du_events.append(crt)
        self.nb_du_in_evt = nb_du_events
        self.nb_events = len(idx_start_events)
        assert self.zenith.size == self.nb_events
        self.idx_start_events = idx_start_events

    def plot_stats_events(self):
        l_vmax = []
        for idx in range(self.nb_events):
            evt = self.get_3dtraces(idx)
            l_vmax.append(evt.get_max_norm())
        # nunber DU in event
        plt.figure()
        plt.plot(self.nb_du_in_evt,"*")
        plt.title(f"Number of DU trigged in events\n{self.np_file}")
        plt.xlabel("idx event")
        plt.grid()
        # Max distribution
        plt.figure()
        plt.title(f"Max distribution by event\n{self.np_file}")
        labels = range(self.nb_events)
        plt.boxplot(l_vmax,labels=labels)
        plt.grid()
        plt.ylabel("ADU")
        plt.xlabel("idx event")
        # Azimuth
        plt.figure()
        plt.title(f"Estimated azimuth of source events\n{self.np_file}")
        plt.plot(self.azimuth,"*", label="Azimuth")
        plt.plot(np.mod(self.azimuth,90),'d', label="Modulo 90°")
        plt.xlabel("idx event")
        plt.ylabel("Degree")
        plt.legend()
        plt.grid()
        
    def get_info(self):
        str_info = f"nb events in file {self.nb_events}"
        return str_info

    def get_info_event(self, idx):
        assert idx < self.nb_events
        idx_start = self.idx_start_events[idx]
        str_info = f"Events {idx}/{self.event_id[idx_start]}"
        str_info += f" has {self.nb_du_in_evt[idx]} DUs"
        return str_info

    def get_3dtraces(self, idx_evt, adu2volt=False):
        assert idx_evt < self.nb_events
        idx_start = self.idx_start_events[idx_evt]
        event = Handling3dTraces(f"Event {self.event_id[idx_start]}")
        m_slice = np.arange(idx_start, idx_start + self.nb_du_in_evt[idx_evt])
        if self.trigger_time is None:
            trigger_time = None
        else:
            # convert second to nanosecond
            trigger_time = self.trigger_time[m_slice]
            trigger_time -= np.min(trigger_time)
            trigger_time *= 10e9
            logger.info("Use relative trigger time (first DU is origin of time) in ns.")
        event.init_traces(
            self.traces[m_slice, 1:],
            self.du_id[m_slice],
            trigger_time,
            f_samp_mhz=500,
        )
        unit = "ADU"
        if adu2volt:
            fact = np.float64(0.9 / 2 ** 13)
            event.traces = event.traces.astype(np.float64)
            unit = "Volt"
        event.init_network(self.du_coord[m_slice, 1:])
        event.network.name = "GP13"
        event.set_unit_axis(unit, "dir", "Trace")
        logger.debug(event.traces.shape)
        return event

    def get_azi_elev(self, idx_evt):
        assert idx_evt < self.nb_events
        pars = {}
        pars["azi"] = self.azimuth[idx_evt]
        pars["d_zen"] = self.zenith[idx_evt]
        return pars
