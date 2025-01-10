"""
Created on 28 mars 2023

@author: jcolley


Read ZHAires text Outputs simulation, can be convert:
* axxx.trace
* xxxx.sry
* ant_pos.txt

"""

import re
import os.path
from logging import getLogger
import tarfile
import tempfile
import pprint

import numpy as np

from rshower.basis.efield_event import HandlingEfield
from .zhaires_base import ZhairesSingleEventBase, get_simu_xmax

logger = getLogger(__name__)

# approximative regular expression of string float
REAL = r"[+-]?[0-9][0-9.eE+-]*"


def convert_str2number(elmt):
    """
    Try convert string value of dictionary in float with recursive scheme

    :param elmt:
    """
    if isinstance(elmt, str):
        try:
            if "." in elmt or "e" in elmt or "E" in elmt:
                return float(elmt)
            else:
                return int(elmt)
        except ValueError:
            return elmt
    elif isinstance(elmt, dict):
        return {key: convert_str2number(val) for key, val in elmt.items()}
    elif isinstance(elmt, list):
        return [convert_str2number(val) for val in elmt]
    else:
        return elmt


class ZhairesSummaryFileVers28:
    def __init__(self, file_sry="", str_sry=""):
        logger.debug(file_sry)
        self.d_sry = {}
        self.l_error = []
        #  Location of max.(Km):
        self.d_re = {
            "t_sample_ns": rf"Time bin size:\s+(?P<t_sample_ns>{REAL})ns",
            "x_max": rf"Location of max\.\((?P<unit>\w+)\):\s+(?P<alt>{REAL})\s+(?P<dist>{REAL})\s+(?P<x>{REAL})\s+(?P<y>{REAL})\s+(?P<z>{REAL})",
            "sl_depth_of_max": rf"Sl\. depth of max\. \((?P<unit>[\w/]+)\):\s+(?P<mean>{REAL})",
            "ground_altitude": rf"Ground altitude:\s+(?P<alt>{REAL})\s+(?P<unit>\w+)\s+",
            "vers_aires": r"This is AIRES version\s+(?P<vers_aires>\w+\.\w+\.\w+)\s+\(",
            "vers_zhaires": r"With ZHAireS version (?P<vers_zhaires>\w+\.\w+\.\w+) \(",
            "primary": r"Primary particle:\s+(?P<primary>[\w^0-9]*)\s+",
            "site": rf"Site:\s+(?P<name>\w+)\s+\(Lat:\s+(?P<lat>{REAL})\s+deg. Long:\s+(?P<lon>{REAL})\s+deg",
            "geo_mag1": rf"Geomagnetic field: Intensity:\s+(?P<norm>{REAL})\s+(?P<unit>\w+)",
            "geo_mag2": rf"\s+I:\s+(?P<inc>{REAL})\s+deg. D:\s+(?P<dec>{REAL})\s+deg",
            "energy": rf"Primary energy:\s+(?P<value>{REAL})\s+(?P<unit>\w+)",
            "shower_zenith": rf"Primary zenith angle:\s+(?P<shower_zenith>{REAL})\s+deg",
            "shower_azimuth": rf"Primary azimuth angle:\s+(?P<shower_azimuth>{REAL})\s+deg",
        }
        self.str_sry = str_sry
        if file_sry != "":
            print(f"Read {file_sry}")
            with open(file_sry) as f_sry:
                self.str_sry = f_sry.read()

    def extract_all(self):
        self.l_error = []
        d_sry = {}
        for key, s_re in self.d_re.items():
            ret = re.search(s_re, self.str_sry)
            if ret:
                d_ret = ret.groupdict()
                if key in d_ret.keys():
                    # single value
                    d_sry.update(d_ret)
                else:
                    # set of values in sub dictionary with key {key}
                    d_sry[key] = d_ret
            else:
                logger.warning(f"Can't find {key} with {s_re}")
                self.l_error.append(key)
                break
        self.d_sry = convert_str2number(d_sry)
        logger.debug(pprint.pformat(self.d_sry))
        pprint.pprint(self.d_sry)

    def get_dict(self):
        return self.d_sry

    def is_ok(self):
        return len(self.l_error) == 0


class ZhairesSummaryFileVers28b(ZhairesSummaryFileVers28):
    def __init__(self, file_sry="", str_sry=""):
        super().__init__(file_sry, str_sry)
        self.d_re[
            "x_max"
        ] = rf"Pos. Max.:\s+(?P<alt>{REAL})\s+(?P<dist>{REAL})\s+(?P<x>{REAL})\s+(?P<y>{REAL})\s+(?P<z>{REAL})\s+"


# add here all version of ZHaireS summary file
L_SRY_VERS = [ZhairesSummaryFileVers28b, ZhairesSummaryFileVers28]


class ZhairesSingleEventText(ZhairesSingleEventBase):
    def __init__(self, path_zhaires):
        super().__init__(path_zhaires)

    def read_all(self):
        self.read_summary_file()
        self.extract_trace()
        if self.status == 0:
            self.read_antpos_file()
            self.read_trace_files()
            if self.path != self.path_traces:
                self.path_traces.cleanup()

    def add_path(self, file):
        return os.path.join(self.path, file)

    def add_path_traces(self, file):
        return os.path.join(self.path_traces, file)

    def extract_trace(self):
        if os.path.exists(self.add_path("antpos.dat")):
            self.path_traces = self.path
            return
        tar_file = os.path.join(self.path, self.path.split("/")[-1] + "_trace.tar.gz")
        self.path_traces = tempfile.TemporaryDirectory()
        try:
            my_tar = tarfile.open(tar_file)
            my_tar.extractall(self.path_traces)
            my_tar.close()
        except:
            # 3 pb tar file
            self.status = 3

    def read_antpos_file(self):
        a_dtype = {
            "names": ("idx", "name", "x", "y", "z"),
            "formats": ("i4", "S20", "f4", "f4", "f4"),
        }
        f_ant = self.add_path_traces("antpos.dat")
        if os.path.exists(f_ant):
            self.ants = np.loadtxt(f_ant, dtype=a_dtype)
            self.nb_ant = self.ants.shape[0]
        else:
            # 4 pb with antpos.dat
            self.status = 4

    def read_summary_file(self):
        # l_files = list(filter(os.path.isfile, os.listdir(self.path)))
        try:
            l_files = os.listdir(self.path)
        except:
            logger.error(f"Unknown path {self.path}")
            return False
        # print(l_files)
        l_sry = []
        for m_file in l_files:
            if ".sry" in m_file and m_file[0] != ".":
                l_sry.append(m_file)
        nb_sry = len(l_sry)
        if nb_sry > 1:
            logger.warning(f"several files summary ! in {self.path}")
            logger.warning(l_sry)
        if nb_sry == 0:
            logger.error(f"no files summary ! in {self.path}")
            raise
        else:
            f_sry = self.add_path(l_sry[0])
            for sry_vers in L_SRY_VERS:
                sry = sry_vers(f_sry)
                sry.extract_all()
                if sry.is_ok():
                    self.d_info = sry.get_dict()
                    self.status = 0
                    return
            logger.error(f"Unknown summary file version {sry.l_error}")
            self.status = 1

    def read_trace_files(self):
        trace_0 = np.loadtxt(self.add_path_traces("a0.trace"))
        nb_sample = trace_0.shape[0]
        self.traces = np.empty((self.nb_ant, 3, nb_sample), dtype=np.float32)
        self.t_start = np.empty(self.nb_ant, dtype=np.float64)
        for idx in range(self.nb_ant):
            f_trace = self.add_path(f"a{idx}.trace")
            # print(f_trace)
            trace = np.loadtxt(f_trace)
            assert trace.shape[0] == nb_sample
            self.traces[idx] = trace.transpose()[1:]
            self.t_start[idx] = trace[0, 0]

    def get_simu_info(self):
        return self.d_info

    def get_object_3dtraces(self):
        o_tevent = HandlingEfield("File: " + self.dir_simu)
        du_id = [str(iddu, "UTF-8") for iddu in self.ants["name"].tolist()]
        #  MHz/ns: 1e-6/1e-9 = 1e3
        sampling_freq_mhz = 1e3 / self.d_info["t_sample_ns"]
        o_tevent.init_traces(
            self.traces,
            du_id,
            self.t_start,
            sampling_freq_mhz,
        )
        ants = np.empty((self.nb_ant, 3), dtype=np.float32)
        ants[:, 0] = self.ants["x"]
        ants[:, 1] = self.ants["y"]
        ants[:, 2] = self.ants["z"]
        o_tevent.init_network(ants)
        i_sim = self.get_simu_info()
        o_tevent.network.name = i_sim["site"]["name"]
        o_tevent.info_shower = f"Xmax dist {i_sim['x_max']['dist']:.1f}km, (azi, zenith): {i_sim['shower_azimuth']:.1f}, {i_sim['shower_zenith']:.1f}deg"
        o_tevent.set_unit_axis(r"$\mu$V/m", "cart", "EField")
        o_tevent.set_xmax(get_simu_xmax(self.d_info))
        o_tevent.network.core_pos = np.array([0,0,0])
        return o_tevent
