"""
use ASDF 

https://asdf.readthedocs.io/en/stable/index.html

"""

import os.path
from logging import getLogger


import numpy as np
import asdf


from sradio.basis.traces_event import Handling3dTracesOfEvent


logger = getLogger(__name__)


def save_asdf_single_event(n_file, event, info_sim, type_file="simu_event"):
    assert isinstance(event, Handling3dTracesOfEvent)
    d_gen = {"type_file": type_file}
    d_gen["simu_pars"] = info_sim
    #
    d_event = {}
    d_event["name"] = event.name
    d_event["traces"] = event.traces
    d_event["f_samp_mhz"] = event.f_samp_mhz
    d_event["idx2idt"] = event.idx2idt
    d_event["name"] = event.name
    d_event["t_start_ns"] = event.t_start_ns
    d_event["unit_trace"] = event.unit_trace
    d_event["axis_name"] = event.axis_name
    d_gen["event"] = d_event
    #
    d_net = {}
    d_net["ant_pos"] = event.network.du_pos
    d_gen["network"] = d_net
    df_simu = asdf.AsdfFile(d_gen)
    df_simu.write_to(n_file, all_array_compression="zlib")


def load_asdf(n_file):
    f_asdf = asdf.open(n_file)
    try:
        type_file = f_asdf["type_file"]
    except:
        logger.error("Unknow file asdf for shower_radio library")
        f_asdf.close()
        return None
    if type_file == "simu_event":
        event, info_sim = load_asdf_simu_single_event(f_asdf)
        f_asdf.close()
        return event, info_sim
    else:
        logger.error(f"Unknow type '{type_file}' asdf for shower_radio library")
        f_asdf.close()
        return None


def load_asdf_simu_single_event(f_asdf):
    info_sim = f_asdf["simu_pars"]
    event = Handling3dTracesOfEvent(f_asdf["event"]["name"])
    print(type(f_asdf["event"]["traces"]))
    event.init_traces(
        np.array(f_asdf["event"]["traces"]),
        f_asdf["event"]["idx2idt"],
        np.array(f_asdf["event"]["t_start_ns"]),
        f_asdf["event"]["f_samp_mhz"],
    )
    event.init_network(np.array(f_asdf["network"]["ant_pos"]))
    event.set_unit_axis(f_asdf["event"]["unit_trace"])
    event.axis_name = f_asdf["event"]["axis_name"]
    return event, info_sim
