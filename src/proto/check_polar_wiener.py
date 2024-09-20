"""
Check polar wiener with DC2 simulation
"""

import os.path
import pprint

import numpy as np
import grand.dataio.root_files as froot

from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield, plt
from rshower.io.rf_fmt import read_TF3_fmt
from rshower.io.leff_fmt import get_leff_default
from rshower.io.events.grand_trigged import get_info_shower
from rshower.model.ant_resp import DetectorUnitAntenna3Axis


import proto.polar_wiener as pw

path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS-AN_0000/"
#path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS-AN/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001/"
f_dc2_adc = "adc_5388-23832_L1_0000.root"
f_dc2_ef = "efield_5388-23832_L0_0000.root"
#f_dc2_ef = "efield_39-24951_L0_0000.root"
pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"




def add_path(f_name):
    return path_dc2 + f_name


def convert_3dtrace(gef, f_efield=False):
    if f_efield:
        tr_ef = HandlingEfield(gef.name)
    else:
        tr_ef = Handling3dTraces(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.set_unit_axis(gef.unit_trace, "dir", gef.type_trace)
    return tr_ef


def plot_dc2_event(i_e):
    assert os.path.isfile(add_path(f_dc2_adc))
    assert os.path.isfile(add_path(f_dc2_ef))
    tradc = froot.get_handling3dtraces(add_path(f_dc2_adc), i_e)
    tref = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    tradc.get_tmax_vmax()
    tradc.plot_footprint_val_max()
    tref.plot_footprint_val_max()


def plot_polar(i_e):
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    pprint.pprint(d_sim)
    # print(gef.network.du_pos)
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.set_xmax(d_sim["xmax_pos_shc"])
    tr_ef.info_shower = get_info_shower(d_sim)
    tr_ef.set_unit_axis(gef.unit_trace, "dir", gef.type_trace)
    # tr_ef.plot_footprint_4d_max()
    tr_ef.plot_footprint_val_max()
    tr_ef.plot_footprint_time_max()
    tr_ef.plot_polar_angle()
    # tr_ef.plot_trace_idx(241)
    # tr_ef.plot_trace_idx(242)
    # tr_ef.plot_trace_idx(243)
    i_du = 0
    tr_ef.plot_trace_3d_idx(i_du)
    tr_ef.plot_trace_idx(i_du)
    # tr_ef.apply_bandpass(70, 190, False)


def get_true_polar_angle(i_e):
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    print(d_sim)
    # print(gef.network.du_pos)
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.set_xmax(d_sim["xmax_pos_shc"])
    tr_ef.info_shower = get_info_shower(d_sim)
    a_pol = tr_ef.get_polar_angle_efield(True)
    return a_pol, d_sim, tr_ef


def estimate_polar_angle(i_e):
    a_pol, d_sim, _ = get_true_polar_angle(i_e)
    gadc = froot.get_handling3dtraces(add_path(f_dc2_adc), i_e)
    evt = convert_3dtrace(gadc)
    fact = np.float64(0.9) / (2 ** 13)
    evt.traces = fact * evt.traces.astype(np.float64)
    evt.unit_trace = "Volt"
    evt.remove_trace_low_signal(0.02)
    evt.info_shower = get_info_shower(d_sim)
    # Load instrument model : antenna
    ant3d = DetectorUnitAntenna3Axis(get_leff_default(pn_fmodel))
    # Load instrument model : RF chain
    rf_fft = read_TF3_fmt(pn_fmodel)
    # evt.plot_footprint_val_max()
    pars = {}
    # pars["azi"] = d_sim["azimuth"]
    # pars["d_zen"] = d_sim["zenith"]
    pars["azi"] = 354.2
    pars["d_zen"] = 75.2
    # pars["azi"] = 102.42129156
    # pars["d_zen"] = 75
    # evt 0, idx 2, SNR 10, pa= 87, 197
    pars["azi"] = 21.2
    pars["d_zen"] = 47.2
    # evt 1, idx 85 (DU153), SNR 30, pa= 105, 285
    pars["azi"] = 203
    pars["d_zen"] = 62.3
    # evt 2, 
    pars["azi"] = 11
    pars["d_zen"] = 78
    # evt 6, idx 199
    #pars["azi"] = 102
    #pars["d_zen"] = 82
    # evt 7, idx 199
    pars["azi"] = 103.7
    pars["d_zen"] = 83.6
    # evt 8, idx 3
    # pars["azi"] = 206.47761649
    # pars["d_zen"] =  55.83650682
    evt.plot_footprint_val_max()
    l_cost = pw.polar_wiener_lost_func_all_du(evt,pars, ant3d, rf_fft)
    pw.plot_polar_angle_max(
        evt,
        l_cost,
        3,
        1,
        band=[50, 90],
        title=f"Polar angle with with the 3 $\delta$Efield, weight by SNR\nDC2 simu",
    )
    pw.plot_polar_angle_max(
        evt,
        l_cost,
        3,
        0,
        band=[50,90],
        title=f"Polar angle with with the 3 voltage residu, weight by SNR\nDC2 simu",
    )
    pw.plot_polar_angle_max(
        evt,
        l_cost,
        2,
        0,
        band=[50,100],
        title=f"Polar angle with with voltage residu axis UP\nDC2 simu",
    )


def simu_trace(i_e, i_du):
    a_pol, d_sim, tr_ef = get_true_polar_angle(i_e)
    gadc = froot.get_handling3dtraces(add_path(f_dc2_adc), i_e)
    evt = convert_3dtrace(gadc)
    evt.info_shower = get_info_shower(d_sim)
    # Load instrument model : antenna
    ant3d = DetectorUnitAntenna3Axis(get_leff_default(pn_fmodel))
    # Load instrument model : RF chain
    rf_fft = read_TF3_fmt(pn_fmodel)


if __name__ == "__main__":
    i_e = 7
    # plot_dc2_event(i_e)
    plot_polar(i_e)
    # simu_trace(i_e,i_du)
    #estimate_polar_angle(i_e)
    plt.show()
