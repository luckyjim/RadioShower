"""
Created on 13 sept. 2024

@author: jcolley
"""

import os.path

import numpy as np
import grand.dataio.root_files as froot

from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield, plt, fit_vec_linear_polar_l2
from rshower.io.rf_fmt import read_TF3_fmt
from rshower.io.leff_fmt import get_leff_default
from rshower.io.events.grand_trigged import get_info_shower
from rshower.model.ant_resp import DetectorUnitAntenna3Axis


path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS-AN_0000/"
f_dc2_adc = "adc_5388-23832_L1_0000.root"
f_dc2_ef = "efield_5388-23832_L0_0000.root"
pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"


def add_path(f_name):
    return path_dc2 + f_name


def plot_polar(i_e):
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    print(d_sim)
    # print(gef.network.du_pos)
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.set_xmax(d_sim["xmax_site_level"])
    tr_ef.set_xmax(d_sim["xmax_pos_shc"])
    tr_ef.info_shower = get_info_shower(d_sim)
    tr_ef.set_unit_axis(gef.unit_trace, "dir", gef.type_trace)
    tr_ef.plot_footprint_4d_max()
    tr_ef.plot_polar_angle()
    i_du = 0
    tr_ef.plot_trace_3d_idx(i_du)
    tr_ef.plot_trace_idx(i_du)
    tr_ef.plot_trace_tan(i_du)
    # tr_ef.apply_bandpass(70, 190, False)


def solve_homogen_system(mat_coef):
    assert mat_coef.shape[1] == 3
    m_ata = np.matmul(mat_coef.T, mat_coef)
    w_p, vec_p = np.linalg.eig(m_ata)
    # logger.debug(f"{w_p}")
    # logger.debug(f"{vec_p}")
    i_nor = np.argmin(w_p)
    i_pol = np.argmax(w_p)
    vec_nor = vec_p[:, i_nor]
    print(f"vec_pol eigen :\n {vec_nor}\n{w_p}")
    res = np.matmul(mat_coef, vec_nor)
    plt.figure()
    plt.title("Residu")
    plt.yscale("log")
    plt.grid()
    plt.hist(res)
    print("check norm.pol =0:")
    print(np.dot(vec_p[:, i_nor], vec_p[:, i_pol]))
    print(f"Vec pol: {vec_p[:, i_pol]}")
    return vec_nor


def check_wave_plan(i_e):
    i_du = 3
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    print(d_sim)
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.set_xmax(d_sim["xmax_site_level"])
    # tr_ef.set_xmax(d_sim["xmax_pos_shc"])
    tr_ef.info_shower = get_info_shower(d_sim)
    tr_ef.set_unit_axis(gef.unit_trace, "dir", gef.type_trace)
    normal = solve_homogen_system(tr_ef.traces[i_du].T)
    normal /= np.linalg.norm(normal)
    print(f"vec normal plan wave: {normal}")
    pol_est, idx_hb = fit_vec_linear_polar_l2(tr_ef.traces[i_du])
    print(f"vec polar: {pol_est}")
    print(np.dot(normal, pol_est))
    polars, dir_angle = tr_ef.get_polar_angle_efield(True)


def check_acp(gef):
    i_du = 2
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    v_pol, v_dir_src = tr_ef._efield_acp()
    max_norm = tr_ef.get_max_norm()
    for idx in range(tr_ef.get_nb_du()):
        pol_est, idx_hb = fit_vec_linear_polar_l2(tr_ef.traces[idx])
        diff_angle = np.arccos(np.dot(v_pol[idx], pol_est))
        print(idx, max_norm[idx], np.rad2deg(diff_angle))
    tr_ef.plot_footprint_val_max()


if __name__ == "__main__":
    i_e = 9
    # plot_polar(i_e)
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    check_acp(gef)
    check_wave_plan(i_e)
    plt.show()
