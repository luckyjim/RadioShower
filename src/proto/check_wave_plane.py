"""
Created on 13 sept. 2024

@author: jcolley
"""
import pickle
import os.path
import pprint

import numpy as np
import grand.dataio.root_files as froot

from rshower.basis.coord import nwu_cart_to_sph_one, nwu_cart_to_dir
from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield, plt, fit_vec_linear_polar_l2
from rshower.io.rf_fmt import read_TF3_fmt
from rshower.io.leff_fmt import get_leff_default
from rshower.io.events.grand_io_fmt import get_info_shower
from rshower.io.shower.zhaires_master import ZhairesMaster

from rshower.model.ant_resp import DetectorUnitAntenna3Axis


# GRAND
path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
f_dc2_adc = "adc_29-24992_L0_0000.root"
f_dc2_ef = "efield_29-24992_L0_0000.root"

# ZHAIRES
pd_zhaires = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_2.784_74.8_0.0_1"
)
pd_zhaires = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_0.122_74.8_0.0_1"
)
pd_zhaires = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.981_74.8_0.0_1"
)
pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"


def load_zhaires(f_simu=""):
    if f_simu == "":
        f_simu = pd_zhaires
    fzs = ZhairesMaster(f_simu)
    tr3d = fzs.get_object_3dtraces()
    tr3d.plot_footprint_val_max()
    return tr3d, fzs.get_simu_info()


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
    tr_ef.set_xmax(d_sim["FIX_xmax_pos"])
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
    tr_ef.set_xmax(d_sim["FIX_xmax_pos"])
    tr_ef.info_shower = get_info_shower(d_sim)
    tr_ef.set_unit_axis(gef.unit_trace, "dir", gef.type_trace)
    normal = solve_homogen_system(tr_ef.traces[i_du].T)
    normal /= np.linalg.norm(normal)
    print(f"vec normal plan wave: {normal}")
    pol_est, idx_hb = fit_vec_linear_polar_l2(tr_ef.traces[i_du])
    print(f"vec polar: {pol_est}")
    print(np.dot(normal, pol_est))
    polars, dir_angle = tr_ef.get_polar_angle(True)


def check_acp(gef):
    gef.plot_footprint_val_max()
    gef.remove_trace_low_signal(25)
    gef.plot_footprint_val_max()
    if not isinstance(gef, HandlingEfield):
        tr_ef = HandlingEfield(gef.name)
        print(gef.f_samp_mhz.shape)
        tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
        tr_ef.init_network(gef.network.du_pos)
    else:
        tr_ef = gef
    #
    l_dif = []
    l_max = []
    v_pol, v_dir_src = tr_ef.get_pca()
    print(v_pol.shape)
    max_norm = tr_ef.get_max_norm()
    for idx in range(tr_ef.get_nb_trace()):
        pol_est, idx_hb = fit_vec_linear_polar_l2(tr_ef.traces[idx])
        print(pol_est.shape)
        diff_angle = np.rad2deg(np.arccos(np.dot(v_pol[idx], pol_est)))
        l_dif.append(diff_angle)
        l_max.append(max_norm[idx])
        print(idx, max_norm[idx], diff_angle)
    plt.figure()
    plt.hist(l_dif)
    plt.figure()
    plt.plot(l_max, l_dif, "*")
    plt.xlabel("max pic")
    plt.ylabel("diff angle [degree]")
    plt.grid()


def check_direction():
    fgr = froot.get_file_event(add_path(f_dc2_ef))
    l_dist = [[], [], [], [], []]
    l_dif_azi = [[], [], [], [], []]
    l_dif_zen = [[], [], [], [], []]
    l_dif_dir = [[], [], [], [], []]
    for i_e in range(300):
        fgr.load_event_idx(i_e)
        gef = fgr.get_obj_handling3dtraces()
        d_sim = fgr.get_simu_parameters()
        # pprint.pprint(d_sim)
        dif_azi, dif_zen, dif_dir, dist = check_direction_evt_noread(i_e, gef, d_sim)
        try:
            i_dist = int(dist / 50)
        except:
            continue
        if i_dist > 4:
            i_dist = 4
        l_dist[i_dist] += [dist]
        l_dif_azi[i_dist] += dif_azi.tolist()
        l_dif_zen[i_dist] += dif_zen.tolist()
        l_dif_dir[i_dist] += dif_dir.tolist()
        print(i_e, len(l_dif_azi))
    # print(l_dist)
    d_res = {}
    d_res["l_dist"] = l_dist
    d_res["l_dif_azi"] = l_dif_azi
    d_res["l_dif_zen"] = l_dif_zen
    d_res["l_dif_dir"] = l_dif_dir
    with open("res_optical_dif_test.pkl", "wb") as fres:
        pickle.dump(d_res, fres)

    plt.figure()
    plt.title("Histogram diff azimuth, bin 1")
    plt.hist(l_dif_azi[0], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff zenith, bin 1")
    plt.hist(l_dif_zen[0], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff azimuth, bin 5")
    plt.hist(l_dif_azi[4], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff zenith, bin 5")
    plt.hist(l_dif_zen[4], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.boxplot(l_dif_dir, showfliers=False)
    plt.grid()
    plt.title(
        f"Difference of direction of Xmax at DU level,\nbetween normal to Efield and straight line."
    )
    # plt.ylim([0,16])
    plt.xlabel(f"Bin number.\nFile : {f_dc2_ef}")
    plt.ylabel("Difference of direction [degree]")
    #
    plt.figure()
    plt.boxplot(l_dist, showfliers=False)
    plt.grid()
    plt.title("Bin Xmax, interval of distance")
    plt.xlabel("Bin number")
    plt.ylabel("Distance Xmax [km]")
    #
    plt.figure()
    plt.grid()
    plt.title("Bin Xmax, number of event")
    len_bin = [len(l_bin) for l_bin in l_dist]
    plt.plot(1 + np.arange(len(l_dist)), len_bin, "*")
    plt.xlabel("Bin number")
    plt.ylabel("Number of event in bin")
    #
    plt.figure()
    plt.grid()
    plt.title("Bin Xmax, number of DU")
    len_bin = [len(l_bin) for l_bin in l_dif_dir]
    plt.plot(1 + np.arange(len(l_dist)), len_bin, "*")
    plt.xlabel("Bin number")
    plt.ylabel("Number of DU in bin")


def check_direction_slow():
    # fgr = froot.get_file_event(add_path(f_dc2_ef))
    l_dist = [[], [], [], [], []]
    l_dif_azi = [[], [], [], [], []]
    l_dif_zen = [[], [], [], [], []]
    l_dif_dir = [[], [], [], [], []]
    for i_e in range(200):
        # fgr.load_event_idx(i_e)
        gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
        d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
        # pprint.pprint(d_sim)
        dif_azi, dif_zen, dif_dir, dist = check_direction_evt_noread(i_e, gef, d_sim)
        try:
            i_dist = int(dist / 50)
        except:
            continue
        if i_dist > 4:
            i_dist = 4
        l_dist[i_dist] += [dist]
        l_dif_azi[i_dist] += dif_azi.tolist()
        l_dif_zen[i_dist] += dif_zen.tolist()
        l_dif_dir[i_dist] += dif_dir.tolist()
        print(i_e, len(l_dif_azi))
    print(l_dist)
    plt.figure()
    plt.title("Histogram diff azimuth, bin 1")
    plt.hist(l_dif_azi[0], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff zenith, bin 1")
    plt.hist(l_dif_zen[0], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff azimuth, bin 5")
    plt.hist(l_dif_azi[4], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.title("Histogram diff zenith, bin 5")
    plt.hist(l_dif_zen[4], 25)
    plt.xlabel("degree")
    plt.yscale("log")
    plt.grid()
    #
    plt.figure()
    plt.boxplot(l_dif_dir, showfliers=False)
    plt.grid()
    plt.title(
        "Difference of direction of Xmax at DU level,\nbetween optical and geometrical calculation."
    )
    plt.xlabel("Bin number")
    plt.ylabel("Difference of direction [degree]")
    #
    plt.figure()
    plt.boxplot(l_dist, showfliers=False)
    plt.grid()
    plt.xlabel("Bin number")
    plt.ylabel("Distance Xmax [km]")


def check_direction_evt(i_e=0, f_plot=False):
    # 224 , 2 stations
    # 102 , 1 outlier
    # 120, OK
    # 1560, 20
    # 200, 73 OK
    # 210 , 19, bof
    # i_e = 220
    # # plot_polar(i_e)
    fpn = add_path(f_dc2_ef)
    # print(f"ROOT file: {fpn}")
    print(f"Event index: {i_e}")
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    # print(gef.__dict__)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)
    pprint.pprint(d_sim)
    return check_direction_evt_noread(i_e, gef, d_sim, f_plot)


def check_direction_evt_noread(i_e, gef, d_sim, f_plot=False):
    tr_ef = HandlingEfield(gef.name)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.info_shower = gef.info_shower
    tr_ef.unit_trace = gef.unit_trace
    tr_ef.type_trace = gef.type_trace
    tr_ef.network.core_pos = d_sim["shower_core_pos"]
    tr_ef.network.xmax_pos = d_sim["FIX_xmax_pos"]
    dist_xmax_km = np.linalg.norm(tr_ef.network.core_pos - tr_ef.network.xmax_pos) / 1000
    # tr_ef.network.core_pos = d_sim["xmax_pos_shc"]
    if f_plot:
        tr_ef.plot_footprint_val_max()
    try:
        tr_ef.remove_trace_low_signal(75)
    except:
        return np.array([]), np.array([]), np.array([]), np.array([])
    # print(f"Remove trace under 20 uv/m")
    _, v_dir_src = tr_ef.get_polar_normal_vec()
    tr_ef.set_xmax(tr_ef.network.xmax_pos)
    _, dir_angle, dir_vec = tr_ef.get_polar_angle(True)
    dir_angle = dir_angle.transpose()
    dif_angle = np.zeros(tr_ef.get_nb_trace(), dtype=np.float32)
    for idx in range(tr_ef.get_nb_trace()):
        dif_angle[idx] = np.dot(dir_vec[idx], v_dir_src[idx])
    dif_angle = np.rad2deg(np.arccos(dif_angle))
    dir_opt = np.rad2deg(nwu_cart_to_dir(v_dir_src.transpose()))

    #
    dif_azi = dir_opt[0] - dir_angle[0]
    dif_zen = dir_opt[1] - dir_angle[1]
    if f_plot:
        tr_ef.get_tmax_vmax()
        tr_ef.plot_footprint_time_max()
        tr_ef.plot_footprint_val_max()
        tr_ef.network.plot_footprint_1d(
            dir_opt[0], title="azimuth", traces=tr_ef, scale="lin", unit="deg"
        )
        tr_ef.network.plot_footprint_1d(
            dir_opt[1], title="dist_zenith", traces=tr_ef, scale="lin", unit="deg"
        )
        tr_ef.network.plot_footprint_1d(
            dif_azi, title="diff azimuth", traces=tr_ef, scale="lin", unit="deg"
        )
        tr_ef.network.plot_footprint_1d(
            dif_zen, title="diff dist_zenith", traces=tr_ef, scale="lin", unit="deg"
        )
        tr_ef.network.plot_footprint_1d(
            dif_angle, title="diff direction", traces=tr_ef, scale="lin", unit="deg"
        )
        pprint.pprint(d_sim)
        plt.figure()
        plt.title("Histogram diff azimuth")
        plt.hist(dif_azi, 25)
        plt.grid()
        plt.figure()
        plt.title("Histogram diff zenith")
        plt.hist(dif_zen, 25)
        plt.grid()
        plt.figure()
        plt.boxplot(dif_angle)
        plt.grid()
    return dif_azi, dif_zen, dif_angle, dist_xmax_km


def check_dc2_xmax():
    # 224 , 2 stations
    # 237
    i_e = 237
    # # plot_polar(i_e)
    fpn = add_path(f_dc2_ef)
    print(f"ROOT file: {fpn}")
    print(f"Event index: {i_e}")
    gef = froot.get_handling3dtraces(add_path(f_dc2_ef), i_e)
    d_sim = froot.get_simu_parameters(add_path(f_dc2_ef), i_e)

    tr_ef = HandlingEfield(gef.name)
    print(gef.f_samp_mhz.shape)
    tr_ef.init_traces(gef.traces, gef.idx2idt, gef.t_start_ns, gef.f_samp_mhz)
    tr_ef.init_network(gef.network.du_pos)
    tr_ef.plot_footprint_val_max()
    tr_ef.remove_trace_low_signal(20)
    print(f"Remove trace under 20 uv/m")
    tr_ef.plot_footprint_val_max()
    nb_rm = int(tr_ef.get_nb_trace() * 0.03)
    if nb_rm < 2:
        nb_rm = 6
    print(f"Nb rm {nb_rm}")
    for idx in range(nb_rm):
        print(f"Estimation avec {tr_ef.get_nb_trace()} DUs")
        xmax, v_res = tr_ef.estimate_xmax_with_wave_plan()
        if idx == 0:
            plt.figure()
            plt.hist(v_res)
            plt.figure()
            plt.plot(v_res, "*")
        sph = nwu_cart_to_sph_one(xmax - d_sim["shower_core_pos"])
        print(f"Core-Xmax dist: {sph[2]/1000:.1f}")
        print(f"Core-Xmax (azi, zen): ({np.rad2deg(sph[0]):.2f}, {np.rad2deg(sph[1]):.2f})")
        if idx != (nb_rm - 1):
            idx_max = np.argmax(v_res)
            l_ok = list(range(tr_ef.get_nb_trace()))
            l_ok.remove(idx_max)
            tr_ef.keep_only_trace_with_index(l_ok)
        print()
    # tr_ef.network.plot_footprint_1d(v_res,"Residu",tr_ef,"lin")
    plt.figure()
    plt.hist(v_res)
    plt.figure()
    plt.plot(v_res, "*")
    sph = nwu_cart_to_sph_one(xmax)
    # print(f"Xmax dist: {sph[2]/1000:.1f}")
    # print(f"Xmax (azi, zen): ({np.rad2deg(sph[0]):.2f}, {np.rad2deg(sph[1]):.2f})")
    #
    print("True Values:")
    core_xmax = d_sim["FIX_xmax_pos"] - d_sim["shower_core_pos"]
    sph = nwu_cart_to_sph_one(core_xmax)
    print(f"Core-Xmax dist: {sph[2]/1000:.1f}")
    print(f"Core-Xmax (azi, zen): ({np.rad2deg(sph[0]):.2f}, {np.rad2deg(sph[1]):.2f})")
    pprint.pprint(d_sim)


def check_zhaires_xmax():
    tr3d, d_simu = load_zhaires()
    assert isinstance(tr3d, HandlingEfield)
    tr3d.get_tmax_vmax()
    tr3d.plot_footprint_val_max()
    # tr3d.remove_trace_low_signal(100)
    tr3d.get_tmax_vmax(True, "auto")
    tr3d.plot_footprint_val_max()
    for idx in range(10):
        xmax, v_res = tr3d.estimate_xmax_with_wave_plan()
        plt.figure()
        plt.hist(v_res)
        plt.figure()
        plt.plot(v_res, "*")
        sph = nwu_cart_to_sph_one(xmax)
        print(f"Xmax dist: {sph[2]}")
        print(f"Xmax (azi, zen): ({np.rad2deg(sph[0]):.1f}, {np.rad2deg(sph[1]):.1f})")
        idx_max = np.argmax(v_res)
        l_ok = range(tr3d.get_nb_trace())
        l_ok.remove(idx_max)
        tr3d.keep_only_trace_with_index(l_ok)


if __name__ == "__main__":
    # check_zhaires_xmax()
    # check_dc2_xmax()
    check_direction_evt(399, True)
    # check_direction()
    # check_direction_slow()
    plt.show()
