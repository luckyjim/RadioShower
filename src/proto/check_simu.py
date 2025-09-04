"""
Check simulation Efield to Voltage with polarization
"""

import copy
import pprint

import uproot
import numpy as np
import scipy.fft as sf

import grand.dataio.root_files as froot
import matplotlib.pyplot as plt
from grand.basis.traces_event import Handling3dTraces

import rshower.basis.efield_event as efe
import rshower.basis.traces_event as tre
from rshower.basis.frame import FrameDuFrameTan
from rshower.io.leff_fmt import get_leff_default
import rshower.io.rf_fmt as rfchain
from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from numpy import isin


# DATA
# No noise
P_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
F_efield = "efield_29-24992_L0_0000.root"
F_volt = "voltage_29-24992_L0_0000.root"
F_shower = "shower_29-24992_L0_0000.root"
# F_volt = "simu_dc2_pacman_L0_jmc.root"
F_adc = "adc_29-24992_L1_0000.root"

# MODEL
PN_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model/"


def select_event():
    dfrs = uproot.open(P_dc2 + F_shower)
    d_azi = dfrs["tshower"]["azimuth"].array(library="np")
    d_zen = dfrs["tshower"]["zenith"].array(library="np")
    a_idx = np.argwhere(np.logical_and.reduce((d_azi % 90 > 40, d_azi % 90 < 50, d_zen < 50)))


def max_algebra(array_1d):
    idx = np.argmax(np.abs(array_1d))
    return array_1d[idx]


def rotate_via_numpy(xy, radians):
    c, s = np.cos(radians), np.sin(radians)
    rot = np.array([[c, -s], [s, c]])
    return np.dot(rot, xy)


def relative_error_trace(tra, trb, s_tra="A", s_trb="B"):
    l_err = []
    for axis in range(3):
        idx = np.squeeze(np.argwhere(np.abs(tra[axis]) > 200))
        err = np.empty(len(idx), dtype=np.float32)
        err = 100 * (tra[axis, idx] - trb[axis, idx]) / tra[axis, idx]
        l_err.append(err)
        plt.figure()
        plt.title(f"axis {axis}")
        plt.plot(tra[axis], "y", label=s_tra)
        plt.plot(trb[axis], "b", label=s_trb)
        plt.plot(tra[axis] - trb[axis], "--", color="k", label=f"{s_tra}-{s_trb}")
        plt.grid()
        plt.legend()
        plt.figure()
        plt.title(f"Relative error trace {axis}")
        plt.hist(err, 70)
        plt.grid()
        plt.yscale("log")
        plt.xlabel("%")
    return l_err


def compare_simu_du_tan(i_e=0):
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_du = do_simu_in_du_frame(tref, d_simu, ant3d, rf_fft)
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_tan = do_simu_in_tan_frame(tref, d_simu, ant3d, rf_fft)
    ident_du = tr_du.idx2idt[0]
    ident_du = 102
    print(tr_du.idt2idx[ident_du])
    print(tr_tan.idt2idx[ident_du])
    tr_tan.plot_trace_du(ident_du)
    sim_du = tr_du.traces[tr_du.idt2idx[ident_du]]
    sim_tan = tr_tan.traces[tr_tan.idt2idx[ident_du]]
    tref.plot_trace_3d_idx(tref.idt2idx[ident_du])
    relative_error_trace(sim_du, sim_tan, "DU", "TAN")


def compare_simu_du_polar(i_e=0):
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_du = do_simu_in_du_frame(tref, d_simu, ant3d, rf_fft)
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_pol = do_simu_in_polar_frame(tref, d_simu, ant3d, rf_fft)
    ident_du = tr_du.idx2idt[0]
    ident_du = 100
    print(tr_du.idt2idx[ident_du])
    print(tr_pol.idt2idx[ident_du])
    tr_pol.plot_trace_du(ident_du)
    sim_du = tr_du.traces[tr_du.idt2idx[ident_du]]
    sim_pol = tr_pol.traces[tr_pol.idt2idx[ident_du]]
    tref.plot_trace_3d_idx(tref.idt2idx[ident_du])
    relative_error_trace(sim_du, sim_pol, "DU", "POLAR")


def compare_simu_polar_tan(i_e=0):
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_pol = do_simu_in_polar_frame(tref, d_simu, ant3d, rf_fft)
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_tan = do_simu_in_tan_frame(tref, d_simu, ant3d, rf_fft)
    ident_du = tr_pol.idx2idt[0]
    ident_du = 70
    print(tr_pol.idt2idx[ident_du])
    print(tr_tan.idt2idx[ident_du])
    tr_tan.plot_trace_du(ident_du)
    sim_pol = tr_pol.traces[tr_pol.idt2idx[ident_du]]
    sim_tan = tr_tan.traces[tr_tan.idt2idx[ident_du]]
    tref.plot_trace_3d_idx(tref.idt2idx[ident_du])
    relative_error_trace(sim_pol, sim_tan, "POLAR", "TAN")


def compare_simu_polar_dc2(i_e=0):
    tref2, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    tr_pol = do_simu_in_polar_frame(tref2, d_simu, ant3d, rf_fft)
    tr_dc2 = froot.get_handling3dtraces(P_dc2 + F_volt, i_e)
    tr_dc2.plot_footprint_val_max()
    ident_du = 70
    print(tr_pol.idt2idx[ident_du])
    print(tr_dc2.idt2idx[ident_du])
    sim_pol = tr_pol.traces[tr_pol.idt2idx[ident_du]]
    sim_dc2 = tr_dc2.traces[tr_dc2.idt2idx[ident_du]]
    relative_error_trace(sim_pol, sim_dc2, "POLAR", "DC2")
    return tr_pol


def compare_simu_tan_dc2(i_e=0):
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    trv = do_simu_in_tan_frame(tref, d_simu, ant3d, rf_fft)
    assert isinstance(tref, efe.HandlingEfield)
    assert isinstance(trv, tre.Handling3dTraces)
    tr_dc2 = froot.get_handling3dtraces(P_dc2 + F_volt, i_e)
    tr_dc2.plot_footprint_val_max()
    ident_du = trv.idx2idt[0]
    ident_du = 70
    print(trv.idt2idx[ident_du])
    print(tr_dc2.idt2idx[ident_du])
    trv.plot_trace_du(ident_du)
    tr_dc2.plot_trace_du(ident_du)
    sim_pol = trv.traces[trv.idt2idx[ident_du]]
    sim_dc2 = tr_dc2.traces[tr_dc2.idt2idx[ident_du]]
    relative_error_trace(sim_pol, sim_dc2, "TAN", "DC2")
    return trv


def compare_simu_du_dc2(i_e=0):
    tref2, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    return
    tr_du = do_simu_in_du_frame(tref2, d_simu, ant3d, rf_fft)
    tr_dc2 = froot.get_handling3dtraces(P_dc2 + F_volt, i_e)
    tr_dc2.plot_footprint_val_max()
    ident_du = 100
    # ident_du = tr_du.idx2idt[0]
    print(tr_du.idt2idx[ident_du])
    print(tr_dc2.idt2idx[ident_du])
    tr_dc2.plot_trace_du(ident_du)
    sim_du = tr_du.traces[tr_du.idt2idx[ident_du]]
    sim_dc2 = tr_dc2.traces[tr_dc2.idt2idx[ident_du]]
    relative_error_trace(sim_du, sim_dc2, "DU", "DC2")
    return tr_du


def loading_simu_file_in(i_e):
    """

    :param i_e:
    """
    ant3d = DetectorUnitAntenna3Axis(get_leff_default(PN_fmodel))
    # Load instrument model : RF chain
    rf_fft = rfchain.read_TF1_fmt(PN_fmodel)
    rfchain.plot_global_rf_chain_TF1(PN_fmodel)
    tref = froot.get_handling3dtraces(P_dc2 + F_efield, i_e)
    d_simu = froot.get_simu_parameters(P_dc2 + F_efield, i_e)
    pprint.pprint(d_simu)
    tref2 = convert_3dtrace_grandlib(tref, True)
    return tref2, d_simu, ant3d, rf_fft


def do_simu_in_polar_frame(tref, d_simu, ant3d, rf_fft):
    # 1. compute for each trace:
    #    1. efield in polar frame
    #    2. direction xmax at DU level
    # 2. for each trace
    #    1. set direction
    #    2. define Leff
    #    3. compute Volt for each axis
    assert isinstance(tref, efe.HandlingEfield)
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    out_freq = sf.rfftfreq(tref.get_size_trace(), 1e-6 / tref.f_samp_mhz[0])
    out_freq *= 1e-6
    ant3d.set_freq_out_mhz(out_freq)
    rf_out = rfchain.interpol_RF(rf_fft, out_freq)
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    tref.network.core_pos = d_simu["shower_core_pos"]
    # for idx in range(tref.get_nb_trace()):
    #     tref.network.du_pos[idx] = d_simu["shower_core_pos"]
    tref.remove_trace_low_signal(75)
    # tref.plot_footprint_val_max()
    polars, dir_angle, dir_vec = tref.get_polar_angle()
    tr_out = tref.copy(0)
    tr_out.__class__ = tre.Handling3dTraces
    assert isinstance(tr_out, tre.Handling3dTraces)
    tr_out.set_unit_axis("uVolt", "dir", "Volt")
    tr_out.name = "Simu polar"
    print(f"{tref.get_nb_trace()} DUs")
    for idx in range(tref.get_nb_trace()):
        cor_dir = dir_angle[idx]
        ant3d.set_dir_source(cor_dir)
        print(f"GRAND pos : {tref.network.du_pos[idx]}")
        print(f"Core pos : {tref.network.du_pos[idx]-tref.network.core_pos}")
        print(f"src dir: {np.rad2deg(cor_dir)}\n")
        ant3d.interp_leff.set_angle_polar(polars[idx])
        fft_voc = ant3d.get_resp_1d_efield_pol(sf.rfft(tref.ef_pol[idx]))
        tr_out.traces[idx] = sf.irfft(fft_voc * rf_out)
    # tr_out.plot_footprint_val_max()
    return tr_out


def do_simu_in_tan_frame(tref, d_simu, ant3d, rf_fft):
    # 1. compute for each trace:
    #    1. efield in polar frame
    #    2. direction xmax at DU level
    # 2. for each trace
    #    1. set direction
    #    2. define Leff
    #    3. compute Volt for each axis
    assert isinstance(tref, efe.HandlingEfield)
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    out_freq = sf.rfftfreq(tref.get_size_trace(), 1e-6 / tref.f_samp_mhz[0])
    out_freq *= 1e-6
    ant3d.set_freq_out_mhz(out_freq)
    rf_out = rfchain.interpol_RF(rf_fft, out_freq)
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    tref.network.core_pos = d_simu["shower_core_pos"]
    tref.remove_trace_low_signal(75)
    tref.plot_footprint_val_max()
    polars, dir_angle, dir_vec = tref.get_polar_angle()
    tr_out = tref.copy(0)
    tr_out.__class__ = tre.Handling3dTraces
    assert isinstance(tr_out, tre.Handling3dTraces)
    tr_out.set_unit_axis("uVolt", "dir", "Volt")
    tr_out.name = "Simu tangential"
    for idx in range(tref.get_nb_trace()):
        t_dutan = FrameDuFrameTan(dir_angle[idx])
        # Electric Field in TAN frame
        ef_tan = t_dutan.vec_to(tref.traces[idx], "TAN")
        if idx == 1480:
            print(tref.traces[idx].shape)
            print(ef_tan.shape)
            print(dir_angle[idx])
            tref.plot_trace_idx(idx)
            fig = plt.figure()
            title = "Trace in tangential frame"
            plt.title(title)
            plt.axis("equal")
            sca = plt.scatter(ef_tan[0], ef_tan[1], s=100)
            fig.colorbar(sca, label="")
            plt.grid()
        ant3d.set_dir_source(dir_angle[idx])
        fft_ef = sf.rfft(ef_tan[:2], axis=-1)
        fft_voc = ant3d.get_resp_2d_efield_tan(fft_ef)
        tr_out.traces[idx] = sf.irfft(fft_voc * rf_out)
    tr_out.plot_footprint_val_max()
    # tr_out.plot_trace_idx(165)
    return tr_out


def do_simu_in_tan_frame_all_polar(tref, d_simu, ant3d, rf_fft):
    # 1. compute for each trace:
    #    1. efield in polar frame
    #    2. direction xmax at DU level
    # 2. for each trace
    #    1. set direction
    #    2. define Leff
    #    3. compute Volt for each axis
    assert isinstance(tref, efe.HandlingEfield)
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    out_freq = sf.rfftfreq(tref.get_size_trace(), 1e-6 / tref.f_samp_mhz[0])
    out_freq *= 1e-6
    ant3d.set_freq_out_mhz(out_freq)
    rf_out = rfchain.interpol_RF(rf_fft, out_freq)
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    tref.network.core_pos = d_simu["shower_core_pos"]
    tref.remove_trace_low_signal(75)
    tref.plot_footprint_val_max()
    polars, dir_angle, dir_vec = tref.get_polar_angle()
    tr_out = tref.copy(0)
    tr_out.__class__ = tre.Handling3dTraces
    assert isinstance(tr_out, tre.Handling3dTraces)
    tr_out.set_unit_axis("uVolt", "dir", "Volt")
    tr_out.name = "Simu tangential"
    idx = tref.idt2idx[221]
    t_dutan = FrameDuFrameTan(dir_angle[idx])
    ant3d.set_dir_source(dir_angle[idx])
    ef_tan = t_dutan.vec_to(tref.traces[idx], "TAN")
    print(ef_tan.shape)
    deg_dir = np.rad2deg(dir_angle[idx])
    rot_event = Handling3dTraces(f"azi: {deg_dir[0]:.1f}, d_zen: {deg_dir[1]:.1f}")
    rot_event.set_unit_axis(r"$\mu$V", "dir", "Voltage")
    t_size = tref.get_size_trace()
    rot_event.init_traces(
        np.zeros((1, 3, t_size)), du_id=[tref.idx2idt[idx]], f_samp_mhz=tref.f_samp_mhz
    )
    rot_event.range_plot = [1600, 1900]
    max_ef = max_algebra(ef_tan[:2].ravel()) * 1.2
    max_ef = np.max(np.linalg.norm(ef_tan[:2], axis=0)) * 1.1
    print(max_ef)
    for a_pol in range(0, 360, 5):
        ef_tan_rot = rotate_via_numpy(ef_tan[:2], np.deg2rad(a_pol))
        # ef_tan_rot = ef_tan[:2]
        fft_ef = sf.rfft(ef_tan_rot, axis=-1)
        fft_voc = ant3d.get_resp_2d_efield_tan(fft_ef)
        rot_event.traces[0] = sf.irfft(fft_voc * rf_out)
        fig = plt.figure()
        plt.title(f"EField in plane wave with rotation {a_pol:03d} degree")
        plt.axis("equal")
        sca = plt.scatter(ef_tan_rot[0], ef_tan_rot[1], s=100)
        plt.xlabel("(vertical component) ${e_\\theta}$,  [$\mu$V/m]")
        plt.ylabel("(horizontal component) $e_\phi$,  [$\mu$V/m]")
        plt.xlim([-max_ef, max_ef])
        plt.ylim([-max_ef, max_ef])
        fig.colorbar(sca, label="")
        plt.grid()
        plt.savefig(f"data/EField_rot_{a_pol:03d}.png", bbox_inches="tight")
        amp_v = 14000
        rot_event.ylim_plot = [-amp_v, amp_v]
        rot_event.plot_trace_idx(0)
        plt.savefig(f"data/Voltage_rot_{a_pol:03d}.png", bbox_inches="tight")
    # tr_out.plot_trace_idx(165)
    return tr_out


def do_simu_in_du_frame(tref, d_simu, ant3d, rf_fft):
    # 1. compute for each trace:
    #    1. efield in polar frame
    #    2. direction xmax at DU level
    # 2. for each trace
    #    1. set direction
    #    2. define Leff
    #    3. compute Volt for each axis
    assert isinstance(tref, efe.HandlingEfield)
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    out_freq = sf.rfftfreq(tref.get_size_trace(), 1e-6 / tref.f_samp_mhz[0])
    out_freq *= 1e-6
    ant3d.set_freq_out_mhz(out_freq)
    rf_out = rfchain.interpol_RF(rf_fft, out_freq)
    tref.set_xmax(d_simu["FIX_xmax_pos"])
    tref.network.core_pos = d_simu["shower_core_pos"]
    tref.remove_trace_low_signal(75)
    tref.plot_footprint_val_max()
    tr_out = tref.copy(0)
    tr_out.__class__ = tre.Handling3dTraces
    assert isinstance(tr_out, tre.Handling3dTraces)
    tr_out.set_unit_axis("uVolt", "dir", "Volt")
    tr_out.name = "Simu DU"
    for idx in range(tref.get_nb_trace()):
        ant3d.set_pos_source(tref.xmax)
        fft_ef = sf.rfft(tref.traces[idx], axis=-1)
        fft_voc = ant3d.get_resp_3d_efield_du(fft_ef)
        tr_out.traces[idx] = sf.irfft(fft_voc * rf_out)
    # tr_out.plot_footprint_val_max()
    # tr_out.plot_trace_idx(165)
    return tr_out


def plot_input_data(i_e=0):
    tref = froot.get_handling3dtraces(P_dc2 + F_efield, i_e)
    d_simu = froot.get_simu_parameters(P_dc2 + F_efield, i_e)
    pprint.pprint(d_simu)
    assert isinstance(tref, Handling3dTraces)
    trv = froot.get_handling3dtraces(P_dc2 + F_volt, i_e)
    assert isinstance(trv, Handling3dTraces)
    tradc = froot.get_handling3dtraces(P_dc2 + F_adc, i_e)
    tref.plot_footprint_val_max()
    trv.plot_footprint_val_max()
    tradc.plot_footprint_val_max()
    tref2 = convert_3dtrace_grandlib(tref, True)
    assert isinstance(tref2, efe.HandlingEfield)
    tref2.remove_trace_low_signal(75)
    tref2.set_xmax(d_simu["FIX_xmax_pos"])
    tref2.network.core_pos = d_simu["shower_core_pos"]
    tref3 = copy.deepcopy(tref2)
    tref4 = copy.deepcopy(tref2)
    assert isinstance(tref3, efe.HandlingEfield)
    tref3.apply_bandpass(80, 220, True)
    tref3.remove_trace_low_signal(75)
    tref2.keep_only_trace_with_ident(tref3.idx2idt)
    tref2.plot_polar_angle()
    tref3.plot_polar_angle()
    tref4.apply_bandpass(80, 220, False)
    tref4.keep_only_trace_with_ident(tref3.idx2idt)
    tref4.get_polar_angle()
    plt.figure()
    plt.title("Difference polar angle Causal")
    plt.hist(np.rad2deg(tref2.polar_angle_rad - tref3.polar_angle_rad))
    plt.xlabel("Degree")
    plt.grid()
    plt.figure()
    plt.title("Difference polar angle No Causal")
    plt.hist(np.rad2deg(tref2.polar_angle_rad - tref4.polar_angle_rad))
    plt.xlabel("Degree")
    plt.grid()
    tref4.plot_footprint_val_max()


def view_effect_polar(i_e):
    tref, d_simu, ant3d, rf_fft = loading_simu_file_in(i_e)
    do_simu_in_tan_frame_all_polar(tref, d_simu, ant3d, rf_fft)


if __name__ == "__main__":
    i_e = 774  # 75km, azi 306°
    # i_e = 766  #  6 km
    i_e = 764  # 70km, azi=107
    # i_e = 762 # 8km
    # i_e = 759 # 234km, azi=11, >40
    # i_e = 755  # ok !! pacmac
    # i_e = 755
    # plot_input_data(i_e)
    # compare_simu_du_tan(i_e)
    # compare_simu_du_polar(i_e)
    # compare_simu_polar_dc2(i_e)
    # compare_simu_tan_dc2(i_e)
    compare_simu_du_dc2(i_e)
    # view_effect_polar(i_e)
    plt.show()
