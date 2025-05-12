import numpy as np
import grand.dataio.root_files as froot

import rshower.basis.frame as frame
import rshower.basis.coord as coord
from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield

from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib

P_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
F_efield = "efield_29-24992_L0_0000.root"


def read_dc2(i_e):
    tref_gl = froot.get_handling3dtraces(P_dc2 + F_efield, i_e)
    d_simu = froot.get_simu_parameters(P_dc2 + F_efield, i_e)
    tref = convert_3dtrace_grandlib(tref_gl, True)
    return tref, d_simu


def collect_polar_parameters(tref, d_simu):
    """
    Collect polar parameters from the event
    :param tref:
    :param d_simu:
    :return:
    """
    assert isinstance(tref, HandlingEfield)
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    threshold = 30
    tref.remove_trace_low_signal(threshold)
    if tref.get_nb_trace() == 0:
        return None
    polars, dir_angle, _ = tref.get_polar_angle()
    dxmax = np.linalg.norm(tref.network.du_pos - tref.network.xmax_pos, axis=-1)
    trshw = frame.FrameNetFrameShower()
    azi = d_simu["azimuth"]
    zen = d_simu["zenith"]
    core = d_simu["shower_core_pos"]
    xc_pos = core - tref.network.xmax_pos
    inc = np.deg2rad(d_simu["magnetic_field"][0])
    trshw.init_v_inc(xc_pos, inc, tref.network.xmax_pos)
    pos_sh = trshw.pos_to(tref.network.du_pos.T, "SHW")
    core_sh = trshw.pos_to(core, "SHW")
    # print("======== Rot")
    # print(trshw.rot_b2a)
    # print("======== res pos")
    # print(pos_sh)
    # print("======== res core")
    # print(core_sh)
    # print(core.shape)
    # print("========")
    angle_sh = coord.nwu_cart_to_dir(pos_sh)
    return np.rad2deg(polars), np.rad2deg(dir_angle), dxmax, np.rad2deg(angle_sh)


def collect_polar_parameters_cart(tref, d_simu):
    """
    Collect polar parameters from the event
    :param tref:
    :param d_simu:
    :return:
    """
    assert isinstance(tref, HandlingEfield)
    print(d_simu)
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    threshold = 30
    tref.remove_trace_low_signal(threshold)
    if tref.get_nb_trace() == 0:
        return None
    dmax = np.linalg.norm(d_simu["FIX_xmax_pos_grandlib"])
    print("dmax: ", dmax)
    if dmax < 80000:
        return None
    polars, _, dir_u = tref.get_polar_angle()
    dxmax = np.linalg.norm(tref.network.du_pos - tref.network.xmax_pos, axis=-1)
    trshw = frame.FrameNetFrameShower()
    azi = d_simu["azimuth"]
    zen = d_simu["zenith"]
    core = d_simu["shower_core_pos"]
    xc_pos = core - tref.network.xmax_pos
    inc = np.deg2rad(d_simu["magnetic_field"][0])
    trshw.init_v_inc(xc_pos, inc, tref.network.xmax_pos)
    pos_sh = trshw.pos_to(tref.network.du_pos.T, "SHW")
    assert pos_sh.shape[0] == 3
    # print(pos_sh)
    # print("\nnorm: ",np.linalg.norm(pos_sh,axis=0))
    pos_sh /= np.linalg.norm(pos_sh, axis=0)
    # print("\nnorm: ",np.linalg.norm(pos_sh,axis=0))
    # print(pos_sh)
    return np.rad2deg(polars), dir_u, dxmax, pos_sh


def test_polar_parameters():
    cpt = 0
    for i_e in range(1000):
        tref, d_simu = read_dc2(i_e)
        res = collect_polar_parameters_cart(tref, d_simu)
        if res is None:
            continue
        polars, dir_angle, dxmax, angle_sh = res
        cpt += len(polars)
        print("Polar parameters: \n", polars)
        print("Direction angle: \n", dir_angle)
        print("Distance to xmax: ", dxmax)
        print("Angle in SHW frame: \n", angle_sh.T)
    print(cpt)


test_polar_parameters()
