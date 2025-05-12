import pprint
import logging
from logging import getLogger

logger = getLogger(__name__)

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


def collect_polar_parameters_cart(tref, d_simu, threshold=30):
    """
    Collect polar parameters from the event
    :param tref:
    :param d_simu:
    :return:
    """
    assert isinstance(tref, HandlingEfield)
    pprint.pprint(d_simu)
    logger.info(f"Nb DU : {tref.get_nb_trace()}")
    tref.set_xmax(d_simu["FIX_xmax_pos_grandlib"])
    tref.remove_trace_low_signal(threshold)
    if tref.get_nb_trace() == 0:
        print("Pass signal too low")
        return None
    dmax = np.linalg.norm(d_simu["FIX_xmax_pos_grandlib"])
    logger.info(f"Nb DU filter: {tref.get_nb_trace()}, dmax: {dmax:.1f}")
    if dmax < 30000:
        print("Pass too near")
        return None
    polars_r, _, dir_u = tref.get_polar_angle()
    i_sort = np.argsort(polars_r)
    pol_deg = np.rad2deg(polars_r[i_sort])
    logger.info(f"\n{pol_deg}")
    print(f"Min: {pol_deg[0]:.2f} Max:  {pol_deg[-1]:.2f}")
    print(f"Delta polar : {pol_deg[-1]-pol_deg[0]}, nb du : {tref.get_nb_trace()}")
    l_iok = [0]
    # take DU each 1 degree of polar angle
    for idx, pol in enumerate(pol_deg[1:]):
        if pol - pol_deg[l_iok[-1]] > 0.9:
            l_iok.append(idx + 1)
            print(f"add {pol} {pol_deg[idx+1]}")
    logger.info(l_iok)
    # Select DU
    i_sample = i_sort[l_iok]
    tref.keep_only_trace_with_index(i_sample)
    assert tref.network.get_nb_du() == len(i_sample)
    assert tref.network.get_nb_du() == tref.network.du_pos.shape[0]
    polars = polars_r[i_sample]
    dir_u = dir_u[i_sample]
    # Distance DU Xmax
    dxmax = np.linalg.norm(tref.network.du_pos - tref.network.xmax_pos, axis=-1)
    # DU coordinate in shower frame
    core = d_simu["shower_core_pos"]
    xc_pos = core - tref.network.xmax_pos
    inc = np.deg2rad(d_simu["magnetic_field"][0])
    trshw = frame.FrameNetFrameShower()
    trshw.init_v_inc(xc_pos, inc, tref.network.xmax_pos)
    pos_sh = trshw.pos_to(tref.network.du_pos.T, "SHW")
    assert pos_sh.shape[1] == len(i_sample)
    pos_sh /= np.linalg.norm(pos_sh, axis=0)
    logger.info("End of collect")
    return dxmax, dir_u, pos_sh.T, polars


def test_polar_parameters():
    cpt = 0
    l_polar_par = []
    l_size = []
    for i_e in range(1000):
        print(f"===================== evt {i_e}")
        tref, d_simu = read_dc2(i_e)
        res = collect_polar_parameters_cart(tref, d_simu)
        if res is None:
            continue
        dxmax, dir_net, dir_sh, polars = res
        l_polar_par.append(res)
        l_size.append(len(polars))
        cpt += l_size[-1]
        print("Distance to xmax: ", dxmax)
        print("Polar angle: \n", np.rad2deg(polars))
        print("Direction Xmax: \n", dir_net)
        print("Direction DU : \n", dir_sh)
    print(f"collect {cpt} points")
    logger.info("convert to numpy")
    polar_par = np.empty((cpt, 8), dtype=np.float64)
    idx = 0
    for size, par in zip(l_size, l_polar_par):
        polar_par[idx : idx + size, 0] = par[0]
        polar_par[idx : idx + size, 1:4] = par[1]
        polar_par[idx : idx + size, 4:7] = par[2]
        polar_par[idx : idx + size, 7] = par[3]
        idx += size
    logger.info(polar_par[cpt-1])
    del l_size, l_polar_par
    print(f"Nb point polar: {cpt}")
    np.save(f"par_model_{F_efield.split('.')[0]}", polar_par)


TPL_FMT_LOGGER = "%(asctime)s %(levelname)5s [%(name)s %(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=TPL_FMT_LOGGER)
test_polar_parameters()
