"""
Colley Jean-Marc, CNRS/IN2P3/LPNHE
"""

import matplotlib.pyplot as plt

import grand.dataio.root_files as froot

import rshower.basis.coord as coord
from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.frame import *
from rshower.basis.efield_event import HandlingEfield
from rshower.basis.traces_event import Handling3dTraces
from numba.core.types import none


P_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAires-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
F_efield = "efield_29-24992_L0_0000.root"


def check_dc2(i_e):
    tref_gl = froot.get_handling3dtraces(P_dc2 + F_efield, i_e)
    d_simu = froot.get_simu_parameters(P_dc2 + F_efield, i_e)
    tref = convert_3dtrace_grandlib(tref_gl, True)
    print(d_simu)
    tref.set_xmax(d_simu["FIX_xmax_pos"])
    tref.network.core_pos = d_simu["shower_core_pos"]
    assert isinstance(tref, HandlingEfield)
    tref.plot_footprint_val_max()
    trshw = FrameNetFrameShower()
    zen = d_simu["zenith"]
    azi = d_simu["azimuth"]
    print(zen, azi)
    dir_x = coord.nwu_sph_to_cart(np.array([np.deg2rad(azi), np.deg2rad(zen), 1.0]))
    inc  = np.deg2rad(d_simu["magnetic_field"][0])
    trshw.init_v_inc(-dir_x, inc, d_simu["FIX_xmax_pos"])
    print(-dir_x)
    #trshw.init_v_inc(-dir_x, 0)
    print(tref.network.du_pos.shape)
    pos_sh = trshw.pos_to(tref.network.du_pos.T, "SHW")
    print(pos_sh.shape)
    print(pos_sh.T[:3])
    tref_sh = tref.copy()
    tref_sh.network.du_pos = pos_sh.T
    tref_sh.network.xmax_pos = None
    tref_sh.network.core_pos = None
    tref_sh.network.axis_key = "SHW"
    tref_sh.plot_footprint_val_max()


def check_frame_to_shower():
    aa = np.linspace(0, 2 * 3.14159, 20)
    elp2 = np.array([20 * np.cos(aa), 50 * np.sin(aa)])
    pos = np.zeros((3, 20), dtype=np.float32)
    pos[:2] = elp2
    plt.figure()
    plt.title("ellipse in network frame")
    plt.grid()
    plt.xlabel("x => North")
    plt.ylabel("y => West")
    plt.plot(pos[0], pos[1], "*")
    plt.axis("equal")
    zenith = 90 - np.rad2deg(np.arcsin(20 / 50))
    print(zenith)
    dir = coord.nwu_sph_to_cart(np.array([np.deg2rad(90), np.deg2rad(zenith), 1.0]))
    trshw = FrameNetFrameShower()
    trshw.init_v_inc(-dir, 0)
    pos_sh = trshw.pos_to(pos, "SHW")
    print(pos_sh.shape)
    plt.figure()
    plt.title("ellipse in shower frame")
    plt.xlabel("vxB")
    plt.ylabel("vx(vxB)")
    plt.plot(pos_sh[0], pos_sh[1], "*")
    plt.axis("equal")
    plt.grid()


def check_frame_shower2net():
    aa = np.linspace(0, 2 * 3.14159, 20)
    elp2 = np.array([50 * np.cos(aa), 50 * np.sin(aa)])
    pos = np.zeros((3, 20), dtype=np.float32)
    pos[:2] = elp2
    plt.figure()
    plt.title("circle in Shower frame")
    plt.grid()
    plt.xlabel("vxB")
    plt.ylabel("vxvxB")
    plt.plot(pos[0], pos[1], "*")
    plt.axis("equal")
    dir = coord.nwu_sph_to_cart(np.array([np.deg2rad(30), np.deg2rad(85), 1.0]))
    trshw = FrameNetFrameShower()
    trshw.init_v_inc(-dir, 0)
    pos_net = trshw.pos_to(pos, "NET")
    plt.figure()
    plt.title("circle in NET frame")
    plt.xlabel("x => North")
    plt.ylabel("y => West")
    plt.plot(pos_net[0], pos_net[1], "*")
    plt.axis("equal")
    plt.grid()


if __name__ == "__main__":
    # check_frame_to_shower()
    # check_frame_shower2net()
    check_dc2(314)
    plt.show()
