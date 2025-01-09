"""
Colley Jean-Marc, CNRS/IN2P3/LPNHE
"""

import matplotlib.pyplot as plt

import rshower.basis.coord as coord
from rshower.basis.frame import *


def check_frame_shower():
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
    dir = coord.du_sph_to_cart(np.array([np.deg2rad(90), np.deg2rad(zenith), 1.0]))
    trshw = FrameNetFrameShower(-dir, 0)
    pos_sh = trshw.pos_to(pos, "SHW")
    print(pos_sh.shape)
    plt.figure()
    plt.title("ellipse in shower frame")
    plt.xlabel("vxB")
    plt.ylabel("vx(vxB)")
    plt.plot(pos_sh[0], pos_sh[1], "*")
    plt.axis("equal")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    check_frame_shower()
    plt.show()
