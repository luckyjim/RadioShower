"""
Colley Jean-Marc, CNRS/IN2P3/LPNHE

Coordinate transformation in **same** frame.

see frame.py module to have frame definition and specific convention of axis and angle

https://en.wikipedia.org/wiki/Spherical_coordinate_system#Definition
polar angle is distance zenithal

"""

from logging import getLogger

import numpy as np

G_2PI = 2 * np.pi
logger = getLogger(__name__)

#
# [DU/NET/COR] frame in NWU convention, see definition in frame.py
#


def nwu_cart_to_dir(xyz):
    """Convert cartesian vector xyz to direction in [DU] frame

    :param xyz: cartesian vector
    :type xyz: float (3,n)

    :return: angle direction: azimuth, distance zenithal
    :rtype: float (2,n)
    """
    azi_w = np.arctan2(xyz[1], xyz[0])
    idx = np.argwhere(azi_w < 0)
    azi_w[idx] += G_2PI
    rho = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
    d_zen = np.arctan2(rho, xyz[2])
    return np.array([azi_w, d_zen])


def nwu_cart_to_dir_one(xyz):
    return np.squeeze(nwu_cart_to_dir(xyz[:, None]))


def nwu_cart_to_sph(xyz):
    """Convert cartesian vector xyz to spherical in [DU] frame

    :param xyz: cartesian vector
    :type xyz: float (3,n)

    :return: azimuth, distance zenithal, norm
    :rtype: float (3,n)
    """
    # TODO: rewrite for vector input like (n,3)
    azi_w = np.arctan2(xyz[1], xyz[0])
    idx = np.argwhere(azi_w < 0)
    azi_w[idx] += G_2PI
    rho_2 = xyz[0] ** 2 + xyz[1] ** 2
    rho = np.sqrt(rho_2)
    d_zen = np.arctan2(rho, xyz[2])
    return np.array([azi_w, d_zen, np.sqrt(rho_2 + xyz[2] ** 2)])


def nwu_cart_to_sph_one(xyz):
    return np.squeeze(nwu_cart_to_sph(xyz[:, None]))


def nwu_sph_to_cart(sph):
    """Convert spherical vector xyz to cartesian in [DU] frame

    :param sph: azimuth, distance zenithal, norm
    :type sph: float (3,)

    :return: cartesian vector
    :rtype: float (3,)
    """
    c_a, s_a = np.cos(sph[0]), np.sin(sph[0])
    c_dz, s_dz = np.cos(sph[1]), np.sin(sph[1])
    xyz = np.empty_like(sph)
    xyz[0] = s_dz * c_a
    xyz[1] = s_dz * s_a
    xyz[2] = c_dz
    return sph[2] * xyz


#
# [TAN] see convention of this frame in module frame.py
#


def tan_cart_to_angle_e_theta(xyz):
    return np.arctan2(xyz[1], xyz[0])


def tan_cart_to_polar_angle(xyz):
    return tan_cart_to_angle_e_theta(xyz)
