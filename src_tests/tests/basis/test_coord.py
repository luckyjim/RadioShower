"""
Created on 20 avr. 2023

@author: jcolley
"""


from rshower.basis.coord import *


def test_du_cart_to_sph():
    #
    xyz = np.array([1, 0, 0], dtype=np.float32)
    ret = np.rad2deg(nwu_cart_to_dir_one(xyz))
    assert np.allclose(ret, [0, 90])
    #
    xyz = np.array([0, 0, 1], dtype=np.float32)
    ret = np.rad2deg(nwu_cart_to_dir_one(xyz))
    assert np.allclose(ret, [0, 0])
    #
    xyz = np.array([0, 1, 0], dtype=np.float32)
    ret = np.rad2deg(nwu_cart_to_dir_one(xyz))
    assert np.allclose(ret, [90, 90])
    #
    xyz = np.array([1, 1, 1], dtype=np.float32)
    ret = np.rad2deg(nwu_cart_to_dir_one(xyz))
    assert np.allclose(ret, [45, 54.735610317245346])
