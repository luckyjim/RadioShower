"""
Created on 26 avr. 2023

@author: jcolley
"""

from rshower.basis.frame import *
from rshower.basis import coord


def test_du_tan():
    vec_dir_du = np.deg2rad(np.array([0, 90]))
    cf_du_tan = FrameDuFrameTan(vec_dir_du)
    ret = cf_du_tan.vec_to_a(np.array([1, 0, 0]))
    assert np.allclose(ret, [0, 0, -1])
    ret = cf_du_tan.vec_to_a(np.array([0, 1, 0]))
    assert np.allclose(ret, [0, 1, 0])
    ret = cf_du_tan.vec_to_a(np.array([0, 0, 1]))
    assert np.allclose(ret, [1, 0, 0])
    vec_dir_du = np.deg2rad(np.array([45, 10]))
    cf_du_tan = FrameDuFrameTan(vec_dir_du)
    ret = cf_du_tan.vec_to_a(np.array([1, 0, 0]))
    # e_theta dans le quadran positif
    assert ret[0] > 0
    assert ret[1] > 0
    # e_theta pointe vers le bas dans DU
    assert ret[2] < 0
    ret = cf_du_tan.vec_to_a(np.array([0, 1, 0]))
    # e_phi paralle au plan O,x,y
    assert np.allclose(ret[2], 0)
    # direction of pointing in DU must be (0,0,1) in TAN
    cart_dir_du = coord.nwu_sph_to_cart(np.array([vec_dir_du[0], vec_dir_du[1], 1]))
    ret = cf_du_tan.vec_to_b(cart_dir_du)
    assert np.allclose(ret, [0, 0, 1])
