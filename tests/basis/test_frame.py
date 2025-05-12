"""
Created on 26 avr. 2023

@author: jcolley
"""

from rshower.basis.frame import *
from rshower.basis import coord


def test_du_tan():
    """
    Test the FrameDuFrameTan class for coordinate transformations between DU (Down-Up)
    and TAN (Tangent) frames.

    This test verifies the following:
    1. Transformation of unit vectors from DU to TAN coordinates for specific
       orientations of the DU frame.
    2. Correct behavior of the transformation for edge cases and arbitrary angles.

    Test Cases:
    - For vec_dir_du = [0, 90] degrees:
        - Transform [1, 0, 0] from DU to TAN and verify it equals [0, 0, -1].
        - Transform [0, 1, 0] from DU to TAN and verify it equals [0, 1, 0].
        - Transform [0, 0, 1] from DU to TAN and verify it equals [1, 0, 0].

    - For vec_dir_du = [45, 10] degrees:
        - Transform [1, 0, 0] from DU to TAN and verify:
            - The first two components are positive.
            - The third component is negative.
        - Transform [0, 1, 0] from DU to TAN and verify:
            - The third component is approximately zero.
        - Transform the DU pointing direction (converted to Cartesian) to TAN
          and verify it equals [0, 0, 1].

    Notes:
    - The test uses spherical to Cartesian conversion for verifying the pointing
      direction transformation.
    - The np.allclose function is used for approximate equality checks.
    """
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


def test_FrameNetFrameShower_01():
    """
        Network : origine 0 (N, W, U)
     
        X*       *U 
                 |  *N
                 | /
        C*W------*O
        
        C core (0,1,0) = west unit
        X Xmax (0,1,1) 
        
        Shower : origine X (vxB, vxvxB, v)          
        
        X*-------*vxB
       / |
 vxvxB*  | 
         *v
        
        O (1,0,1)  
        C (0,0,1)     
    """
    tns = FrameNetFrameShower()
    inc_b = 0
    XC = np.array([0, 0, -1])
    tns.init_v_inc(XC, inc_b, np.array([0, 1, 1]))
    assert np.allclose(tns.rot_b2a, np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    pos_O_shw = tns.pos_to(np.array([0, 0, 0]), "SHW").squeeze()
    assert np.allclose(pos_O_shw, np.array([1, 0, 1]))
    pos_C_shw = tns.pos_to(np.array([0, 1, 0]), "SHW").squeeze()
    assert np.allclose(pos_C_shw, np.array([0, 0, 1]))
    vec_N_shw = tns.vec_to(np.array([1, 0, 0]), "SHW").squeeze()
    assert np.allclose(vec_N_shw, np.array([0, -1, 0]))
    pos_N_shw = tns.pos_to(np.array([1, 0, 0]), "SHW").squeeze()
    assert np.allclose(pos_N_shw, np.array([1, -1, 1]))
    sph_O_shw = coord.nwu_cart_to_sph_one(pos_O_shw)
    assert np.allclose(np.rad2deg(sph_O_shw[:2]),np.array([0, 45] ))
    sph_C_shw = coord.nwu_cart_to_sph_one(pos_C_shw)
    assert np.allclose(np.rad2deg(sph_C_shw[:2]),np.array([0, 0] ))