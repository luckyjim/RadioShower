"""

See coord.py module for specific spherical/angle convention associated to each frame


Frame available:
================

    [W84] for WGS84 is the frame of GPS
        * Remark : use to define position of DU, 
                   for shower physic with time measure, position of DU arround 10 cm seems ok
        * Origin : center of earth
        * Cartesian: ECEF
          * X: to equator longitude 0, Y: to equator to East, Z:  North rotation axis
        * Spherical with GRS80 ellipsoide
          * longitude geodetic
          * latitide geodetic
          * altitude:
              * above ellipsoide or geoide EGM96
              

    [XCS] is the frame associated to XCore of air shower used by ZHAireS
        * Origin [W84]: XCore position at Sea level
        * Cartesian: NWU ie tangential to the surface of the earth
          * X: North mag, Y: West mag, Z: normal up to earth 
        * Spherical
          * azi_w (phi_n)[0,360] = angle between X and azi_w(West)=90 degree
          * d_zen (theta_n) = angle from zenith , d_zen(horizon)=90 degree
        * Remark : so in fact it's a familly of frame 
        * Example: ZHaireS simulation     
        

    [NET] is the frame associated to NETwork stations
        * Origin [W84]: TBD, can be center of network or x core
        * Cartesian: NWU ie tangential to the surface of the earth
          * X: North mag, Y: West mag, Z: normal up to earth 
        * Spherical
          * azi_w (phi_n)[0,360] = angle between X and azi_w(West)=90 degree
          * d_zen (theta_n) = angle from zenith , d_zen(horizon)=90 degree
        * Remark : so in fact it's a familly of frame 
        * Example: 
        

    [DU] is the frame associated to one Detector Unit (DU)
        * Origin [W84]/[N]: antenna position given by GPS position
        * Remark : normaly we must indicate the id of the DU, like [DUi] but as we don't have 
                   computation between DU it's not necessary to specify it
        * Cartesian: NWU ie tangential to the surface of the earth
          * X: North mag, Y: West mag, Z: Up
        * Spherical
          * azi_w (phi_du) [0,360] = angle between X and azi_w(West)=90 degree
          * d_zen (theta_du) = angle from zenith , d_zen(horizon)=90 degree


    [TAN] is the frame associated to Tangential ANtenna in direction of E field source (phi_src, theta_src) 
        * Origin [DU]: position associated with unit vector with direction (phi_src, theta_src)
        * Cartesian: 
          * X: e_theta, Y: e_phi, Z: normal up to sphere
        * Spherical
           * only angle between e_theta in trigo orientation, 90 deg for e_phi direction in (e_theta, e_phi) plane
           * is vector is p linear polarization, the angle is polar angle
           
           
    [POL] is the frame associated to linear polarization of E field
         * Origin : center of antenna 
         * Cartesian: only one dimension is used
           * X: p is linear polarization. p is in plane (e_theta,e_phi) of [TAN]
         
    Remark:
       In case of small network (20-30km) [NET]/[XC] and [DU] are equivalent for vector orientation 
       because local normal and magnetic field can be considered as constant on this aera.


    Notation:
       convention xxx_yy variable means position of xxx is in [yy] frame.
       example : efield_tan is E field in tangential frame of antenna
       
    Note:
      all transformation between frame used cartesian coordinate

"""

from logging import getLogger

import numpy as np
from scipy.spatial.transform import Rotation as Rot

#
#
logger = getLogger(__name__)


class FrameAFrameB:
    def __init__(self, offset_ab_a, rot_b2a):
        self.offset_ab_a = offset_ab_a
        self.rot_b2a = rot_b2a
        self.offset_ab_b = np.matmul(self.rot_b2a.T, offset_ab_a)
        self._d_frame = {"fa": "a", "fb": "b"}

    def pos_to(self, pos, id_frame):
        if self._d_frame[id_frame] == "a":
            return self.pos_to_a(pos)
        elif self._d_frame[id_frame] == "b":
            return self.pos_to_b(pos)
        else:
            raise

    def vec_to(self, vec, id_frame):
        if self._d_frame[id_frame] == "a":
            return self.vec_to_a(vec)
        elif self._d_frame[id_frame] == "b":
            return self.vec_to_b(vec)
        else:
            raise

    def pos_to_a(self, pos_b):
        return self.offset_ab_a + np.matmul(self.rot_b2a, pos_b)

    def pos_to_b(self, pos_a):
        return -self.offset_ab_b + np.matmul(self.rot_b2a.T, pos_a)

    def vec_to_a(self, vec_b):
        return np.matmul(self.rot_b2a, vec_b)

    def vec_to_b(self, vec_a):
        return np.matmul(self.rot_b2a.T, vec_a)


class FrameDuFrameTan(FrameAFrameB):
    """Transformation between frame [TAN] and [DU]

    rot_b2a or rot_du2tan is defined by 2 elementaries rotation

     [DU]     [DU]    [I]
    M     = M2    x M1
     [TAN]    [I]     [TAN]

    Where M1 and M2
      * M1 is rotation around z for angle azi_w in positive way
      * M2 is rotation around y for angle d_zen in positive way

    So with euler notation and scipy API M ie rot_b2a) is Rot.from_euler('YZ', [d_zen, azi_w])
    with upper case X,Y,Z and not lower case x,y,z (see documentation )

    ..note:
        [DU]       <->   [TAN]
          N(orth)          e_theta
          W(est)           e_phi
          Up               e_up


    """

    def __init__(self, vec_dir_du):
        """

        :param vec_dir_du: angle azi, dist zen
        :type vec_dir_du:
        """
        offset_ab_a = np.zeros(3, dtype=vec_dir_du.dtype)
        azi_w = vec_dir_du[0]
        d_zen = vec_dir_du[1]
        # Warning : use intrinsec notation upper case X,Y,Z and not lower case x,y, z !!!!
        m1 = Rot.from_euler("Y", d_zen).as_matrix()
        m2 = Rot.from_euler("Z", azi_w).as_matrix()
        rot_b2a = np.matmul(m2, m1)
        # OR
        me = Rot.from_euler("ZY", [azi_w, d_zen]).as_matrix()
        assert np.allclose(rot_b2a, me)
        super().__init__(offset_ab_a, rot_b2a)
        self._d_frame = {"DU": "a", "TAN": "b"}
