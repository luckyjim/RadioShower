"""
Created on 29 mai 2025

@author: jcolley
"""
from logging import getLogger
import logging
import pprint

import numpy as np
import matplotlib.pyplot as plt
import grand.dataio.root_files as froot
from rshower.io.events.grand_io_fmt import convert_3dtrace_grandlib
from rshower.basis.traces_event import Handling3dTraces, get_psd
from rshower.basis.efield_event import HandlingEfield
from proto.simu_dc2.simu_ash import get_efield_ref_values

logger = getLogger(__name__)


# def get_efield_ref_values(tref):
#     assert isinstance(tref, HandlingEfield)        
#     #tref.remove_trace_low_signal(60)
#     tref_gd = tref.copy()
#     assert isinstance(tref_gd, HandlingEfield)
#     tref_gd.apply_bandpass(40, 210, causal=True)
#     l_ok = tref_gd.remove_trace_low_signal(60)
#     tref.keep_only_trace_with_index(l_ok)
#     t_max, v_max = tref.get_tmax_vmax(hilbert=False, interpol="parab")
#     t_gd_max, v_gd_max = tref_gd.get_tmax_vmax(hilbert=True, interpol="parab")
#     polars, _, _ = tref.get_polar_angle(degree=True)
#     print(tref.ef_pol[0].shape)
#     print(tref.traces.shape)
#     print(tref.f_samp_mhz[0])
#     freq, psd_all = get_psd(tref.ef_pol, 2000.0,200)
#     # pds_all = np.zeros((tref.get_nb_trace(), psd.shape[0]), dtype=psd.dtype)
#     # pds_all[0] = psd
#     # for idx in range(1, tref.get_nb_trace()):
#     #     freq, psd = get_psd(tref.ef_pol[idx], tref.f_samp_mhz,200)
#     #     pds_all[idx] = psd
#     tref_gd.plot_footprint_val_max()
#     return t_max, v_max, t_gd_max, v_gd_max, polars, freq, psd_all


class EfieldRefValues:
    """
    t_max , E_max for full freq band
    t_GD_max , E__GD_max for GRAND freq band
    PSD at ~60 and ~210 MHz
    polar angle
    """

    def __init__(self, pn_efield):
        self.pn_efield = pn_efield
        self.f_ef = froot.get_file_event(pn_efield)

    def get_ref_values(self, i_e):
        self.f_ef.load_event_idx(i_e)
        tref_gd = self.f_ef.get_obj_handling3dtraces()
        d_simu = self.f_ef.get_simu_parameters()
        logger.info(f"Load {i_e} => run/evt : {self.f_ef.run_number}/{self.f_ef.event_number}")
        tref = convert_3dtrace_grandlib(tref_gd, True)
        assert isinstance(tref, HandlingEfield)
        tref.set_xmax(d_simu["FIX_xmax_pos"])
        tref.network.core_pos = d_simu["shower_core_pos"]
        tref.network.name = self.f_ef.tt_run.site
        # remove low level Efield
        tref2 = tref.copy()
        assert isinstance(tref2, HandlingEfield)
        tref2.apply_bandpass(40, 210, causal=False)
        l_ok = tref2.remove_trace_low_signal(75)
        tref.keep_only_trace_with_index(l_ok)
        t_max, v_max, t_gd_max, v_gd_max, polars, freq, pds_all = get_efield_ref_values(tref)
        np.rad2deg(polars, out=polars)
        tref.plot_footprint_val_max()
        tref_pol = tref.copy(0)
        assert isinstance(tref_pol, HandlingEfield)
        tref_pol.name= "Efield polar linear"
        tref_pol.traces[:,0] = tref.ef_pol
        tref_pol.apply_bandpass(40, 210, causal=False)
        tref_pol.plot_footprint_val_max()
        return
        
        
        plt.figure()
        plt.title('Difference t_max with passband')
        plt.hist(t_max-t_gd_max)
        plt.xlabel('ns')
        plt.grid()
        
        plt.figure()
        plt.title('E max versus E max GRAND band')
        plt.plot(v_gd_max,v_max, '.')
        plt.grid()
        
        plt.figure()
        plt.title('Difference t_max versus E max')
        plt.plot(v_max,t_max-t_gd_max, '.')
        plt.xlabel('v_max')
        plt.grid()
        
        plt.figure()
        plt.title('Difference t_max versus E max GRAND band')
        plt.plot(v_gd_max,t_max-t_gd_max, '.')
        plt.xlabel('v_gd_max')
        plt.grid()
       
        plt.figure()
        plt.title('Polar angle distribution')
        plt.hist(polars)
        plt.xlabel('Degree')
        plt.grid()
        
        plt.figure()
        plt.title('PSD Efied')
        i_tr = tref.get_nb_trace()//2+2
        plt.semilogy(freq[2:], pds_all[i_tr][2:])
        plt.grid()


if __name__ == "__main__":
    path_data = "/home/jcolley/projet/grand_wk/data/root/dc2/"
    path_dc2 = path_data + "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/"
    f_adc = "adc_29-24992_L1_0000.root"
    f_ef = "efield_29-24992_L0_0000.root"
    
    eref = EfieldRefValues(path_dc2+f_ef)
    # 400 402 405
    eref.get_ref_values(400)
    plt.show()