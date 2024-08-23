'''
Created on 23 août 2024

@author: jcolley
'''

import numpy as np 
import matplotlib.pyplot as plt

from rshower.io.leff_fmt import get_leff_default
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
import rshower.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

G_pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"

def check_AntennaLeffStorage():
    o_leff = get_leff_default(G_pn_fmodel)
    o_leff["sn"].plot_leff(89,80)
    o_leff["ew"].plot_leff(0,80)

def check_DetectorUnitAntenna3Axis_leff():
    ant_resp = DetectorUnitAntenna3Axis(get_leff_default(G_pn_fmodel))
    dir_du_deg = np.array([1.001, 82.001])
    dir_du = np.deg2rad(dir_du_deg)
    ant_resp.sn_leff.plot_leff(dir_du_deg[0], dir_du_deg[1])
    ant_resp.ew_leff.plot_leff(dir_du_deg[0], dir_du_deg[1])
    ant_resp.set_freq_out_mhz(ant_resp.sn_leff.freq_mhz[::2])
    ant_resp.set_dir_source(dir_du)
    ant_resp.interp_leff.get_fft_leff_tan(ant_resp.sn_leff)
    ant_resp.interp_leff.plot_leff_tan()
    ant_resp.interp_leff.get_fft_leff_tan(ant_resp.ew_leff)
    ant_resp.interp_leff.plot_leff_tan()
    
if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #check_AntennaLeffStorage()
    check_DetectorUnitAntenna3Axis_leff()
    plt.show()