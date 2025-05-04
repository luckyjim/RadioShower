"""

"""
import numpy as np
import matplotlib.pyplot as plt

from rshower.io.gal_psd_fmt import read_grand_galaxy_vout2
from rshower.model.galaxy import GalaxyModelVolt
import rshower.num.signal as rs_s
import rshower.manage_log as mlg

G_pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

def check_GalaxyModelVolt_ampl():
    idx2freq, gal, idx2sideral_h = read_grand_galaxy_vout2(G_pn_fmodel)
    gal = GalaxyModelVolt(idx2freq, gal, idx2sideral_h, False)
    gal.plot_gal_psd(18)
    gal.plot_gal_psd(12)
    gal.plot_gal_psd(6)
    

def check_GalaxyModelVolt_signal():
    idx2freq, gal, idx2sideral_h = read_grand_galaxy_vout2(G_pn_fmodel)
    gal = GalaxyModelVolt(idx2freq, gal, idx2sideral_h, False)
    size_with_pad, freqs_out_mhz = rs_s.get_fastest_size_rfft(1024, 500)
    gal.get_volt_all_du(0, 1, freqs_out_mhz, 2)
    gal.plot_gal_realisation()


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    check_GalaxyModelVolt_ampl()
    #check_GalaxyModelVolt_signal()
    plt.show()
