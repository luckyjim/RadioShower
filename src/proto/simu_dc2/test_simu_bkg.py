"""
Created on 9 oct. 2025

@author: jcolley
"""

from proto.simu_dc2.simu_bkg import *


p_efield = "/home/jcolley/projet/lucky/data/"
pn_efield = p_efield + "efield6_39-24951.asdf"


def test_efield_remove_cherenkov():
    f_ef = f_tr.AsdfReadTraces(pn_efield)
    tref = f_ef.get_event(0)
    assert isinstance(tref, Handling3dTraces)
    tref.plot_footprint_val_max()
    ntref = tref.copy()
    efield_remove_cherenkov(ntref)
    ntref.plot_footprint_val_max()


if __name__ == "__main__":
    test_efield_remove_cherenkov()
    plt.show()
