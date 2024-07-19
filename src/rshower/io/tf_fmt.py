'''

'''

import os.path 
import numpy as np 

def read_one_column(pn_file, f_0_mhz=10, step_mhz=1):
    tfc = np.loadtxt(pn_file)
    assert tfc.ndim == 1
    freq = f_0_mhz + np.arange(tfc.shape[0])*step_mhz
    return freq, tfc

def read_tf_total_fmt(n_path, n_ew, n_ns, n_z):
    pn_file = os.path.join(n_path,n_ew)
    _, tf_ew = read_one_column(pn_file)
    pn_file = os.path.join(n_path,n_ns)
    _, tf_ns = read_one_column(pn_file)
    assert tf_ew.shape[0] == tf_ns.shape[0]
    pn_file = os.path.join(n_path,n_z)
    freq, tf_z = read_one_column(pn_file)
    assert tf_z.shape[0] == tf_ns.shape[0]
    return freq, tf_ew, tf_ns, tf_z
    