"""
Created on 16 oct. 2025

@author: jcolley
"""

from proto.trigger_polar.model_dirvolt import *


def test_process_evt():
    evt = Handling3dTraces()
    traces = np.zeros((2, 3, 8), dtype=np.float64)
    # x*x + constante
    # 30, 30, 22 => max entre 30 et 30
    # -10, -11, -10 => min -11 sommet pas d'interpolation
    traces[0, 0] = np.array([1, 30, 30.00000001, 22, -1, -10, -11, -10])
    # max 31, min -11
    traces[0, 1] = 2 * traces[0, 0]
    # max 62, min -22
    traces[0, 2] = -traces[0, 0]
    # max 11, min -31
    traces[1] = -traces[0]
    evt.init_traces(traces)
    mdv = ModelDirectionVoltage("")
    res = mdv.process_evt(evt)
    print(res[:, :6] * res[0, -1])
    res_true = np.array([[31, 62, 11, -11, -22, -31], [11, 22, 31, -31, -62, -11]])
    print(res_true)
    assert np.allclose(res[:, :6] * res[0, -1], res_true)
    print(res)


if __name__ == "__main__":
    test_process_evt()
