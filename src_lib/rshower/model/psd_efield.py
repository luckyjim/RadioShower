"""
Created on 24 nov. 2025

@author: jcolley
"""

from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit

from rshower.basis.traces_event import Handling3dTraces


logger = getLogger(__name__)


def psd_efield(freq, sigma=np.sqrt(1e-11), f_knee=250, alpha=2, vmx=5e-8):
    # psd at 0 is 0
    psd = np.empty_like(freq)
    i_b = 0
    if freq[0] == 0:
        i_b = 1
        psd[0] = 0.0
    a = (2 * np.log(sigma) - np.log(vmx)) / (f_knee**alpha)
    a = np.log(sigma**2 / vmx) / (f_knee**alpha)
    print(a, alpha)
    psd[i_b:] = vmx * np.exp(a * (freq[i_b:]) ** alpha) + sigma**2
    return psd


def modelPSD_4params(freqs, m_p, a_p, alpha_p, sigma):
    # print(m_p**2, a_p**2, alpha_p**2, sigma**2)
    value = (m_p**2) * np.exp(-(a_p**2) * np.power(freqs, alpha_p**2)) + sigma**2
    return value


def cost_func_log_residu(params, freqs, log_psd):
    par = params.valuesdict()
    model = modelPSD_4params(freqs, par["r_max"], par["r_a"], par["r_alpha"], par["sigma"])
    res = log_psd - np.log(model)
    # print(log_psd[:2])    #print(np.log(model[:2]))    #print(np.sum(res**2))
    return res


def init_fit(freq, data):
    r_m1 = np.sqrt(data[:2].mean())
    sig0 = np.sqrt(data[-10:].mean())
    idx_noise = np.argwhere(data < sig0**2)
    if len(idx_noise) == 0:
        f_k = freq[-1]
    else:
        f_k = freq[np.min(idx_noise)]
    # print("F knee:", f_k)
    #r_a = np.log(sig0**2 / r_m1**2) / (f_k)
    r_a = np.log(r_m1**2/sig0**2 ) / (f_k)
    r_a = np.sqrt(r_a)
    return r_m1, r_a, 1, sig0


def test_fit():
    # dataset to fit
    freq = np.linspace(0, 1000, 1000)
    sig = np.sqrt(1e-12)
    f_k = 350
    alpha = 5.5
    m1 = 1e-7
    psd = psd_efield(freq, sig, f_k, alpha, m1)
    data = np.random.normal(psd, psd * 0.3)
    freq = freq[1:]
    data = data[1:]
    psd = psd[1:]
    psd_pars = Parameters()
    if False:
        noise = data - psd
        # define parameters
        sig0 = sig
        r_alpha = np.sqrt(alpha)
        r_m1 = np.sqrt(data[:2].mean())
        # r_m1 = np.sqrt(m1)
        psd_pars.add("r_max", r_m1)
        psd_pars.add("r_alpha", 1)
        r_a = np.log(sig * sig / m1) / (f_k ** (1))
        r_a = np.sqrt(np.abs(r_a))
        # r_a = np.sqrt(0.00028782313662425573)
        # r_a = 1
        psd_pars.add("r_a", r_a)
        sig0 = np.sqrt(data[-10:].mean())
        # sig0 = sigguess
        psd_pars.add("sigma", sig0)
    else:
        r_m1, r_a, r_alpha, sig0 = init_fit(freq, data)
        psd_pars.add("r_max", r_m1)
        psd_pars.add("r_alpha", 1)
        psd_pars.add("r_a", r_a)
        psd_pars.add("sigma", sig0)

    par = psd_pars.valuesdict()
    model0 = modelPSD_4params(freq, par["r_max"], par["r_a"], par["r_alpha"], par["sigma"])
    # r_a
    plt.figure()
    plt.title("Fit PSD")
    plt.plot(freq, data, "+", label="data")
    plt.plot(freq, psd, label="True")
    plt.plot(freq, model0, "b", label="Guess")
    # plt.plot(freq, noise, label="Noise")
    plt.semilogy()
    plt.legend()
    plt.grid()
    #
    minner = Minimizer(cost_func_log_residu, psd_pars, fcn_args=(freq, np.log(data)))
    result = minner.minimize()
    print(r_a)
    # calculate final result
    p_est = result.params.valuesdict()
    final = modelPSD_4params(freq, p_est["r_max"], p_est["r_a"], p_est["r_alpha"], p_est["sigma"])
    # write error report
    report_fit(result)
    # try to plot results
    plt.plot(freq, final, "k", label="Fit")
    plt.legend()
    plt.show()


def test_AirShowerEfieldPSDmodel_fit():
    import rshower.io.events.asdf_traces as f_tr

    f_ef = f_tr.AsdfReadTraces("/home/jcolley/projet/lucky/data/efield_39-24951.asdf")
    f_volt = f_tr.AsdfReadTraces("/home/jcolley/projet/lucky/data/volt-ash_39-24951.asdf")
    # evt = f_ef.get_event(121)
    # evt = f_ef.get_event(400)
    # evt = f_ef.get_event(350)
    # evt = f_ef.get_event(1)
    # evt = f_ef.get_event(60)  # anneau 92 du
    # evt = f_ef.get_event(100) 120/121
    # evt = f_ef.get_event(410)
    # evt = f_ef.get_event(281) 180/183
    # evt = f_ef.get_event(284) # très proche => rebond
    idx = 60
    evt = f_ef.get_event(idx)
    volt = f_volt.get_event(idx)
    volt.plot_footprint_val_max()
    epsd = AirShowerEfieldPSDmodel()
    epsd.set_efield(evt)
    evt.plot_footprint_val_max()
    # epsd.fit_idx(4, True)
    # epsd.fit_idx(5, True)
    # epsd.fit_idx(0, True)
    # epsd.fit_idx(12, True)
    epsd.fit_all()
    epsd.plot_residu()
    i_bad = np.squeeze(np.argwhere(epsd.fit_pars[:, 4] > 1))
    print(i_bad)
    for idx in i_bad:
        epsd.fit_idx(idx, True)


class AirShowerEfieldPSDmodel:

    def __init__(self):
        pass

    def set_efield(self, etraces):
        self.etraces = etraces
        assert isinstance(self.etraces, Handling3dTraces)

    def fit_idx(self, idx=0, f_plot=False):
        # freq , PSD start mode 1
        freq, psd = self.etraces.get_psd_trace_idx(idx)[0]
        freq = freq[2:]
        psd = psd[2:]
        # Guessepsd
        r_m1, r_a, r_alpha, sig0 = init_fit(freq, psd)
        psd_pars = Parameters()
        psd_pars.add("r_max", r_m1)
        psd_pars.add("r_alpha", r_alpha)
        psd_pars.add("r_a", r_a)
        psd_pars.add("sigma", sig0)
        #
        minner = Minimizer(cost_func_log_residu, psd_pars, fcn_args=(freq, np.log(psd)))
        result = minner.minimize()
        # calculate final result
        if result.success:
            par = result.params.valuesdict()
            if f_plot:
                self.etraces.plot_psd_trace_idx(idx)
                psd_fit = modelPSD_4params(
                    freq, par["r_max"], par["r_a"], par["r_alpha"], par["sigma"]
                )
                alpha = par["r_alpha"] ** 2
                m1 = par["r_max"] ** 2
                coef = par["r_a"] ** 2
                plt.semilogy(
                    freq,
                    psd_fit,
                    "y",
                    label="fit: " + f"Max={m1:.1e}, a={coef:.1e}" + r", $\alpha$" + f"={alpha:.2f}",
                )
                plt.legend()
            redchi = result.redchi
            ret = np.array([par["r_max"], par["r_a"], par["r_alpha"], par["sigma"], redchi])
            return ret
        else:
            logger.error(result.message)
            logger.error(result.lmdif_message)
            return np.array([-1.0, 0.0, 0.0, 0.0, 0.0])

    def fit_all(self):
        assert isinstance(self.etraces, Handling3dTraces)
        nb_tr = self.etraces.get_nb_trace()
        self.fit_pars = np.zeros((nb_tr, 5), dtype=np.float32)
        for idx in range(nb_tr):
            self.fit_pars[idx] = self.fit_idx(idx)
        nb_ok = len(np.argwhere(self.fit_pars[:, 0] > 0))
        #print(self.fit_pars)
        #print(f"{nb_ok}/{nb_tr}")

    def plot_residu(self, threshold=0.3):
        # i_ok = np.argwhere(self.fit_pars[4]> threshold)
        # r_ok = self.fit_pars[4]
        plt.figure()
        plt.title("Residu normalisé")
        plt.hist(self.fit_pars[:, 4])
        plt.semilogy()
        plt.grid()


if __name__ == "__main__":
    test_fit()
    #test_AirShowerEfieldPSDmodel_fit()
    plt.show()
