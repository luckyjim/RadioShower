"""
Created on 28 nov. 2025

@author: jcolley
"""

from pathlib import Path
from logging import getLogger

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_hist_params_old(pn_model):
    dataset = np.load(pn_model)
    i_ok = np.argwhere(dataset[:, 7] != 0).squeeze()
    dataset = dataset[i_ok]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(r"Histo PSD parameters: $M.e^{-a.f^{\alpha}}+\sigma^2$ ")
    axs[0, 0].hist(
        dataset[:, 3] ** 2, bins=np.logspace(np.log10(1e-8), np.log10(1e-2), 50)
    )
    axs[0, 0].semilogx()
    axs[0, 0].semilogy()
    axs[0, 0].grid()
    axs[0, 0].set_title("Max value: $M$")
    axs[0, 1].hist(
        dataset[:, 4] ** 2, bins=np.logspace(np.log10(1e-8), np.log10(2), 50)
    )
    axs[0, 1].semilogx()
    axs[0, 1].semilogy()
    axs[0, 1].grid()
    axs[0, 1].set_title("coef: $a$")
    axs[1, 0].hist(dataset[:, 5] ** 2, 50)
    axs[1, 0].semilogy()
    axs[1, 0].grid()
    axs[1, 0].set_title(r"alpha: $\alpha$")
    axs[1, 1].hist(
        dataset[:, 6] ** 2, bins=np.logspace(np.log10(1e-13), np.log10(1e-4), 50)
    )
    axs[1, 1].semilogx()
    plt.plot(f_knee, dataset["d_XC"], ".")
    plt.xlabel(r"f_knee")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    axs[1, 1].semilogy()
    axs[1, 1].grid()
    axs[1, 1].set_title(r"variance: $\sigma^2$ ")

    plt.figure()
    plt.plot(dataset[:, 4] ** 2, dataset[:, 5] ** 2, ".")
    plt.ylabel(r"alpha: $\alpha$")
    plt.xlabel(r"coef: $a$")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(f_knee, dataset["d_XC"], ".")
    plt.xlabel(r"f_knee")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    plt.plot(dataset[:, 4] ** 2, dataset[:, 3] ** 2, ".")
    plt.ylabel(r"Max value: $M$")
    plt.xlabel(r"coef: $a$")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset[:, 4] ** 2, dataset[:, 2] ** 2, ".")
    plt.xlabel(r"coef: $a$")
    plt.ylabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset[:, 5] ** 2, dataset[:, 2] ** 2, ".")
    plt.xlabel(r"alpha: $\alpha$")
    plt.ylabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()
    alpha = dataset[:, 5] ** 2
    coef = dataset[:, 4] ** 2
    var = dataset[:, 6] ** 2
    mmax = dataset[:, 3] ** 2

    f_knee = np.power(np.log(mmax / var) / coef, 1 / alpha)
    f_knee = f_knee[~np.isnan(f_knee)]
    print(f_knee.shape)
    plt.figure()
    plt.hist(f_knee, bins=np.logspace(np.log10(30), np.log10(12000), 50))
    plt.semilogx()
    plt.semilogy()
    plt.xlabel(r"$f_{knee}$ [MHz]")
    plt.grid()


def plot_hist_params(pn_model):
    dataset = np.load(pn_model)
    i_ok = np.argwhere(dataset["p_psd"][:, 1] != 0).squeeze()
    dataset = dataset[i_ok]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(r"Histo PSD parameters: $M.e^{-a.f^{\alpha}}+\sigma^2$ ")
    axs[0, 0].hist(
        dataset["p_psd"][:, 0] ** 2,
        bins=np.logspace(np.log10(1e-8), np.log10(1e-2), 50),
    )
    axs[0, 0].semilogx()
    axs[0, 0].semilogy()
    axs[0, 0].grid()
    axs[0, 0].set_title("Max value: $M$")
    axs[0, 1].hist(
        dataset["p_psd"][:, 1] ** 2, bins=np.logspace(np.log10(1e-8), np.log10(2), 50)
    )
    axs[0, 1].semilogx()
    axs[0, 1].semilogy()
    axs[0, 1].grid()
    axs[0, 1].set_title("coef: $a$")
    axs[1, 0].hist(dataset["p_psd"][:, 2] ** 2, 50)
    axs[1, 0].semilogy()
    axs[1, 0].grid()
    axs[1, 0].set_title(r"alpha: $\alpha$")
    axs[1, 1].hist(
        dataset["p_psd"][:, 3] ** 2,
        bins=np.logspace(np.log10(1e-13), np.log10(1e-4), 50),
    )
    axs[1, 1].semilogx()
    axs[1, 1].semilogy()
    axs[1, 1].grid()
    axs[1, 1].set_title(r"variance: $\sigma^2$ ")

    plt.figure()
    plt.plot(dataset["p_psd"][:, 1] ** 2, dataset["p_psd"][:, 2] ** 2, ".")
    plt.ylabel(r"alpha: $\alpha$")
    plt.xlabel(r"coef: $a$")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 1] ** 2, dataset["p_psd"][:, 0] ** 2, ".")
    plt.ylabel(r"Max value: $M$")
    plt.xlabel(r"coef: $a$")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 1] ** 2, dataset["vmax"], ".")
    plt.xlabel(r"coef: $a$")
    plt.ylabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 3] ** 2, dataset["vmax"], ".")
    plt.xlabel(r"coef: $\sigma^2$")
    plt.ylabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 2] ** 2, dataset["vmax"], ".")
    plt.xlabel(r"alpha: $\alpha$")
    plt.ylabel(r"Max voltage")
    # plt.semilogx()
    plt.semilogy()
    plt.grid()
    alpha = dataset["p_psd"][:, 2] ** 2
    coef = dataset["p_psd"][:, 1] ** 2
    var = dataset["p_psd"][:, 3] ** 2
    mmax = dataset["p_psd"][:, 0] ** 2

    plt.figure()
    plt.plot(dataset["p_psd"][:, 2] ** 2, dataset["d_XC"], ".")
    plt.xlabel(r"alpha: $\alpha$")
    plt.ylabel(r"d_XC")
    # plt.semilogx()
    # plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 1] ** 2, dataset["d_XC"], ".")
    plt.xlabel(r"$a$")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    alpha = dataset["p_psd"][:, 2] ** 2
    coef = dataset["p_psd"][:, 1] ** 2
    var = dataset["p_psd"][:, 3] ** 2
    mmax = dataset["p_psd"][:, 0] ** 2

    f_knee = np.power(np.log(mmax / var) / coef, 1 / alpha)
    f_knee = f_knee[~np.isnan(f_knee)]
    print(f_knee.shape)
    plt.figure()
    plt.hist(f_knee, bins=np.logspace(np.log10(30), np.log10(12000), 50))
    plt.semilogx()
    plt.semilogy()
    plt.xlabel(r"$f_{knee}$ [MHz]")
    plt.grid()

    plt.figure()
    plt.title("azimuth histo")
    plt.hist(dataset["azi"])

    i_azi = np.argwhere(dataset["azi"] > 80).squeeze()
    dataset_azi = dataset[i_azi]
    f_knee_azi = f_knee[i_azi]
    i_azi = np.argwhere(dataset_azi["azi"] < 110).squeeze()
    dataset_azi = dataset_azi[i_azi]
    f_knee_azi = f_knee_azi[i_azi]
    plt.figure()
    plt.title("azi 80 a 120 degree")
    plt.plot(dataset_azi["vmax"], f_knee_azi, ".")
    plt.ylabel(r"$f_{knee}$ [MHz]")
    plt.xlabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    i_azi = np.argwhere(dataset["azi"] > 30).squeeze()
    dataset_azi = dataset[i_azi]
    f_knee_azi = f_knee[i_azi]
    i_azi = np.argwhere(dataset_azi["azi"] < 60).squeeze()
    dataset_azi = dataset_azi[i_azi]
    f_knee_azi = f_knee_azi[i_azi]

    fig, ax1 = plt.subplots(1, 1)
    a_values = dataset_azi["d_XC"]
    vmin = np.nanmin(a_values)
    vmax = np.nanmax(a_values)
    norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
    scm = ax1.scatter(
        dataset_azi["vmax"],
        f_knee_azi,
        norm=norm_user,
        s=50,
        c=a_values,
        edgecolors="k",
        cmap="Blues",
    )
    fig.colorbar(scm, label="km")
    ax1.semilogx()
    ax1.semilogy()
    # ax1.xlabel(r"Max voltage")
    # ax1.ylabel(r"$f_{knee}$ [MHz]")
    ax1.grid()

    plt.figure()
    plt.title("azi 170 a 190 degree")
    plt.plot(dataset_azi["vmax"], f_knee_azi, ".")
    plt.ylabel(r"$f_{knee}$ [MHz]")
    plt.xlabel(r"Max voltage")
    plt.semilogx()
    plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(f_knee, dataset["d_XC"], ".")
    plt.xlabel(r"f_knee")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(f_knee, dataset["p_psd"][:, 2] ** 2, ".")
    plt.xlabel(r"f_knee")
    plt.ylabel(r"alpha")
    plt.semilogx()
    # plt.semilogy()
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 0] ** 2, dataset["d_XC"], ".")
    plt.xlabel(r"Max")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    # plt.title("")
    plt.grid()

    plt.figure()
    plt.plot(dataset["p_psd"][:, 0] ** 2, dataset["d_XC"], ".")
    plt.xlabel(r"Max")
    plt.ylabel(r"d_XC")
    plt.semilogx()
    # plt.semilogy()
    # plt.title("")
    plt.grid()

    plt.figure()
    plt.hist(dataset["p_psd"][:, 4], 50)
    plt.semilogy()
    plt.title("Residu Fit")
    plt.grid()


if __name__ == "__main__":
    # plot_hist_params_old("/home/jcolley/projet/lucky/data/model/epsd_model_old.npy")
    plot_hist_params("/home/jcolley/projet/lucky/data/model/epsd_model.npy")
    plt.show()
