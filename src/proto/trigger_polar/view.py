"""
Created on 17 oct. 2025

@author: jcolley
"""

import pathlib
import re
import matplotlib.pyplot as plt

import numpy as np
import healpy as hp

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

np.random.seed(0)

ds_dir = "/home/jcolley/projet/lucky/data/v2/"


def plot_hit_test(adir, to_rad=True):
    if to_rad:
        rad_dir = np.deg2rad(adir)
    else:
        rad_dir = adir
    if rad_dir.shape[1] == 2:
        rad_dir = rad_dir.T
    a_pix = hp.ang2pix(NSIDE, rad_dir[1], rad_dir[0])
    p_lg60 = hp.ang2pix(NSIDE, np.deg2rad(30), np.deg2rad(60))
    hsph = np.zeros(NPIX)
    for pix in a_pix:
        hsph[pix] += 1
    hsph[p_lg60] = 3000
    print(hsph.sum())
    idx_0 = np.argwhere(hsph == 0.0)
    hsph[idx_0] = hp.UNSEEN
    hp.orthview(
        hsph,
        rot=(0, 0, 0),
        half_sky=True,
        title="Carte de hit \nNord",
        flip="geo",
        cbar=True,
        # norm='log'
    )
    hp.orthview(
        hsph,
        rot=(0, 0, 0),
        title="     latra=[0,90]Carte de hit",
        flip="geo",
        cbar=True,
        # norm='log'
    )

    hp.graticule()
    hp.projview(
        hsph,
        rot=(180, 0, 0),
        graticule=True,
        graticule_labels=True,
        unit="hit",
        cb_orientation="horizontal",
        projection_type="polar",
        title="Polar projection",
        phi_convention="clockwise",
    )
    hp.cartview(hsph, notext=False)
    hp.graticule()

    masked_m = np.ma.masked_values(hsph, hp.UNSEEN)

    hp.projview(
        masked_m,
        graticule=True,
        graticule_labels=True,
        unit="cbar label",
        xlabel="longitude",
        ylabel="latitude",
        cb_orientation="vertical",
        projection_type="mollweide",
    )


def plot_std_ra(a_ampldir, a_pix):
    print(a_ampldir.shape, a_pix.shape)
    a_std = np.zeros((NPIX, 6), dtype=np.float32)
    a_mean = np.zeros((NPIX, 6), dtype=np.float32)
    a_pix2 = a_pix.copy()
    a_ampl = a_ampldir[:,1:7].copy()
    cpt = 0
    while True:
        #print(a_ampl.shape)
        idx_p = np.argwhere(a_pix2 == a_pix2[0])
        cpt += len(idx_p)
        #print(len(idx_p))
        a_ampl_p = a_ampl[idx_p]
        a_std[a_pix2[0]] = np.std(a_ampl_p, axis=0)
        a_mean[a_pix2[0]] = np.mean(a_ampl_p, axis=0)
        a_pix2 = np.delete(a_pix2, idx_p, 0)
        a_ampl = np.delete(a_ampl, idx_p, 0)
        if a_pix2.shape[0] == 0:
            break
    print(cpt)
    for aaxis in range(6):
        idx_0 = np.argwhere(a_std[:,aaxis] == 0.0)
        a_std[idx_0, aaxis] = hp.UNSEEN
        masked_m = np.ma.masked_values(a_std[:,aaxis] , hp.UNSEEN)
        hp.projview(
            masked_m,
            title=f"direction Air Shower at {a_pix.shape[0]} DUs",
            flip="geo",
            phi_convention="clockwise",
            graticule=True,
            graticule_labels=True,
            unit=f"standard error amplitude {aaxis}",
            xlabel="longitude",
            ylabel="elevation",
            cb_orientation="horizontal",
            projection_type="mollweide",
        )
    for aaxis in range(6):
        idx_0 = np.argwhere(a_mean[:,aaxis] == 0.0)
        a_mean[idx_0, aaxis] = hp.UNSEEN
        masked_m = np.ma.masked_values(a_mean[:,aaxis] , hp.UNSEEN)
        hp.projview(
            masked_m,
            title=f"direction Air Shower at {a_pix.shape[0]} DUs",
            flip="geo",
            phi_convention="clockwise",
            graticule=True,
            graticule_labels=True,
            unit=f"mean amplitude {aaxis}",
            xlabel="longitude",
            ylabel="elevation",
            cb_orientation="horizontal",
            projection_type="polar", 
        )
    


def plot_hit(adir, to_rad=True):
    if to_rad:
        rad_dir = np.deg2rad(adir)
    else:
        rad_dir = adir
    if rad_dir.shape[1] == 2:
        rad_dir = rad_dir.T
    a_pix = hp.ang2pix(NSIDE, rad_dir[1], rad_dir[0])
    p_lg60 = hp.ang2pix(NSIDE, np.deg2rad(60), np.deg2rad(60))
    hsph = np.zeros(NPIX)
    for pix in a_pix:
        hsph[pix] += 1
    # hsph[p_lg60] = 3000
    print(hsph.sum())
    idx_0 = np.argwhere(hsph == 0.0)
    hsph[idx_0] = hp.UNSEEN
    masked_m = np.ma.masked_values(hsph, hp.UNSEEN)
    hp.projview(
        masked_m,
        title=f"direction Air Shower at {a_pix.shape[0]} DUs",
        flip="geo",
        phi_convention="clockwise",
        graticule=True,
        graticule_labels=True,
        unit="hit direction Air Shower",
        xlabel="azimuth",
        ylabel="elevation",
        cb_orientation="horizontal",
        projection_type="mollweide",
    )


def check_plot():
    nb_pts = 10000
    adir = np.empty((nb_pts, 2), dtype=np.float32)
    adir[:, 0] = np.random.uniform(0, 360, nb_pts)
    adir[:, 1] = np.random.normal(45, 10, nb_pts)
    plot_hit(adir, True)


def healpy_test():
    NSIDE = 16
    NPIX = hp.nside2npix(NSIDE)
    map = np.zeros(NPIX)
    nord = hp.ang2pix(NSIDE, np.deg2rad(80), 0)
    map[nord] = 30
    weast = hp.ang2pix(NSIDE, np.deg2rad(80), np.deg2rad(90))
    map[weast] = 50

    zenith = hp.ang2pix(NSIDE, 0, 0.1)
    map[zenith] = 20

    hp.orthview(
        map,
        rot=(180, 90, 0),
        half_sky=True,
        title="Carte de hit \nNord",
        flip="geo",
        cbar=True,
    )
    hp.graticule()


def plot_hit_dataset(pn_ds):
    m_ds = np.load(pn_ds)
    print(m_ds.shape)
    plot_hit(m_ds[:, 8:], True)


def concatenate_dataset():
    pattern = re.compile(r"^dirvolt-ash")
    rep = pathlib.Path(ds_dir)

    v_dir = None
    cpt_file = 0
    for m_f in rep.iterdir():
        print(m_f)
        if m_f.is_file() and pattern.search(m_f.name):
            f_ds = str(m_f.absolute())
            m_ds = np.load(f_ds)
            if v_dir is None:
                v_dir = m_ds
            else:
                v_dir = np.vstack((v_dir, m_ds))
            cpt_file += 1
            print(v_dir.shape)
            # if cpt_file == 9:
            #     break
    rad_dir = np.deg2rad(v_dir[:, 8:]).T
    a_pix = hp.ang2pix(NSIDE, rad_dir[1], rad_dir[0])

    plot_hit(v_dir[:, 8:], True)
    #plot_std_ra(v_dir, a_pix)


if __name__ == "__main__":
    # plot_hit_dataset(ds_dir+"dirvolt-ash_0-24984.npy")
    # healpy_test()
    # check_plot()
    concatenate_dataset()
    #plot_hit()
    plt.show()
