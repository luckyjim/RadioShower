import pprint
from logging import getLogger

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pyplot as plt

import rshower.manage_log as mlg
from rshower.basis.traces_event import Handling3dTraces
from rshower.num.wiener import WienerDeconvolution
import rshower.num.signal as rss
from rshower.io.events.grand_io_fmt import GrandEventsSelectedFmt01
from rshower.io.leff_fmt import get_leff_default
import rshower.io.rf_fmt as rfchain
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from rshower.basis import coord
from scipy.fft._basic import fft


# define a handler for logger : standard only
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("info", log_stdout=True)


def max_algebra(array_1d):
    idx = np.argmax(np.abs(array_1d))
    return array_1d[idx]


def weight_efield_estimation(tfd_ef, weight, t_plot=""):
    """

    :param tfd_ef:
    :type tfd_ef: float (3,n_f)
    :param weight:
    :type weight: float (3,n_f)
    """
    assert tfd_ef.shape == weight.shape
    l2_2 = True
    if l2_2:
        w_2 = weight * weight
        w_ef = np.sum(tfd_ef * w_2, axis=0)
        best_ef = w_ef / np.sum(w_2, axis=0)
    else:
        w_ef = np.sum(tfd_ef * weight, axis=0)
        best_ef = w_ef / np.sum(weight, axis=0)
    if len(t_plot) > 0:
        if False:
            plt.figure()
            plt.plot(np.abs(tfd_ef[0]), label="0")
            plt.plot(np.abs(tfd_ef[1]), label="1")
            plt.plot(np.abs(tfd_ef[2]), label="2")
            plt.plot(np.abs(best_ef), label="weight sol")
            plt.grid()
            plt.legend()
            plt.figure()
            plt.plot(weight[0], label="0")
            plt.plot(weight[1], label="1")
            plt.plot(weight[2], label="2")
            plt.grid()
            plt.legend()
        plt.figure()
        plt.title(t_plot)
        l_col = ["k", "y", "b"]
        for idx in range(3):
            plt.plot(sf.irfft(tfd_ef[idx]), l_col[idx], label=f"{idx}")
        plt.plot(sf.irfft(best_ef), label=f"Weight solution")
        plt.grid()
        plt.legend()
    return best_ef


def loss_func_polar_3du(angle_pol, data):
    fft_volt = data[0]
    spect_volt = data[1]
    ant3d = data[2]
    wiener = data[3]
    sigma = data[4]
    coef_func2 = data[6]
    #
    ant3d.interp_leff.set_angle_polar(angle_pol)
    # SN
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    wiener.set_rfft_kernel(leff_pol_sn)
    _, fft_sig_sn = wiener.deconv_white_noise_fft_in(fft_volt[0], sigma)
    # wiener.plot_measure_signal(" SN")
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    wiener.set_rfft_kernel(leff_pol_ew)
    _, fft_sig_ew = wiener.deconv_white_noise_fft_in(fft_volt[1], sigma)
    # wiener.plot_measure_signal(f" EW")
    # wiener.plot_spectrum(False)
    # UP
    leff_pol_up = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
    wiener.set_rfft_kernel(leff_pol_up)
    _, fft_sig_up = wiener.deconv_white_noise_fft_in(fft_volt[2], sigma)
    # wiener.plot_measure_signal(" UP")
    # wiener.plot_spectrum()
    # best sol
    a_fft = np.array([fft_sig_sn, fft_sig_ew, fft_sig_up])
    # best_fft_sig = weight_efield_estimation(a_fft, spect_volt)
    best_fft_sig = np.sum(a_fft * spect_volt, axis=0) / np.sum(spect_volt, axis=0)
    data[5] = best_fft_sig
    # residu
    # r_sn_w = (fft_volt[0] - leff_pol_sn * best_fft_sig) * spect_volt[0]
    # r_ew_w = (fft_volt[1]arctan2 - leff_pol_ew * best_fft_sig) * spect_volt[1]
    # r_up_w = (fft_volt[2] - leff_pol_up * best_fft_sig) * spect_volt[2]
    r_sn_w = fft_volt[0] - leff_pol_sn * best_fft_sig
    r_ew_w = fft_volt[1] - leff_pol_ew * best_fft_sig
    r_up_w = fft_volt[2] - leff_pol_up * best_fft_sig
    spect_volt_sum = np.sum(spect_volt, axis=0)
    residu = (
        r_sn_w * spect_volt[0] + r_ew_w * spect_volt[1] + r_up_w * spect_volt[2]
    ) / spect_volt_sum
    loss_func1 = np.sum((residu * np.conj(residu)).real)
    diff = r_sn_w - r_ew_w
    diff1 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[1])
    diff = r_sn_w - r_up_w
    diff2 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[2])
    diff = r_up_w - r_ew_w
    diff3 = ((diff * np.conj(diff)).real) * (spect_volt[2] + spect_volt[1])
    diff = (diff1 + diff2 + diff3) / (2 * spect_volt_sum)
    loss_func2 = np.sum(diff)
    loss_func = loss_func1 + coef_func2 * loss_func2
    logger.debug(f"for {np.rad2deg(angle_pol):5.1f} loss func: {loss_func:.1f}")
    return loss_func


def loss_func_polar_2du(angle_pol, data):
    fft_volt = data[0]
    spect_volt = data[1]
    ant3d = data[2]
    wiener = data[3]
    sigma = data[4]
    coef_func2 = data[6]
    #
    ant3d.interp_leff.set_angle_polar(angle_pol)
    # SN
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    wiener.set_rfft_kernel(leff_pol_sn)
    _, fft_sig_sn = wiener.deconv_white_noise_fft_in(fft_volt[0], sigma)
    # wiener.plot_measure_signal(" SN")
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    wiener.set_rfft_kernel(leff_pol_ew)
    _, fft_sig_ew = wiener.deconv_white_noise_fft_in(fft_volt[1], sigma)
    # wiener.plot_measure_signal(f" EW")
    # wiener.plot_spectrum(False)
    # best sol
    a_fft = np.array([fft_sig_sn, fft_sig_ew])
    # best_fft_sig = weight_efield_estimation(a_fft, spect_volt)
    best_fft_sig = np.sum(a_fft * spect_volt[:2], axis=0) / np.sum(spect_volt[:2], axis=0)
    data[5] = best_fft_sig
    # residu
    # r_sn_w = (fft_volt[0] - leff_pol_sn * best_fft_sig) * spect_volt[0]
    # r_ew_w = (fft_volt[1] - leff_pol_ew * best_fft_sig) * spect_volt[1]
    r_sn_w = fft_volt[0] - leff_pol_sn * best_fft_sig
    r_ew_w = fft_volt[1] - leff_pol_ew * best_fft_sig

    spect_volt_sum = np.sum(spect_volt[:2], axis=0)
    residu = (r_sn_w * spect_volt[0] + r_ew_w * spect_volt[1]) / spect_volt_sum
    loss_residu = np.sum((residu * np.conj(residu)).real)
    diff = r_sn_w - r_ew_w

    loss_diff = np.sum((diff * np.conj(diff)).real)
    loss_func = loss_residu + coef_func2 * loss_diff
    logger.debug(f"for {np.rad2deg(angle_pol):5.1f} loss func: {loss_func:.1f}")
    return loss_func


def deconv_load_data(pn_fevents, pn_fmodel, idx_evt=1):
    """
    Deal with format and event selection:
        Load data to init Handling3dTraces object
        Load instrument model : antenna
        Load instrument model : RF chain

    :param pn_fevents: Path, name of file events
    :param pn_fmodel:  Path, name of file models
    """
    # Load data to init Handling3dTraces object
    df_events = GrandEventsSelectedFmt01(pn_fevents)
    # Load instrument model : antenna
    ant_resp = DetectorUnitAntenna3Axis(get_leff_default(pn_fmodel))
    # Load instrument model : RF chain
    rf_fft = rfchain.read_TF3_fmt(pn_fmodel)
    return df_events, ant_resp, rf_fft


def deconv_all_du(idx_evt, df_events, ant3d, rf_fft):
    """
    Deconvolution of all du in events
    """
    assert isinstance(df_events, GrandEventsSelectedFmt01)
    # Some Inits
    ## pre-compute Wiener object
    evt = df_events.get_3dtraces(idx_evt, adu2volt=True)
    wiener = WienerDeconvolution(evt.f_samp_mhz * 1e6)
    ## Set antenna object
    pars = df_events.get_azi_elev(idx_evt)
    dir_evt_rad = np.array([pars["azi"], pars["d_zen"]])
    dir_evt_rad = np.deg2rad(dir_evt_rad)
    ant3d.set_dir_source(dir_evt_rad)
    size_with_pad, freqs_out_mhz = rss.get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.05,
    )
    logger.debug(freqs_out_mhz.shape)
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## Outputs arrays
    a_pol = np.zeros(evt.get_nb_du(), dtype=np.float32)
    for idx in range(evt.get_nb_du):
        evt = df_events.get_3dtraces(idx, adu2volt=True)
        pol = deconv_du_polar_wiener(idx, evt, ant3d, rf_fft, wiener)
        a_pol[idx_evt] = pol


def norm2_cplx(v_cplx):
    return np.sum((v_cplx * np.conj(v_cplx)).real)


def plot_polar_angle_max(evt, l_cost, idx_cost, type_cost, band=[0, 359], title=""):
    plt.figure()
    plt.title(title)
    max_val = evt.get_max_norm()
    l_min = []
    for idx in range(evt.get_nb_trace()):
        data = l_cost[idx][type_cost][band[0] : band[1], idx_cost]
        # plt.figure()
        # plt.plot(data)
        # plt.yscale("log")
        # return
        min_angle = np.argmin(data) + band[0]
        l_min.append(min_angle)
        plt.plot(min_angle, max_val[idx], "*", markersize=15, label=evt.idx2idt[idx])
    p_med = np.median(l_min)
    plt.axvline(x=p_med, label=f"Median {p_med:.1f}")
    plt.xlabel("Polar angle estimation, deg")
    plt.ylabel(f"Max value of trace in {evt.unit_trace}")
    # plt.xlim([150,190])
    plt.legend()
    plt.grid()


def polar_wiener_lost_func_all_du_gp13(df_events, ant3d, rf_fft, idx_evt=0):
    evt = df_events.get_3dtraces(idx_evt, adu2volt=True)
    pars = df_events.get_azi_elev(idx_evt)
    l_cost = polar_wiener_lost_func_all_du(evt, pars, ant3d, rf_fft)
    #  evt 1
    plot_polar_angle_max(
        evt, l_cost, 0, 0, band=[100, 250], title="Polar angle with residu of SN voltage"
    )
    plot_polar_angle_max(
        evt, l_cost, 1, 1, band=[100, 250], title="Polar angle with $\delta$Efield (EW-UP) "
    )
    # evt 14
    # plot_polar_angle_max(
    #     evt,
    #     l_cost,
    #     3,
    #     1,
    #     title=f"Polar angle with with the 3 $\delta$Efield, weight by SNR\nGP13 evt {idx_evt}",
    # )


def polar_wiener_lost_func_all_du(evt, pars, ant3d, rf_fft):
    evt.get_tmax_vmax()
    evt.plot_footprint_val_max()
    l_cost = []
    for idx in range(evt.get_nb_trace()):
        cost_res, cost_dif = polar_wiener_lost_func(evt, ant3d, rf_fft, pars, idx)
        l_cost.append([cost_res, cost_dif])
    return l_cost

# def polar_wiener_lost_func_ref(evt, ant3d, rf_fft, pars, i_du=0):
#     """
#     Plot residu versus polar angle
#
#     :param df_events:
#     :param ant3d:
#     :param rf_fpolar_wiener_lost_func_all_duft:
#     :param wiener:
#     """
#     f_plot = False
#     assert isinstance(ant3d, DetectorUnitAntenna3Axis)
#     l_axis = ["SN", "EW", "UP"]
#     l_col = ["k", "y", "b", "r"]
#     # l_axis = ["SN", "EW"]
#     size_per = 100
#     size_trace = evt.get_size_trace()
#     evt.set_periodogram(size_per)
#     if f_plot:
#         evt.plot_trace_idx(i_du)
#         evt.plot_psd_trace_idx(i_du)
#         evt.plot_footprint_val_max()
#         evt.plot_footprint_time_max()
#     # Out freq definition
#     size_padd, freqs_out_mhz = rss.get_fastest_size_rfft(evt.get_size_trace(), evt.f_samp_mhz[0])
#     size_fft = freqs_out_mhz.shape[0]
#     logger.debug(f"size_fft: {size_fft}, size_padd: {size_padd}")
#     ant3d.set_freq_out_mhz(freqs_out_mhz)
#     fft_evt = sf.rfft(evt.traces)
#     # RF chain
#     freq, tf_ew, tf_ns, tf_z = rf_fft
#     tf_elec = np.zeros((3, size_fft), dtype=np.complex64)
#     tf_elec[0] = rss.interpol_at_new_x(freq, tf_ns, freqs_out_mhz)
#     tf_elec[1] = rss.interpol_at_new_x(freq, tf_ew, freqs_out_mhz)
#     tf_elec[2] = rss.interpol_at_new_x(freq, tf_z, freqs_out_mhz)
#     # Antenna
#     ant3d.set_freq_out_mhz(freqs_out_mhz)
#     l_azi_offset = [0, 0, 0]
#     if "xmax" in pars.keys():
#         v_dux = pars["xmax"] - evt.network.du_pos[i_du]
#         dir_evt_rad = coord.nwu_cart_to_dir_NOK(v_dux)
#         print(f"Xmax direction : {np.rad2deg(dir_evt_rad)}")
#     else:
#         dir_evt_rad = np.array([pars["azi"], pars["d_zen"]])
#         dir_evt_rad = np.deg2rad(dir_evt_rad)
#     ant3d.set_dir_source(dir_evt_rad)
#
#     # Wiener
#     wiener = WienerDeconvolution(evt.f_samp_mhz[0] * 1e6)
#     logger.debug(tf_elec.shape)
#     wiener.set_rfft_kernel(tf_elec[0])
#     # wiener.plot_ker_pow2(", tf_elec[0]")
#     bandwidth = [70, 190]
#     wiener.set_band(bandwidth)
#     # psd galaxy/noise
#     psd_noise = np.zeros((3, size_fft), dtype=np.float32)
#     noise = Handling3dTraces("Noise")
#     noise.init_traces(evt.traces[:, :, -400:], evt.idx2idt, f_samp_mhz=evt.f_samp_mhz)
#     noise.set_periodogram(size_per)
#     if f_plot:
#         noise.plot_trace_idx(i_du)
#         noise.plot_psd_trace_idx(i_du)
#     freq_psd, psd = noise.get_psd_trace_idx(i_du)
#
#     psd_noise[0] = wiener.get_interpol(freq_psd, psd[0])
#     psd_noise[1] = wiener.get_interpol(freq_psd, psd[1])
#     psd_noise[2] = wiener.get_interpol(freq_psd, psd[2])
#     # psd signal
#     evt.set_periodogram(size_per)
#     psd_meas = np.zeros((3, size_fft), dtype=np.float32)
#     freq_psd, psd = evt.get_psd_trace_idx(i_du)
#     psd_meas[0] = wiener.get_interpol(freq_psd, psd[0])
#     psd_meas[1] = wiener.get_interpol(freq_psd, psd[1])
#     psd_meas[2] = wiener.get_interpol(freq_psd, psd[2])
#     # out
#     fft_ef = np.zeros((3, size_fft), dtype=np.complex64)
#     w_ef = np.zeros((3, size_fft), dtype=np.float32)
#     cost_res = np.zeros((180, 4), dtype=np.float64)
#     cost_dif = np.zeros((180, 4), dtype=np.float64)
#     pic_epol = np.zeros(180, dtype=np.float32)
#     for angle in range(180):
#         logger.debug(f"=============> angle {angle}")
#         t_best_sol = ""
#         if f_plot and (angle == 61 or angle == 241):
#             t_best_sol = f" Angle {angle}"
#         ant3d.interp_leff.set_angle_polar(np.deg2rad(angle))
#         for i_a, axis in enumerate(l_axis):
#             #dir_evt_rad = np.array([(pars["azi"] + l_azi_offset[i_a]) % 360, pars["d_zen"]])
#             #dir_evt_rad = np.deg2rad(dir_evt_rad)
#             ant3d.set_dir_source(dir_evt_rad)
#             leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
#             #print(angle, np.sum(np.abs(leff_pol)))
#             wiener.set_rfft_kernel(leff_pol * tf_elec[i_a])
#             # wiener.plot_ker_pow2()
#             wiener.set_psd_noise(psd_noise[i_a])
#             psd_sig = psd_meas[i_a] - psd_noise[i_a]
#             psd_sig = np.clip(psd_sig, 1e-9, 1)
#             wiener.set_psd_sig(psd_sig)
#             # wiener.plot_psd(f", axis {axis}")
#             sig_est, fft_est = wiener.deconv_measure(evt.traces[i_du, i_a], psd_sig)
#             if f_plot and (angle == 61 or angle == 241):
#                 wiener.plot_measure_est(f", Angle {angle}, {axis}")
#                 wiener.plot_signal_est(f", Angle {angle}, {axis}")
#                 # ant3d.interp_leff.plot_leff_pol()
#                 # wiener.plot_ker_pow2(f", Angle {angle}, {axis}")
#             pic_epol[angle] = max_algebra(sig_est)
#             fft_ef[i_a] = fft_est.copy()
#             w_ef[i_a] = wiener.snr.copy()
#         wbf_ef = w_ef[:, wiener.r_freq]
#         # best Wiener solution
#         best_tfd_ef = weight_efield_estimation(fft_ef, w_ef, t_best_sol)
#         # Cost function
#         wd_scal = np.ones(3, dtype=np.float64)
#         for i_a, axis in enumerate(l_axis):
#             dir_evt_rad = np.array([(pars["azi"] + l_azi_offset[i_a]) % 360, pars["d_zen"]])
#             dir_evt_rad = np.deg2rad(dir_evt_rad)
#             ant3d.set_dir_source(dir_evt_rad)
#             leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
#             res = fft_evt[i_du, i_a] - leff_pol * tf_elec[i_a] * best_tfd_ef
#             # res /= 1 + np.abs(fft_evt[i_du, i_a])
#             cost_res[angle, i_a] = norm2_cplx(res[wiener.r_freq])
#             i_ap = (i_a + 1) % 3
#             diff = fft_ef[i_a] - fft_ef[i_ap]
#             dem = 1 + np.max(np.array([np.abs(fft_ef[i_a]), np.abs(fft_ef[i_ap])]), axis=0)
#             # diff /= dem
#             wd_scal[i_a] = wbf_ef[i_a].sum() ** 2 + wbf_ef[i_ap].sum() ** 2
#             # wd_scal[i_a] = np.sum(1/(wbf_ef[i_a] - wbf_ef[i_ap])**2)
#             # if angle == 62:
#             #print(axis, wd_scal[i_a])
#             cost_dif[angle, i_a] = norm2_cplx(diff[wiener.r_freq])
#         # diff = (fft_ef[0]*w_ef[0] - fft_ef[1]*w_ef[1])/(w_ef[0]+w_ef[1])
#         cost_res[angle, 3] = np.sum(cost_res[angle, :3])
#         cost_dif[angle, 3] = np.sum(cost_dif[angle, :3] * wd_scal) / wd_scal.sum()
#
#     if i_du % 15 == 0:
#         print("Estimator SN-WE: ", np.argmin(cost_dif[:, 0]) % 180)
#         print("Estimator min diff: ", np.argmin(cost_dif[:, :3]) % 180)
#         print("Estimator weight all diff: ", np.argmin(cost_dif[:, 3]) % 180)
#         i_bsnr = np.argmax(wd_scal)
#         print("Estimator diff best SNR: ", np.argmin(cost_dif[:, i_bsnr]) % 180)
#         print("Estimator residu: ", np.argmin(cost_res[:, 3]) % 180)
#
#         plt.figure()
#         plt.title(f"cost function $||residu||^2$, DU index {i_du} ({evt.idx2idt[i_du]})")
#         for i_a, axis in enumerate(l_axis):
#             plt.semilogy(cost_res[:, i_a], l_col[i_a], label=axis)
#         plt.semilogy(cost_res[:, 3], label="Total")
#         plt.legend()
#         plt.grid()
#         plt.xlabel("Deg")
#         plt.figure()
#         plt.title(f"cost function $||\delta E||^2$, DU index {i_du} ({evt.idx2idt[i_du]})")
#         for i_a, axis in enumerate(l_axis):
#             m_lab = axis + "-" + l_axis[(i_a + 1) % (len(l_axis))]
#             plt.semilogy(cost_dif[:, i_a], l_col[i_a], label=m_lab)
#         plt.semilogy(cost_dif[:, 3], label="Total")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Deg")
#         plt.figure()
#         plt.title(f"DU {i_du}")
#         plt.plot(pic_epol)
#         plt.grid()
#     return cost_res, cost_dif


def polar_wiener_lost_func(evt, ant3d, rf_fft, pars, i_du=0):
    """
    Plot residu versus polar angle

    :param df_events:
    :param ant3d:
    :param rf_fpolar_wiener_lost_func_all_duft:
    :param wiener:
    """
    f_plot = False
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    l_axis = ["SN", "EW", "UP"]
    l_col = ["k", "y", "b", "r"]
    size_per = 100
    size_trace = evt.get_size_trace()
    evt.set_periodogram(size_per)
    if f_plot:
        evt.plot_trace_idx(i_du)
        evt.plot_psd_trace_idx(i_du)
        evt.plot_footprint_val_max()
        evt.plot_footprint_time_max()
    # Out freq definition
    size_padd, freqs_out_mhz = rss.get_fastest_size_rfft(evt.get_size_trace(), evt.f_samp_mhz[0])
    size_fft = freqs_out_mhz.shape[0]
    logger.debug(f"size_fft: {size_fft}, size_padd: {size_padd}")
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    fft_evt = sf.rfft(evt.traces)
    # RF chain
    freq, tf_ns, tf_ew, tf_z = rf_fft
    tf_elec = np.zeros((3, size_fft), dtype=np.complex64)
    tf_elec[0] = rss.interpol_at_new_x(freq, tf_ns, freqs_out_mhz)
    tf_elec[1] = rss.interpol_at_new_x(freq, tf_ew, freqs_out_mhz)
    tf_elec[2] = rss.interpol_at_new_x(freq, tf_z, freqs_out_mhz)
    # Antenna
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    if "xmax" in pars.keys():
        v_dux = pars["xmax"] - evt.network.du_pos[i_du]
        #dir_evt_rad = coord.nwu_cart_to_dir_NOK(v_dux)
        dir_evt_rad = np.squeeze(coord.nwu_cart_to_dir(v_dux[:,None]))
        print(f"Xmax direction : {np.rad2deg(dir_evt_rad)}")
    else:
        dir_evt_rad = np.array([pars["azi"], pars["d_zen"]])
        dir_evt_rad = np.deg2rad(dir_evt_rad)
    ant3d.set_dir_source(dir_evt_rad)
    
    # Wiener
    wiener = WienerDeconvolution(evt.f_samp_mhz[0] * 1e6)
    logger.debug(tf_elec.shape)
    wiener.set_rfft_kernel(tf_elec[0])
    # wiener.plot_ker_pow2(", tf_elec[0]")
    bandwidth = [90, 200]
    wiener.set_band(bandwidth)
    # psd galaxy/noise
    psd_noise = np.zeros((3, size_fft), dtype=np.float32)
    noise = Handling3dTraces("Noise")
    noise.init_traces(evt.traces[:, :, -400:], evt.idx2idt, f_samp_mhz=evt.f_samp_mhz)
    noise.set_periodogram(size_per)
    if f_plot:
        noise.plot_trace_idx(i_du)
        noise.plot_psd_trace_idx(i_du)
    freq_psd, psd = noise.get_psd_trace_idx(i_du)

    psd_noise[0] = wiener.get_interpol(freq_psd, psd[0])
    psd_noise[1] = wiener.get_interpol(freq_psd, psd[1])
    psd_noise[2] = wiener.get_interpol(freq_psd, psd[2])
    # psd signal
    evt.set_periodogram(size_per)
    psd_meas = np.zeros((3, size_fft), dtype=np.float32)
    freq_psd, psd = evt.get_psd_trace_idx(i_du)
    psd_meas[0] = wiener.get_interpol(freq_psd, psd[0])
    psd_meas[1] = wiener.get_interpol(freq_psd, psd[1])
    psd_meas[2] = wiener.get_interpol(freq_psd, psd[2])
    # out
    fft_ef = np.zeros((3, size_fft), dtype=np.complex64)
    w_ef = np.zeros((3, size_fft), dtype=np.float32)
    cost_res = np.zeros((360, 4), dtype=np.float64)
    cost_dif = np.zeros((360, 4), dtype=np.float64)
    pic_epol = np.zeros(360, dtype=np.float32)
    for angle in range(360):
        logger.debug(f"=============> angle {angle}")
        t_best_sol = ""
        if f_plot and (angle == 61 or angle == 241):
            t_best_sol = f" Angle {angle}"
        ant3d.interp_leff.set_angle_polar(np.deg2rad(angle))
        for i_a, axis in enumerate(l_axis):
            ant3d.set_dir_source(dir_evt_rad)
            leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
            #print(angle, np.sum(np.abs(leff_pol)))
            wiener.set_rfft_kernel(leff_pol * tf_elec[i_a])
            # wiener.plot_ker_pow2()
            wiener.set_psd_noise(psd_noise[i_a])
            psd_sig = psd_meas[i_a] - psd_noise[i_a]
            psd_sig = np.clip(psd_sig, 1e-9, 1)
            wiener.set_psd_sig(psd_sig)
            # wiener.plot_psd(f", axis {axis}")
            sig_est, fft_est = wiener.deconv_measure(evt.traces[i_du, i_a], psd_sig)
            if f_plot and (angle == 61 or angle == 241):
                wiener.plot_measure_est(f", Angle {angle}, {axis}")
                wiener.plot_signal_est(f", Angle {angle}, {axis}")
                # ant3d.interp_leff.plot_leff_pol()
                # wiener.plot_ker_pow2(f", Angle {angle}, {axis}")
            pic_epol[angle] = max_algebra(sig_est)
            fft_ef[i_a] = fft_est.copy()
            w_ef[i_a] = wiener.snr.copy()
        wbf_ef = w_ef[:, wiener.r_freq]
        # best Wiener solution
        best_tfd_ef = weight_efield_estimation(fft_ef, w_ef, t_best_sol)
        # Cost function
        wd_scal = np.ones(3, dtype=np.float64)
        for i_a, axis in enumerate(l_axis):
            leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
            res = fft_evt[i_du, i_a] - leff_pol * tf_elec[i_a] * best_tfd_ef
            # res /= 1 + np.abs(fft_evt[i_du, i_a])
            cost_res[angle, i_a] = norm2_cplx(res[wiener.r_freq])
            i_ap = (i_a + 1) % 3
            diff = fft_ef[i_a] - fft_ef[i_ap]
            dem = 1 + np.max(np.array([np.abs(fft_ef[i_a]), np.abs(fft_ef[i_ap])]), axis=0)
            # diff /= dem
            wd_scal[i_a] = wbf_ef[i_a].sum() ** 2 + wbf_ef[i_ap].sum() ** 2
            # wd_scal[i_a] = np.sum(1/(wbf_ef[i_a] - wbf_ef[i_ap])**2)
            # if angle == 62:
            #print(axis, wd_scal[i_a])
            cost_dif[angle, i_a] = norm2_cplx(diff[wiener.r_freq])
        # diff = (fft_ef[0]*w_ef[0] - fft_ef[1]*w_ef[1])/(w_ef[0]+w_ef[1])
        cost_res[angle, 3] = np.sum(cost_res[angle, :3])
        cost_dif[angle, 3] = np.sum(cost_dif[angle, :3] * wd_scal) / wd_scal.sum()

    if i_du % 1 == 0:
        print("Estimator SN-WE: ", np.argmin(cost_dif[:, 0]) % 180)
        print("Estimator min diff: ", np.argmin(cost_dif[:, :3]) % 180)
        print("Estimator weight all diff: ", np.argmin(cost_dif[:, 3]) % 180)
        i_bsnr = np.argmax(wd_scal)
        print("Estimator diff best SNR: ", np.argmin(cost_dif[:, i_bsnr]) % 180)
        print("Estimator residu: ", np.argmin(cost_res[:, 3]) % 180)

        plt.figure()
        plt.title(f"cost function $||residu||^2$, DU index {i_du} ({evt.idx2idt[i_du]})")
        for i_a, axis in enumerate(l_axis):
            plt.semilogy(cost_res[:, i_a], l_col[i_a], label=axis)
        plt.semilogy(cost_res[:, 3], label="Total")
        plt.legend()
        plt.grid()
        plt.xlabel("Deg")
        plt.figure()
        plt.title(f"cost function $||\delta E||^2$, DU index {i_du} ({evt.idx2idt[i_du]})")
        for i_a, axis in enumerate(l_axis):
            m_lab = axis + "-" + l_axis[(i_a + 1) % (len(l_axis))]
            plt.semilogy(cost_dif[:, i_a], l_col[i_a], label=m_lab)
        plt.semilogy(cost_dif[:, 3], label="Total")
        plt.grid()
        plt.legend()
        plt.xlabel("Deg")
        plt.figure()
        plt.title(f"DU {i_du}")
        plt.plot(pic_epol)
        plt.grid()
    return cost_res, cost_dif


def deconv_du_polar_wiener(i_du, evt, ant3d, rf_fft, wiener):
    """
    Deconvolution of traces (ie 2 or 3 traces)of one DU
    """
    loss_func_polar = loss_func_polar_2du
    ant3d.interpget_polar_angle_efield_leff.set_angle_polar(0)
    # ## define energy spectrum of signal
    # leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    # wiener.set_rfft_kernel(leff_pol_sn)
    # es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
    # wiener.set_spectrum_sig(es_sig_est)
    # #
    # # minimize
    data = [1, 2, 3, 4, 5, 6, 7]
    # fft_volt
    v_0 = sf.rfft(evt.traces[i_du][0], n=wiener.sig_size)
    v_1 = sf.rfft(evt.traces[i_du][1], n=wiener.sig_size)
    v_2 = sf.rfft(evt.traces[i_du][2], n=wiener.sig_size)
    data[0] = np.array([v_0, v_1, v_2])
    # spect_volt
    sp_1 = wiener.get_spectrum_vec(evt.traces[i_du, 0])
    sp_2 = wiener.get_spectrum_vec(evt.traces[i_du, 1])
    sp_3 = wiener.get_spectrum_vec(evt.traces[i_du, 2])
    data[1] = np.array([sp_1, sp_2, sp_3])
    ## ant3d = data[2]
    data[2] = ant3d
    ## wiener = data[3]
    data[3] = wiener
    ## sigma = data[4]
    data[4] = sigma

    logger.info(mlg.chrono_start())
    res = sco.minimize_scalar(
        loss_func_polar, method="brent", args=data, tol=np.deg2rad(0.5), options={"disp": True}
    )
    logger.info(mlg.chrono_string_duration())
    logger.info(res.message)
    pol = np.rad2deg(res.x) % 180
    logger.info(pol)
    return pol


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    logger.debug("++++++++++++++++++++++++++++++++++++++++++++")
    # =============================================
    pn_fevents = "/home/jcolley/projet/grand_wk/data/event/gp13_2024_polar/GP13_UD_240616_240708_with_time.npz"
    pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"
    df_events, ant_resp, rf_fft = deconv_load_data(pn_fevents, pn_fmodel)
    # plt.show()
    polar_wiener_lost_func_all_du_gp13(df_events, ant_resp, rf_fft, 1)
    # =============================================
    plt.show()
    logger.info(mlg.string_end_script())
