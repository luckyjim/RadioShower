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
from rshower.io.events.grand_trigged import GrandEventsSelectedFmt01
from rshower.io.leff_fmt import get_leff_default
from rshower.io.rf_fmt import read_TF3_fmt, plot_global_rf_chain_TF3
from rshower.model.ant_resp import DetectorUnitAntenna3Axis
from scipy.fft._basic import fft


BANDWIDTH = [52, 185]

# define a handler for logger : standard only
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("info", log_stdout=True)


def weight_efield_estimation(tfd_ef, weight, plot=False):
    """

    :param tfd_ef:
    :type tfd_ef: float (3,n_f)
    :param weight:
    :type weight: float (3,n_f)
    """
    assert tfd_ef.shape == weight.shape
    l2_2 = False
    if l2_2:
        w_2 = weight * weight
        w_ef = np.sum(tfd_ef * w_2, axis=0)
        best_ef = w_ef / np.sum(w_2, axis=0)
    else:
        w_ef = np.sum(tfd_ef * weight, axis=0)
        best_ef = w_ef / np.sum(weight, axis=0)
    if plot:
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
    # r_ew_w = (fft_volt[1] - leff_pol_ew * best_fft_sig) * spect_volt[1]
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
    #df_events.plot_stats_events()
    # evt = df_events.get_3dtraces(idx_evt, adu2volt=True)
    pars_evt = df_events.get_azi_elev(idx_evt)
    # evt.set_periodogram(80)
    # evt.plot_footprint_val_max()
    # Load instrument model : antenna
    ant_resp = DetectorUnitAntenna3Axis(get_leff_default(pn_fmodel))
    # Load instrument model : RF chain
    rf_fft = read_TF3_fmt(pn_fmodel)
    plot_global_rf_chain_TF3(pn_fmodel)
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


def polar_wiener_lost_func_all_du(df_events, ant3d, rf_fft):
    pass


def polar_wiener_lost_func(df_events, ant3d, rf_fft, i_du=0, idx_evt=0):
    """
    Plot residu versus polar angle

    :param df_events:
    :param ant3d:
    :param rf_fft:
    :param wiener:
    """
    assert isinstance(ant3d, DetectorUnitAntenna3Axis)
    l_axis = ["SN", "EW", "UP"]
    l_axis = ["SN", "EW"]
    size_per = 100
    evt = df_events.get_3dtraces(idx_evt, adu2volt=True)
    size_trace = evt.get_size_trace()
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
    freq, tf_ew, tf_ns, tf_z = rf_fft
    tf_elec = np.zeros((3, size_fft), dtype=np.complex64)
    tf_elec[0] = rss.interpol_at_new_x(freq, tf_ns, freqs_out_mhz)
    tf_elec[1] = rss.interpol_at_new_x(freq, tf_ew, freqs_out_mhz)
    tf_elec[2] = rss.interpol_at_new_x(freq, tf_z, freqs_out_mhz)
    # Antenna
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    pars = df_events.get_azi_elev(idx_evt)
    dir_evt_rad = np.array([pars["azi"], pars["d_zen"]])
    print(dir_evt_rad)
    dir_evt_rad = np.deg2rad(dir_evt_rad)
    ant3d.set_dir_source(dir_evt_rad)
    # Wiener
    wiener = WienerDeconvolution(evt.f_samp_mhz[0] * 1e6)
    logger.debug(tf_elec.shape)
    wiener.set_rfft_kernel(tf_elec[0])
    wiener.plot_ker_pow2(", tf_elec[0]")
    bandwidth = [50, 190]
    wiener.set_band(bandwidth)
    # psd galaxy/noise
    psd_noise = np.zeros((3, size_fft), dtype=np.float32)
    noise = Handling3dTraces("Noise")
    noise.init_traces(evt.traces[:, :, 600:], evt.idx2idt, f_samp_mhz=evt.f_samp_mhz)
    noise.set_periodogram(size_per)
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
    cost = np.zeros((360, 2), dtype=np.float32)
    for angle in range(360):
        logger.debug(f"=============> angle {angle}")
        ant3d.interp_leff.set_angle_polar(np.deg2rad(angle))
        for i_a, axis in enumerate(l_axis):
            leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
            wiener.set_rfft_kernel(leff_pol * tf_elec[i_a])
            # wiener.plot_ker_pow2()
            wiener.set_psd_noise(psd_noise[i_a])
            psd_sig = psd_meas[i_a] - psd_noise[i_a]
            psd_sig = np.clip(psd_sig, 1e-9, 1)
            wiener.set_psd_sig(psd_sig)
            # wiener.plot_psd(f", axis {axis}")
            _, fft_est = wiener.deconv_measure(evt.traces[i_du, i_a], psd_sig)
            fft_ef[i_a] = fft_est
            w_ef[i_a] = wiener.snr
        # best Wiener solution
        best_tfd_ef = weight_efield_estimation(fft_ef, w_ef)
        # Cost function
        cost[angle] = 0
        for i_a, axis in enumerate(l_axis):
            leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.leff[i_a])
            res = fft_evt[i_du, i_a] - leff_pol * tf_elec[i_a] * best_tfd_ef
            cost[angle, 0] += norm2_cplx(res[wiener.r_freq])
        # diff = (fft_ef[0]*w_ef[0] - fft_ef[1]*w_ef[1])/(w_ef[0]+w_ef[1])
        diff = fft_ef[0] - fft_ef[1]
        cost[angle, 1] = norm2_cplx(diff[wiener.r_freq])

    plt.figure()
    plt.title(f"cost function, DU index {i_du}")
    plt.semilogy(cost[:, 0], label="$||E||^2$")
    plt.legend()
    plt.grid()
    plt.xlabel("Deg")
    plt.figure()
    plt.title(f"cost function, DU index {i_du}")
    plt.semilogy(cost[:, 1], label="$||\delta E||^2$")
    plt.grid()
    plt.legend()
    plt.xlabel("Deg")
    return cost


def deconv_du_polar_wiener(i_du, evt, ant3d, rf_fft, wiener):
    """
    Deconvolution of traces (ie 2 or 3 traces)of one DU
    """
    loss_func_polar = loss_func_polar_2du
    ant3d.interp_leff.set_angle_polar(0)
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
    polar_wiener_lost_func(df_events, ant_resp, rf_fft,1,14)
    # =============================================
    plt.show()
    logger.info(mlg.string_end_script())
