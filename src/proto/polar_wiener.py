import pprint
from logging import getLogger

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pyplot as plt

import rshower.manage_log as mlg
from rshower.basis.traces_event import Handling3dTraces
from rshower.num.wiener import WienerDeconvolution
from rshower.io.events.grand_trigged import GrandEventsSelectedFmt01
from rshower.io.leff_fmt import get_leff_default
from rshower.io.rf_fmt import read_TF3_fmt, plot_global_rf_chain_TF3
from rshower.model.ant_resp import DetectorUnitAntenna3Axis

logger = getLogger(__name__)


# define a handler for logger : standard only
mlg.create_output_for_logger("error", log_stdout=True)


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


def deconv_main(pn_fevents, pn_fmodel, idx_evt=3):
    """
    Deal with format and event selection:
        Load data to init Handling3dTraces object
        Load instrument model : antenna
        Load instrument model : RF chain
        Load instrument model : galaxy signale as noise

    :param pn_fevents: Path, name of file events
    :param pn_fmodel:  Path, name of file models
    """
    # Load data to init Handling3dTraces object
    df_events = GrandEventsSelectedFmt01(pn_fevents)
    df_events.plot_stats_events()
    evt = df_events.get_3dtraces(idx_evt, adu2volt=True)
    pars_evt = df_events.get_azi_elev(idx_evt)
    pprint.pprint(pars_evt)
    # evt.set_periodogram(80)
    evt.plot_footprint_val_max()
    # Load instrument model : antenna
    ant_resp = DetectorUnitAntenna3Axis(get_leff_default(pn_fmodel))
    # Load instrument model : RF chain
    rf_fft = read_TF3_fmt(pn_fmodel)
    plot_global_rf_chain_TF3(pn_fmodel)
    # Load instrument model : galaxy signale as noise
    gal_psd = None
    #
    return evt, ant_resp, rf_fft, gal_psd

def deconv_all_du():
    """
    Deconvolution of all du in events
    """
    #
    pass
    # evt, d_simu = fsr.load_asdf(FILE_voc)
    # assert isinstance(evt, Handling3dTraces)
    # pprint.pprint(d_simu)
    # ## pre-compute
    # wiener = WienerDeconvolution(evt.f_samp_mhz * 1e6)
    # # 3)
    # ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files())
    # ## compute relative xmax and direction
    # ant3d.set_pos_source(get_simu_xmax(d_simu))
    # size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
    #     evt.get_size_trace(),
    #     evt.f_samp_mhz,
    #     1.05,
    # )
    # logger.debug(freqs_out_mhz.shape)
    # ant3d.set_freq_out_mhz(freqs_out_mhz)
    # a_pol = np.zeros(evt.get_nb_du(), dtype=np.float32)
    # ## add white noise
    # for idx_du in range(evt.get_nb_du()):
    # # best_sig = sf.irfft(data[5])[: evt.get_size_trace()]
    # # plt.figure()
    # # plt.plot(evt.t_samples[idx_du], best_sig)
    # # plt.grid()
    # evt.network.plot_footprint_1d(a_pol, "fit polar angle", evt, scale="lin", unit="deg")
    # return a_pol


def deconv_polar_wiener_du():
    """
    Deconvolution of traces (ie 2 or 3 traces)of one DU
    """
    pass
    # noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    # evt.traces[idx_du] += noise
    # # evt.plot_trace_idx(idx_du)
    # # 2)
    # ant3d.interp_leff.set_angle_polar(0)
    # ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
    # ## define energy spectrum of signal
    # leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    # wiener.set_rfft_kernel(leff_pol_sn)
    # es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
    # wiener.set_spectrum_sig(es_sig_est)
    # #
    # # minimize
    # data = [1, 2, 3, 4, 5, 6, 7]
    # # fft_volt
    # v_0 = sf.rfft(evt.traces[idx_du][0], n=wiener.sig_size)
    # v_1 = sf.rfft(evt.traces[idx_du][1], n=wiener.sig_size)
    # v_2 = sf.rfft(evt.traces[idx_du][2], n=wiener.sig_size)
    # data[0] = np.array([v_0, v_1, v_2])
    # # spect_volt
    # sp_1 = wiener.get_spectrum_vec(evt.traces[idx_du, 0])
    # sp_2 = wiener.get_spectrum_vec(evt.traces[idx_du, 1])
    # sp_3 = wiener.get_spectrum_vec(evt.traces[idx_du, 2])
    # data[1] = np.array([sp_1, sp_2, sp_3])
    # ## ant3d = data[2]
    # data[2] = ant3d
    # ## wiener = data[3]
    # data[3] = wiener
    # ## sigma = data[4]
    # data[4] = sigma
    # ## coef_func2, weight of second loss function
    # data[6] = coef_func2
    # logger.info(mlg.chrono_start())
    # res = sco.minimize_scalar(
    # loss_func_polar, method="brent", args=data, tol=np.deg2rad(0.5), options={"disp": True}
    # )
    # logger.info(mlg.chrono_string_duration())
    # logger.info(res.message)
    # a_pol[idx_du] = np.rad2deg(res.x) % 180
    # logger.info(a_pol[idx_du])


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    pn_fevents = "/home/jcolley/projet/grand_wk/data/event/gp13_2024_polar/GP13_UD_240616_240708_with_time.npz"
    pn_fmodel = "/home/jcolley/projet/grand_wk/recons/du_model"
    deconv_main(pn_fevents, pn_fmodel)
    # =============================================
    plt.show()
    logger.info(mlg.string_end_script())
