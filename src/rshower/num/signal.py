"""

"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, butter, lfilter, freqz, filtfilt
import scipy.fft as sf
from scipy import interpolate
import matplotlib.pyplot as plt

logger = getLogger(__name__)


def find_max_with_parabola_interp_3pt(x_trace, y_trace, idx_max):
    """Parabolic interpolation of the maximum with 3 points


    trace : all values >= 0

    :param x_trace:
    :param y_trace:
    algo Mode pic, input 3 values and the middle one is max:
        parabola : ax^2 + bx + c
        offset of (x0, y0)
        solve coef a, b , interpolation of the maximum is
          x_m = x0 - b/2a
          y_m = y0 - b^2/4a
    :param idx_max: index of sample max, idx_max < nb_sample
    :type idx_max: int
    :return: x_max, y_max
    """
    if (idx_max >= len(x_trace) - 1) or idx_max == 0:
        return x_trace[idx_max], y_trace[idx_max]
    logger.debug(f"Parabola interp: mode pic {idx_max} {len(x_trace)}")
    # remove offset (x0, v0)
    y_pic = y_trace[idx_max : idx_max + 2] - y_trace[idx_max - 1]
    x_pic = x_trace[idx_max : idx_max + 2] - x_trace[idx_max - 1]
    logger.debug(x_trace[idx_max : idx_max + 2])
    logger.debug(y_trace[idx_max : idx_max + 2])
    # solve coef a, b
    r_pic = y_pic / x_pic
    c_a = (r_pic[1] - r_pic[0]) / (x_pic[1] - x_pic[0])
    c_b = r_pic[0] - c_a * x_pic[0]
    # interpolation of the maximum is
    x_m = -c_b / (2 * c_a)
    x_max = x_trace[idx_max - 1] + x_m
    y_max = y_trace[idx_max - 1] + x_m * c_b / 2
    return x_max, y_max


def find_max_with_parabola_interp(x_trace, y_trace, idx_max, factor_hill=0.8):
    """Parabolic interpolation of the maximum with more than 3 points

    trace : all values >= 0

    algo:
      1. find begin idx, ie trace[--idx_max] > v_max*factor_hill
      2. find end idx, ie trace[idx_max++] > v_max*factor_hill
      3. if nb idx <= 2 : mode pic else mode hill
      4. Mode pic : 3 values and the middle one is max
         4.1 offset of (x0, v0)
         4.2 solve coef a, b => x_m = offset - b/2a ; v_m=offset - b^2/4a
      5. Mode hill:
         5.0 offset of (x, y) of first sample
         5.1 solve overdetermined linear system with a, b, c
         5.2 x_m =offset - b/2a ; v_m=offset - b^2/4a + c

    :param trace:
    :type trace:
    :param idx_max:
    :type idx_max:
    :param factor_hill:
    :type factor_hill:
    """
    y_lim = (y_trace[idx_max - 1 : idx_max + 2].sum() / 3) * factor_hill
    logger.debug(f"y_lim={y_lim}")
    # 1
    b_idx = idx_max - 1
    out_lim = 6
    nb_out = 0
    last_idx = b_idx
    while b_idx > 0 and nb_out < out_lim:
        if y_trace[b_idx] < y_lim:
            nb_out += 1
        else:
            nb_out = 0
            last_idx = b_idx
        b_idx -= 1
    b_idx = last_idx
    # 2
    nb_sple = y_trace.shape[0]
    e_idx = idx_max + 1
    nb_out = 0
    last_idx = e_idx
    while e_idx < nb_sple and nb_out < out_lim:
        if y_trace[e_idx] < y_lim:
            nb_out += 1
        else:
            nb_out = 0
            last_idx = e_idx
        e_idx += 1
    e_idx = last_idx
    logger.debug(f"border around idx max {idx_max} is {b_idx}, {e_idx}")
    logger.debug(f"{x_trace[b_idx]}\t{x_trace[e_idx]}")
    if (e_idx - b_idx) <= 2:
        return find_max_with_parabola_interp_3pt(x_trace, y_trace, idx_max)
    logger.debug(f"Parabola interp: mode hill")
    # mode hill
    y_hill = y_trace[b_idx : e_idx + 1] - y_trace[b_idx]
    x_hill = x_trace[b_idx : e_idx + 1] - x_trace[b_idx]
    mat = np.empty((x_hill.shape[0], 3), dtype=np.float32)
    mat[:, 2] = 1
    mat[:, 1] = x_hill
    mat[:, 0] = x_hill * x_hill
    sol = np.linalg.lstsq(mat, y_hill, rcond=None)[0]
    if -1e-5 < sol[0] and sol[0] < 1e-5:
        # very flat case
        return x_trace[idx_max], y_trace[idx_max]
    x_m = -sol[1] / (2 * sol[0])
    x_max = x_trace[b_idx] + x_m
    y_max = y_trace[b_idx] + x_m * sol[1] / 2 + sol[2]
    return x_max, y_max


def filter_butter_band(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    order = 9
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    filtered = filtfilt(coeff_b, coeff_a, t_series)
    return filtered.real


def filter_butter_band_fft(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly window with fft method, seems equivalent to
        filtered = filtfilt(coeff_b, coeff_a, t_series, axis=0)

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    order = 4
    size_sig = t_series.shape[2]
    size_fft = 2 * size_sig
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    w_butter, h_butter = freqz(coeff_b, coeff_a, fs=f_hz, worN=size_fft)
    if False:
        plt.figure()
        plt.plot(w_butter * 1e-6, np.abs(h_butter), ".")
        plt.grid()
    abs_h = np.abs(h_butter)
    f_fft = sf.fft(t_series, n=size_fft)
    f_fft = f_fft * abs_h
    filtered = sf.ifft(f_fft)[:, :, :size_sig]
    return filtered.real


def filter_butter_band_fft2(t_series, fr_min, fr_max, f_sample, mhz=True):
    """
    band filter with butterfly window with fft method, seems equivalent to
        filtered = filtfilt(coeff_b, coeff_a, t_series, axis=0)

    :return: filtered trace in time domain
    """
    print("t_series.shape: ", t_series.shape)
    if mhz:
        low = fr_min * 1e6
        high = fr_max * 1e6
        f_hz = f_sample * 1e6
    else:
        low = fr_min
        high = fr_max
        f_hz = f_sample
    print(f_hz, low, high)
    order = 6
    size_sig = t_series.shape[0]
    size_fft = 2 * size_sig
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    print(t_series[0])
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=size_fft)
    if False:
        plt.figure()
        print(w.shape)
        plt.plot(w * 1e-6, abs(h), ".")
        plt.grid()
    abs_h = abs(h)
    # abs_h /= abs_h.sum()
    # filtered = lfilter(coeff_b, coeff_a, t_series, axis=0)
    # filtered = filtfilt(coeff_b, coeff_a, t_series, axis=0)
    print(abs_h.shape)
    f_fft = sf.fft(t_series, size_fft)
    print("before ", f_fft.shape)
    f_fft = f_fft * abs_h
    print("after ", f_fft.shape)
    print(t_series.shape, f_fft.shape)
    filtered = sf.ifft(f_fft)[:size_sig]
    print("filtered")
    print(filtered.shape)
    return filtered.real


def filter_butter_band_lfilter(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly window

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 9
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    filtered = lfilter(coeff_b, coeff_a, t_series)
    return filtered.real


def filter_butter_band_causal(t_series, fr_min, fr_max, f_sample, f_plot=False):
    """
    passband filter **causal** with butterfly window with fft

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=t_series.shape[0])
    if f_plot:
        plt.figure()
        plt.title(f"Power sprectum of Butterworth band filter [{fr_min}, {fr_max}]")
        plt.plot(w * 1e-6, abs(h), label="no causal")
        plt.xlabel("MHz")
    # add causality condition
    #    add minus to have the signal in right direction, why ?
    h.imag = -hilbert(np.real(h)).imag
    if f_plot:
        print(w.shape)
        plt.plot(w * 1e-6, abs(h), ".", label="causal")
        plt.grid()
        plt.legend()
    f_fft = sf.fft(t_series.T) * h
    filtered = sf.ifft(f_fft)
    print("filtered:", filtered.shape)
    return np.real(filtered.T)


def filter_butter_band_causal_hc(t_series, fr_min, fr_max, f_sample, f_plot=False):
    """
    passband filter **causal** with butterfly window with fft

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=t_series.shape[0])
    if f_plot:
        plt.figure()
        plt.title(f"Power sprectum of Butterworth band filter [{fr_min}, {fr_max}]")
        plt.plot(w * 1e-6, abs(h), label="no causal")
        plt.xlabel("MHz")
    # add causality condition
    #    add minus to have the signal in right direction, why ?
    h.imag = -hilbert(np.real(h)).imag
    size_h = w.shape[0]
    h_hc = h[: size_h // 2 + 1]
    if f_plot:
        print(w.shape)
        plt.plot(w * 1e-6, abs(h), ".", label="causal")
        plt.grid()
        plt.legend()
    f_fft = sf.rfft(t_series.T) * h_hc
    filtered = sf.irfft(f_fft, 999)
    print("filtered:", filtered.shape)
    return filtered.T



def get_peakamptime_norm_hilbert(a2_time, a3_trace):
    """
    Get peak Hilbert amplitude norm of trace (v_max) and its time t_max without interpolation

    :param time (D,S): time, with D number of vector of trace, S number of sample
    :param traces (D,3,S): trace

    :return: t_max float(D,) v_max float(D,), norm_hilbert_amp float(D,S),
            idx_max int, norm_hilbert_amp float(D,S)
    """
    hilbert_amp = np.abs(hilbert(a3_trace, axis=-1))
    norm_hilbert_amp = np.linalg.norm(hilbert_amp, axis=1)
    # add dimension for np.take_along_axis()
    idx_max = np.argmax(norm_hilbert_amp, axis=1)[:, np.newaxis]
    t_max = np.take_along_axis(a2_time, idx_max, axis=1)
    v_max = np.take_along_axis(norm_hilbert_amp, idx_max, axis=1)
    # remove dimension (np.squeeze) to have ~vector ie shape is (n,) instead (n,1)
    return np.squeeze(t_max), np.squeeze(v_max), np.squeeze(idx_max), norm_hilbert_amp


def get_fastest_size_rfft(sig_size, f_samp_mhz, padding_fact=1):
    """

    :param sig_size:
    :param f_samp_mhz:
    :param padding_fact:

    :return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_fact >= 1
    dt_s = 1e-6 / f_samp_mhz
    fastest_size_fft = sf.next_fast_len(int(padding_fact * sig_size + 0.5))
    freqs_mhz = sf.rfftfreq(fastest_size_fft, dt_s) * 1e-6
    return fastest_size_fft, freqs_mhz


def interpol_at_new_x(a_x, a_y, new_x, kind="linear"):
    """
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(a_x, a_y, kind, bounds_error=False, fill_value=(0.0, 0.0))
    return func_interpol(new_x)


def halfcplx_fullcplx(v_half, even=True):
    """
    Return fft with full complex format where vector has half complex format,
    ie v_half=rfft(signal) in numpy/scipy convention

    For N size of signal and f frequency sampling

    numpy and scipy.fft convention:
    ===============================

    halfcplx: for N=4 =>  size of format halfcplx is N//2 + 1=3
      f*0, f*1/N, f*2/N
      - f*2/N is Nyquist frequency
      - for real signal, f*0 and f*2/N mode are real in Fourier space
            => same number of value in direct space and Fourier space to define signal

    fullcplx: for N=4
      f*0, f*1/N, -f*2/N, -f*1/N
      - Nyquist frequency is negative

    @note:
      Numpy reference : https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html

    :param v_half (array 1D complex): complex vector in half complex format, ie from rfft(signal)
    :param even (bool): True if size of signal is even

    @return (array 1D complex) : fft(signal) in full complex format
    """
    if even:
        return np.concatenate((v_half, np.flip(np.conj(v_half[1:-1]))))
    return np.concatenate((v_half, np.flip(np.conj(v_half[1:]))))


