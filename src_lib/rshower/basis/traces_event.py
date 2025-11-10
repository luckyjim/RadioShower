"""
Colley Jean-Marc, CNRS/IN2P3/LPNHE

Handling a set of 3D traces
"""

from logging import getLogger
import copy

import numpy as np
import scipy.signal as ssig
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import CheckButtons

from rshower.basis.du_network import DetectorUnitNetwork
import rshower.num.signal as rss


logger = getLogger(__name__)

G_3d_widget = None


def get_psd(trace, f_samp_mhz, nperseg=0):
    """Reference estimation of power spectrum density by Welch's method

    :param trace: floatX(nb_sample,)
    :param f_samp_mhz: frequency sampling
    :param nperseg: number of sample by periodogram
    """
    if nperseg == 0:
        nperseg = trace.shape[0] / 8

    freq, pxx_den = ssig.welch(
        trace, f_samp_mhz * 1e6, nperseg=nperseg, window="taylor", scaling="density"
    )
    return freq * 1e-6, pxx_den


def get_idx_pulse(trace, support_percent=99.99):
    # define support interval
    marge = (100 - support_percent) / 2
    trace_2 = trace**2
    c_sum = (trace_2).cumsum()
    c_sum_nor = c_sum / c_sum[-1]
    print(c_sum_nor)
    threshold = marge / 100
    for idx in range(len(trace)):
        if c_sum_nor[idx] > threshold:
            i_beg = idx - 1
            break
    threshold = (100 - marge) / 100
    print("Back search:", threshold)
    for idx in range(len(trace)):
        # print(idx, c_sum_nor[-idx])
        if c_sum_nor[-idx - 1] < threshold:
            i_end = len(trace) - idx - 1
            break
    print(i_beg, i_end)
    print(c_sum_nor[i_beg - 1 : i_beg + 2])
    print(c_sum_nor[i_end - 1 : i_end + 2])
    # # --- Tracés ---
    # plt.figure(figsize=(10,4))
    # plt.plot(trace_2)
    # plt.plot(c_sum)
    # plt.axvspan(i_beg, i_end, alpha=0.2, color='orange')
    # plt.title(f"Signal transitoire avec intervalle {support_percent:.1f}% energie")
    # plt.xlabel("idx")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    return i_beg, i_end


def get_psd_pulse(trace, f_samp_mhz, support_percent=99):
    i_beg, i_end = get_idx_pulse(trace, support_percent)
    freq, pxx_den = get_psd(trace[i_beg:i_end], f_samp_mhz, i_end - i_beg)
    return freq, pxx_den


class Handling3dTraces:
    """
    Handling a set of traces associated to one event observed on Detector Unit network

    Initialisation, with two methods:

       * init_traces()
       * optionally with init_network()

    Features:

        * some plots : trace , power spectrum, footprint, ...
        * compute time where the trace is maximun
        * compute time for each sample of trace
        * compute a common time for trace


    Public attributes:

        * name str: name of the set of trace
        * traces float(nb_du, 3, nb_sample): trace 3D
        * idx2idt int(nb_du): array of identifier of DU
        * t_start_ns float(nb_du): [ns] time of first sample of trace
        * t_samples float(nb_du, nb_dim, nb_sample): [ns]
        * f_samp_mhz float(nb_du,): [MHz] frequency sampling
        * idt2idx dict: for each identifier return the index in array
        * unit_trace str: string unit of trace
        * network object: content position network
    """

    def __init__(self, name="NotDefined"):
        logger.info(f"Create Handling3dTraces with name {name}")
        self.name = name
        self.info_shower = ""
        nb_du = 0
        nb_sample = 0
        self.nperseg = 0
        self.func_psd = get_psd_pulse
        self.traces = np.zeros((nb_du, 3, nb_sample))
        self.idx2idt = range(nb_du)
        self.t_start_ns = np.zeros((nb_du), dtype=np.int64)
        self.t_samples = np.zeros((nb_du, nb_sample), dtype=np.float64)
        self.f_samp_mhz = 0.0
        self.idt2idx = {}
        self.unit_trace = "TBD"
        self.type_trace = ""
        self._d_axis_val = {
            "idx": ["0", "1", "2"],
            "port": ["1", "2", "3"],
            "cart": ["X", "Y", "Z"],
            "dir": ["SN", "EW", "UP"],
            "shw": ["vxB", "vx(vxB)", "v"],
            "pol": [r"$e_{polar}$"],
            "tan": [r"$e_{\zenith}$", r"$e_{\azi}$"],
        }
        # blue for UP because the sky is blue
        # yellow for EW because sun is yellow
        #  and it rises in the west and sets in the east
        # k for black because the poles are white
        #  and the reverse of white (not visible on plot) is black
        self._color = ["k", "y", "b"]
        self.l_axis = self._d_axis_val["idx"]
        self.network = None
        # computing by user and store in object
        self.t_max = None
        self.v_max = None
        self.noise_inter = None
        # analogic to numeric
        self.a2n = 8192.0 / 9e5

    ### INTERNAL
    def _reset_max(self):
        self.t_max = None
        self.v_max = None

    ### INIT/SETTER

    def init_traces(self, traces, du_id=None, t_start_ns=None, f_samp_mhz=2000, f_noise=False):
        """

        :param traces: array traces 3D
        :type traces: float (nb_du, 3, nb sample)
        :param du_id:array identifier of DU
        :type du_id: int (nb_du,)
        :param t_start_ns: array time start of trace
        :type t_start_ns: int (nb_du,)
        :param f_samp_mhz: frequency sampling in MHz
        :type f_samp_mhz: float or array
        """
        assert isinstance(self.traces, np.ndarray)
        assert traces.ndim == 3
        assert traces.shape[1] == 3
        self.traces = traces
        if du_id is None:
            du_id = list(range(traces.shape[0]))
        if t_start_ns is None:
            t_start_ns = np.zeros(traces.shape[0], dtype=np.float32)
        self.idx2idt = du_id
        self.idt2idx = {idt: idx for idx, idt in enumerate(self.idx2idt)}
        self.t_start_ns = t_start_ns
        if isinstance(f_samp_mhz, (int, float)):
            self.f_samp_mhz = np.ones(len(du_id)) * f_samp_mhz
        else:
            self.f_samp_mhz = f_samp_mhz
        assert isinstance(self.t_start_ns, np.ndarray)
        assert traces.shape[0] == len(du_id)
        assert len(du_id) == t_start_ns.shape[0]
        self._define_t_samples()
        if f_noise:
            self.set_psd_noise(10)
        else:
            self.set_psd_pulse()

    def init_network(self, du_pos):
        """

        :param du_pos: position of DU in cartesian coordinate
        :type du_pos: float(nb_du,3)
        """
        self.network = DetectorUnitNetwork()
        self.network.init_pos_id(du_pos, self.idx2idt)

    def set_noise_interval(self, i_beg, i_end):
        assert i_beg < i_end
        assert i_beg >= 0
        assert i_end <= self.traces.shape[2]
        self.noise_inter = [i_beg, i_end]

    def set_unit_axis(self, unit_trace="TBD", axis_name="idx", type_trace="Trace"):
        """

        :param str_unit: define the unit of traces
        :type str_unit: string
        :param axis_name: define the type of axis, must in self._d_axis_val
        :type axis_name: string
        :param type_tr: define type of traces
        :type type_tr: string
        """
        assert isinstance(unit_trace, str)
        assert isinstance(axis_name, str)
        assert isinstance(type_trace, str)
        self.type_trace = type_trace
        self.unit_trace = unit_trace
        self.l_axis = self._d_axis_val[axis_name]

    def set_periodogram(self, size):
        """

        :param size: size of periodogram
        """
        assert size > 0
        raise
        self.nperseg = size

    def set_psd_noise(self, nb_period):
        self.trace_with_pulse = False
        self.nperseg = 2 * self.get_size_trace() // nb_period

    def set_psd_pulse(self, percent=None):
        self.trace_with_pulse = True
        if percent:
            self.psd_percent = percent
        else:
            if self.traces[0, 0, :10].std() == 0:
                # no noise case
                print("Pulse no noise")
                self.psd_percent = 99.99
            else:
                self.psd_percent = 99

    ### OPERATIONS

    def apply_lowpass(self, cutoff_mhz, order=5):
        normal_cutoff = cutoff_mhz / (self.f_samp_mhz[0] / 2)
        coeff_b, coeff_a = ssig.butter(order, normal_cutoff, btype="low", analog=False)
        # coef = ssig.firwin(numtaps, cutoff_mhz * 1e-6, fs=self.f_samp_mhz * 1e6, pass_zero="lowpass")
        filtered = ssig.filtfilt(coeff_b, coeff_a, self.traces)
        self.traces = filtered.real
        self._reset_max()

    def apply_bandpass(self, fr_min, fr_max, causal=True, order=9):
        """
        band filter with butterfly window

        :return: filtered trace in time domain
        """
        low = fr_min * 1e6
        high = fr_max * 1e6
        f_hz = self.f_samp_mhz[0] * 1e6
        if causal:
            # second order section format more stable for causal filter
            sos = ssig.butter(order, [low, high], btype="bandpass", fs=f_hz, output="sos")
            filtered = ssig.sosfilt(sos, self.traces)
        else:
            coeff_b, coeff_a = ssig.butter(order, [low, high], btype="bandpass", fs=f_hz)
            filtered = ssig.filtfilt(coeff_b, coeff_a, self.traces)
        self.traces = filtered.real
        self._reset_max()

    def _define_t_samples(self):
        """
        Define time sample for the duration of the trace
        """
        if self.t_samples.size == 0:
            delta_ns = 1e3 / self.f_samp_mhz
            nb_sample = self.traces.shape[2]
            # to use numpy broadcast I need to transpose
            t_trace = (
                np.outer(
                    np.arange(0, nb_sample, dtype=np.float64),
                    delta_ns * np.ones(self.traces.shape[0]),
                )
                + self.t_start_ns
            )
            self.t_samples = t_trace.transpose()
            logger.info(f"shape t_samples =  {self.t_samples.shape}")

    def keep_only_trace_with_ident(self, l_idt):
        """Keep trace with identifier defined in list <l_idt>

        :param l_idt: list of identifier of trace
        :type l_idt: list int or string
        """
        l_idx = [self.idt2idx[idt] for idt in l_idt]
        self.keep_only_trace_with_index(l_idx)

    def keep_only_trace_with_index(self, l_idx):
        """Keep trace at index defined in list <l_idx>

        :param l_idx:list of index of trace
        :type l_idt: list int
        """
        if len(l_idx) == 0:
            logger.info(f"Keep 0 elements, => empty event")
            self.idt2idx = {}
            self.idx2idt = []
            self.traces = None
            return
        du_id = [self.idx2idt[idx] for idx in l_idx]
        self.idx2idt = du_id
        self.idt2idx = {}
        for idx, ident in enumerate(self.idx2idt):
            self.idt2idx[ident] = idx
        self.traces = self.traces[l_idx]
        self.f_samp_mhz = self.f_samp_mhz[l_idx]
        self.t_start_ns = self.t_start_ns[l_idx]
        if self.t_samples.shape[0] > 0:
            self.t_samples = self.t_samples[l_idx]
        if self.network:
            self.network = copy.deepcopy(self.network)
            self.network.keep_only_du_with_index(l_idx)
        self._reset_max()

    def reduce_nb_trace(self, new_nb_du):
        """reduces the number of traces to the first <new_nb_du>

        Feature to reduce computation, for debugging

        :param new_nb_du: keep only new_nb_du first DU
        :type new_nb_du: int
        """
        assert new_nb_du > 0
        assert new_nb_du <= self.get_nb_trace()
        self.idx2idt = self.idx2idt[:new_nb_du]
        self.traces = self.traces[:new_nb_du, :, :]
        self.t_start_ns = self.t_start_ns[:new_nb_du]
        if self.t_samples.shape[0] > 0:
            self.t_samples = self.t_samples[:new_nb_du, :]
        self.network.reduce_nb_du(new_nb_du)

    def reduce_nb_sample(self, new_nb_sample):
        """ """
        # assert new_nb_sample > 0
        # assert new_nb_sample <= self.get_size_trace()
        self.traces = self.traces[:, :, :new_nb_sample]
        self.t_samples = self.t_samples[:, :new_nb_sample]
        self.set_periodogram(new_nb_sample)

    def downsize_sampling(self, fact):
        """Downsampling with scipy decimate function

        :param fact: the downsampling factor
        :type fact: int
        """
        # self.traces = self.traces[:, :, ::fact]
        logger.info(f"{self.traces.shape} ")
        self.traces = ssig.decimate(self.traces, fact)
        logger.info(f"{self.traces.shape} ")
        self.f_samp_mhz /= fact
        self.t_samples = np.zeros((0, 0), dtype=np.float64)
        self._define_t_samples()

    def remove_trace_low_signal(self, threshold, norm_traces=None):
        """Remove trace where <norm_traces> is lower than <threshold>

        :param threshold: value > 0
        :type threshold: number
        """
        if norm_traces is None:
            norm_traces = self.get_max_norm()
        else:
            assert norm_traces.shape[0] == self.get_nb_trace()
        idx_ok = np.squeeze(np.argwhere(norm_traces > threshold))
        idx_ok = np.atleast_1d(idx_ok)
        logger.info(f"Keep {len(idx_ok)} DU on {self.get_nb_trace()}")
        self.keep_only_trace_with_index(idx_ok)
        return idx_ok

    def copy(self, new_traces=None, deepcopy=True):
        """Return a copy of current object where traces can be modify

        The type of copy is copy with reference, not a deepcopy
        https://stackoverflow.com/questions/3975376/why-updating-shallow-copy-dictionary-doesnt-update-original-dictionary/3975388#3975388

        if new_traces is :
          * None : object with same value
          * 0 : the return object has a traces with same shape but set to 0
          * np.array : the return object has new_traces as traces

        :param new_traces: if array must be have the same shape
        :type new_traces: array/None/0
        :return: Handling3dTraces instance
        """
        if deepcopy:
            my_copy = copy.deepcopy(self)
        else:
            my_copy = copy.copy(self)
        if new_traces is not None:
            if isinstance(new_traces, np.ndarray):
                assert self.traces.shape == new_traces.shape
            elif new_traces == 0:
                new_traces = np.zeros_like(self.traces)
            my_copy.traces = new_traces
            self._reset_max()
        return my_copy

    ### GETTER

    def get_delta_t_ns(self):
        """Return sampling rate in ns

        :return: float(nb_3dtrace,)
        """
        ret = 1e3 / self.f_samp_mhz
        return ret

    def get_max_abs(self):
        """Find absolute maximal value in trace for each detector

        :return:  array max of abs value
        :rtype: float (nb_du,)
        """
        return np.max(np.abs(self.traces), axis=(1, 2))

    def get_max_norm(self):
        """Return array of maximal of 3D norm in trace for each detector

        :return: array norm of traces
        :rtype: float (nb_du,)
        """
        # norm on 3D composant => axis=1
        # max on all norm => axis=1
        return np.max(np.linalg.norm(self.traces, axis=1), axis=1)

    def get_norm(self):
        """Return norm of traces for each time sample

        :return:  norm of traces for each time sample
        :rtype: float (nb_du, nb sample)
        """
        return np.linalg.norm(self.traces, axis=1)

    def get_tmax_vmax(self, hilbert=True, interpol="parab"):
        """Return time where norm of the amplitude of the Hilbert tranform  is max

        :param hilbert: True for Hilbert envelop else norm L2
        :type hilbert: bool
        :param interpol: keyword in no, auto, parab
        :type interpol: string
        :return: time of max and max
        :rtype: float(nb_du,) , float(nb_du,)
        """
        if hilbert:
            tmax, vmax, idx_max, tr_norm = rss.get_peakamptime_norm_hilbert(
                self.t_samples, self.traces
            )
        else:
            tr_norm = np.linalg.norm(self.traces, axis=1)
            idx_max = np.argmax(tr_norm, axis=1)
            idx_max = idx_max[:, np.newaxis]
            tmax = np.squeeze(np.take_along_axis(self.t_samples, idx_max, axis=1))
            vmax = np.squeeze(np.take_along_axis(tr_norm, idx_max, axis=1))
        if interpol == "no":
            self.t_max = tmax
            self.v_max = vmax
            return tmax, vmax
        if interpol not in ["parab", "auto"]:
            raise
        t_max = np.empty_like(tmax)
        v_max = np.empty_like(tmax)
        for idx in range(self.get_nb_trace()):
            logger.debug(f"{idx} {self.idx2idt[idx]} {idx_max[idx]}")
            if interpol == "parab":
                t_max[idx], v_max[idx] = rss.find_max_with_parabola_interp_3pt(
                    self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
                )
            else:
                t_max[idx], v_max[idx] = rss.find_max_with_parabola_interp(
                    self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
                )
            logger.debug(f"{t_max[idx]} ; {v_max[idx]}")
        self.t_max = t_max
        self.v_max = v_max
        return t_max, v_max

    def get_min_max_t_start(self):
        """Return time interval of time start of trace

        :return: first and last time start
        :rtype: float, float
        """
        return self.t_start_ns.min(), self.t_start_ns.max()

    def get_nb_trace(self):
        """Return the number of 3d traces
        :return: number of DU
        :rtype: int
        """
        return len(self.idx2idt)

    def get_size_trace(self):
        """Return the number of sample in trace

        :return: number of sample in trace
        :rtype: int
        """
        return self.traces.shape[2]

    def get_std_noise(self, idx=None):
        size_noise = np.min([200, int(self.get_size_trace() / 20)])
        if idx is None:
            noise = np.std(self.traces[:, :, -size_noise:], axis=-1)
            assert noise.shape[0] == self.get_nb_trace()
            return noise
        else:
            return np.std(self.traces[idx, :, -size_noise:], axis=-1)

    def get_snr_and_noise(self):
        """Return a crude estimation of SNR and noise level

        Crude estimation because:
           * noise is estimated at the end of trace
           * max value has different estimator.

        :return: snr float(nb_trace,), noise(nb_trace,)
        """
        noise_mean = np.mean(self.get_std_noise(), axis=1)
        t_max, v_max = self.get_tmax_vmax(hilbert=False, interpol="no")
        snr = v_max / noise_mean
        return snr, noise_mean, t_max, v_max

    def get_extended_traces(self):
        """Return extended traces to time interval of event

        Compute and return traces extended to the entire duration of
        the event with common time

        :return: common time, extended traces
        :rtype: float (nb extended sample), float (nb_du, 3, nb extended sample)
        """
        size_tr = int(self.get_size_trace())
        t_min, t_max = self.get_min_max_t_start()
        delta = self.get_delta_t_ns()[0]
        nb_sample_mm = (t_max - t_min) / delta
        nb_sample = int(np.rint(nb_sample_mm) + size_tr)
        extended_traces = np.zeros((self.get_nb_trace(), 3, nb_sample), dtype=self.traces.dtype)
        # don't use np.uint64 else int+ int =float ??
        i_beg = np.rint((self.t_start_ns - t_min) / delta).astype(np.uint32)
        for idx in range(self.get_nb_trace()):
            extended_traces[idx, :, i_beg[idx] : i_beg[idx] + size_tr] = self.traces[idx]
        common_time = t_min + np.arange(nb_sample, dtype=np.float64) * delta
        return common_time, extended_traces

    def get_psd_trace_idx(self, idx):
        l_psd = []
        if self.trace_with_pulse:
            for idx_axis, axis in enumerate(self.l_axis):
                trace = self.traces[idx, idx_axis]
                i_beg, i_end = get_idx_pulse(trace, self.psd_percent)
                print(i_beg, i_end)
                freq, pxx_den = get_psd(trace[i_beg:i_end], self.f_samp_mhz[idx], i_end - i_beg)
                l_psd.append([freq, pxx_den])
        else:
            for idx_axis, axis in enumerate(self.l_axis):
                freq, pxx_den = get_psd(
                    self.traces[idx, idx_axis], self.f_samp_mhz[idx], self.nperseg
                )
                l_psd.append([freq, pxx_den])
        return l_psd

    def to_digit(self, in_place=False, int_type=np.int16):
        if in_place:
            self.traces *= self.a2n
            np.ceil(self.traces, out=self.traces, dtype=int_type)
            self.unit_trace = "ADU"
            return
        return np.ceil(self.traces * self.a2n, dtype=int_type)

    def to_analog(self, new_unit, in_place=False):
        if in_place:
            self.traces /= self.a2n
            self.unit_trace = new_unit
            return
        return self.traces / self.a2n

    ### PLOTS

    def plot_trace_3d_idx(self, idx):
        global G_3d_widget

        def change_checkbutton(label):
            print(label)
            if label == labels[0]:
                if activated[0]:
                    ax1.axis("auto")
                else:
                    ax1.axis("equal")
                activated[0] = not activated[0]
                plt.draw()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        # 3D plot
        ax1 = fig.add_subplot(projection="3d")
        ax.tick_params(labelleft=False, labelbottom=False)
        data_xd = self.traces[idx]
        ln = ax1.scatter(data_xd[0], data_xd[1], data_xd[2])
        ax1.axis("equal")
        s_title = f"{self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        s_title += f"\n$F_{{sampling}}$={self.f_samp_mhz[idx]:.1f} MHz"
        s_title += f"; {self.get_size_trace()} samples"
        ax1.set_title(s_title)
        ax1.grid()
        # Button
        axcheck = fig.add_axes([0.7, 0.05, 0.2, 0.075])
        labels = ["Same scale"]
        activated = [True]
        chxbox = CheckButtons(axcheck, labels, activated)
        # why global
        # https://stackoverflow.com/questions/42419139/matplotlib-widgets-button-doesnt-work-inside-a-class
        G_3d_widget = chxbox
        chxbox.on_clicked(change_checkbutton)
        plt.show(block=False)

    def plot_trace_idx(self, idx, to_draw="012"):  # pragma: no cover
        """Draw 3 traces associated to DU with index idx

        :param idx: index of DU to draw
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self._define_t_samples()
        plt.figure()
        s_title = f"{self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        s_title += f"\n$F_{{sampling}}$={self.f_samp_mhz[idx]:.1f} MHz"
        s_title += f"; {self.get_size_trace()} samples"
        plt.title(s_title)
        std_3d = self.get_std_noise(idx)
        for idx_axis, axis in enumerate(self.l_axis):
            if str(idx_axis) in to_draw:
                std_axis = std_3d[idx_axis]
                plt.plot(
                    self.t_samples[idx],
                    self.traces[idx, idx_axis],
                    self._color[idx_axis],
                    label=axis + r", $\sigma_{noise}\approx$" + f"{std_axis:.1e}",
                )
        if self.t_max is not None:
            snr = self.v_max[idx] / std_3d.mean()
            plt.plot(
                self.t_max[idx],
                self.v_max[idx],
                "d",
                label=f"Max {self.v_max[idx]:.4e} {self.unit_trace}\n"
                + r"$SNR\approx$"
                + f"{snr:.0f}",
            )
        plt.ylabel(f"{self.unit_trace}")
        plt.xlabel(f"ns\n{self.name}")
        plt.grid()
        plt.legend()

    def plot_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """Draw 3 traces associated to DU idx2idt

        :param idx: index of DU to draw
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_trace_idx(self.idt2idx[du_id], to_draw)

    def plot_psd_trace_idx(self, idx, to_draw="012"):  # pragma: no cover
        """Draw power spectrum for 3 traces associated to DU at index idx

        :param idx: index of trace
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self._define_t_samples()
        plt.figure()
        l_len = []
        l_freq = []
        l_psd = self.get_psd_trace_idx(idx)
        for idx_axis, axis in enumerate(self.l_axis):
            if str(idx_axis) in to_draw:
                freq = l_psd[idx_axis][0]
                pxx_den = l_psd[idx_axis][1]
                plt.semilogy(freq[2:], pxx_den[2:], self._color[idx_axis], label=axis)
                # plt.plot(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
        m_title = f"Power spectrum density of {self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        m_title += f"\nPeriodogram has {(len(freq)-1)*2} samples, delta freq {freq[1]:.2f}MHz"
        plt.title(m_title)
        plt.ylabel(rf"({self.unit_trace})$^2$/Hz")
        plt.xlabel(f"MHz\n{self.name}")
        plt.xlim([0, freq[-1]])
        plt.grid()
        plt.legend()
        #
        if self.noise_inter is not None:
            noise = Handling3dTraces("Noise")
            i_beg, i_end = self.noise_inter[0], self.noise_inter[1]
            noise.init_traces(
                self.traces[:, :, i_beg:i_end], self.idx2idt, f_samp_mhz=self.f_samp_mhz
            )
            noise.type_trace = "Noise"
            noise.plot_psd_trace_idx(idx)

    def plot_psd_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """Draw power spectrum for 3 traces associated to DU idx2idt

        :param idx2idt: DU identifier
        :type idx2idt: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_psd_trace_idx(self.idt2idx[du_id], to_draw)

    def plot_all_traces_as_image(self):  # pragma: no cover
        """Interactive image double click open traces associated"""
        norm = self.get_norm()
        _ = plt.figure()
        # fig.canvas.manager.set_window_title(f"{self.name}")
        plt.title(f"Norm of all traces {self.type_trace} in event")
        col_log = colors.LogNorm(clip=False)
        im_traces = plt.imshow(norm, cmap="Blues", norm=col_log)
        plt.colorbar(im_traces)
        plt.xlabel(f"Index sample\nFile: {self.name}")
        plt.ylabel("Index DU")

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                idx = int(event.ydata + 0.5)
                self.plot_trace_idx(idx)
                self.plot_psd_trace_idx(idx)
                plt.show()

        plt.connect("button_press_event", on_click)

    def plot_histo_t_start(self):  # pragma: no cover
        """Histogram of time start"""
        plt.figure()
        plt.title(rf"{self.name}\nTime start histogram")
        plt.hist(self.t_start_ns)
        plt.xlabel("ns")
        plt.grid()

    def plot_footprint_4d_max(self):  # pragma: no cover
        """Plot time max and max value by component"""
        v_max = np.max(np.abs(self.traces), axis=2)
        self.network.plot_footprint_4d(self, v_max, "3D", unit=self.unit_trace)

    def plot_footprint_val_max(self):  # pragma: no cover
        """Plot footprint max value"""
        self.network.plot_footprint_1d(
            self.get_max_norm(),
            f"Max ||{self.type_trace}||",
            self,
            unit=self.unit_trace,
        )

    def plot_footprint_time_max(self):  # pragma: no cover
        """Plot footprint time associated to max value"""
        tmax, _ = self.get_tmax_vmax(False)
        self.network.plot_footprint_1d(tmax, "Time of max value", self, scale="lin", unit="ns")

    def plot_footprint_time_slider(self):  # pragma: no cover
        """Plot footprint max value"""
        if self.network:
            a_time, a_values = self.get_extended_traces()
            self.network.plot_footprint_time(a_time, a_values, "Max value")
        else:
            logger.error("DU network isn't defined, can't plot footprint")

    def plot_tmax_vmax(self, hline=None, vline=None):
        """
        :param hline: None or [value in [0,100], "legend hline"]
        :param vline: None or [number, "legend vline"]
        """
        plt.figure()
        plt.title(f"Time of max, max value\n{self.name}")
        t_tot = self.get_size_trace() * self.get_delta_t_ns()[0]
        t_max_rel = self.t_max - self.t_samples[:, 0]
        plt.scatter(100 * t_max_rel / t_tot, self.v_max)
        plt.xlim([0, 100])
        plt.xlabel("Position of time of max in trace, % ")
        plt.ylabel(f"Max value of trace, {self.unit_trace}")
        if hline:
            plt.hlines(hline[0], 0, 100, label=hline[1], linestyles="-")
        if vline:
            v_inf = np.min(self.v_max)
            v_sup = np.max(self.v_max)
            if hline:
                v_inf = np.min([v_inf, hline[0]])
                v_sup = np.max([v_sup, hline[0]])
            plt.vlines(vline[0], v_inf, v_sup, label=vline[1], linestyles="-.")
        plt.yscale("log")
        plt.grid()
        plt.legend()
