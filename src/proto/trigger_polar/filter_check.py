'''
Created on 6 nov. 2025

@author: jcolley
'''

# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz, firwin, filtfilt
from matplotlib import pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def fir_lowpass(data, cutoff, fs, order=5):
    b = firwin(10, cutoff, fs=fs, pass_zero='lowpass')
    y = filtfilt(b, [1.0], data)
    return y,b



# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
b_fir = firwin(50, cutoff, fs=fs, pass_zero='lowpass')
# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
wf, hf = freqz(b_fir, [1.0], worN=8000)

plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(0.5*fs*wf/np.pi, np.abs(hf), 'g')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0             # seconds
n = int(T * fs)     # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) \
        + 0.5*np.sin(12.0*2*np.pi*t)
        
data_clean = np.sin(1.2*2*np.pi*t)

# Filter the data, and plot both the original and filteplt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')red signals.
y = butter_lowpass_filter(data, cutoff, fs, order)
y_fir,_ = fir_lowpass(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g+', linewidth=2, label='filtered data')
plt.plot(t, y_fir, 'r-', linewidth=2, label='filtered data FIR')
plt.plot(t, data_clean, 'k-', linewidth=2, label='data clean')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

if __name__ == '__main__':
    pass