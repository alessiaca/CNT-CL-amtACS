# Help functions for Closed-Loop Stimulation

import numpy as np
import pandas
import mne
from mne.time_frequency import psd_multitaper, psd_welch
import matplotlib.pyplot as plt
from scipy import stats, linalg
from scipy.stats import median_absolute_deviation, zscore
from mne.preprocessing import compute_current_source_density
from mne.time_frequency import psd_welch, psd_array_welch
from scipy.signal import hilbert
from scipy.signal import firwin, filtfilt
from scipy.fftpack import next_fast_len


def circ_mean(phases):
    return np.angle(np.exp(1j * phases).mean())


def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi


def find_n_nulls(A, B, D, M, P_laplace):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        P = P_laplace.dot(P)
        se = (P_laplace.dot(B).dot(P_laplace.T) - P.dot(A).dot(P.T)) ** 2
        mse = np.mean(se)
        mses.append(mse)
    return np.argmin(mses)


def compute_P(A, B,P_laplace):
    eigen_values, eigen_vectors = linalg.eig(A, B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:, ix].T
    M = linalg.pinv2(D)
    n_nulls = find_n_nulls(A, B, D, M,P_laplace)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    return P


def SASS(raw_stim, raw_no_stim,SASS_lfreq,SASS_hfreq,P_laplace,filter_type='long',filter_coeffs=None):

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_no_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_no_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    B = np.cov(data)

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    A = np.cov(data)

    P = compute_P(A, B,P_laplace)
    raw_stim._data = P.dot(raw_stim._data)

    return raw_stim

def combined_SASS(raw_stim,raw_replay,raw_no_stim,SASS_lfreq,SASS_hfreq,filter_type='long',filter_coeffs=None):

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_no_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_no_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    B = np.cov(data)

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    A = np.cov(data)

    P = compute_P(A, B)
    raw_stim._data = P.dot(raw_stim._data)
    raw_replay._data = P.dot(raw_replay._data)

    return raw_stim, raw_replay

def combined_window_SASS(raw_stim,raw_replay,raw_no_stim,SASS_lfreq,SASS_hfreq,window_size,filter_type='long',filter_coeffs=None):

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_no_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_no_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    B = np.cov(data)

    if filter_type == 'short':
        stim_filtered_data = filtfilt(filter_coeffs, 1, raw_stim.copy()._data)
    elif filter_type == 'long':
        stim_filtered_data = raw_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data

    for i in np.arange(0,raw_stim.n_times,window_size):
        A = np.cov(stim_filtered_data[:,i:i+window_size])
        P = compute_P(A, B)
        raw_stim._data[:,i:i+window_size] = P.dot(raw_stim._data[:,i:i+window_size])
        raw_replay._data[:,i:i+window_size] = P.dot(raw_replay._data[:,i:i+window_size])

    return raw_stim, raw_replay


def window_SASS(raw_stim,raw_no_stim,SASS_lfreq,SASS_hfreq,window_size,P_laplace,filter_type='long',filter_coeffs=None):

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_no_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_no_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    B = np.cov(data)

    if filter_type == 'short':
        stim_filtered_data = filtfilt(filter_coeffs, 1, raw_stim.copy()._data)
    elif filter_type == 'long':
        stim_filtered_data = raw_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data

    for i in np.arange(0,raw_stim.n_times,window_size):
        A = np.cov(stim_filtered_data[:,i:i+window_size])
        P = compute_P(A, B,P_laplace)
        raw_stim._data[:,i:i+window_size] = P.dot(raw_stim._data[:,i:i+window_size])

    return raw_stim


def compute_phase_diff(signal1, signal2):
    signal_len = signal1.shape[-1]
    fft_len = next_fast_len(signal_len)
    return wrap(np.angle(hilbert(signal1,fft_len)) - np.angle(hilbert(signal2,fft_len)))[...,:signal_len]



def get_target_chan(raw,inner_pick,outer_picks):
    """Computes the laplace over 3 input electrodes"""
    n_outer = len(outer_picks)
    if len(raw._data.shape) == 2:
        target_chan = (np.sum(raw.get_data(outer_picks),axis = 0) - n_outer * raw.get_data(inner_pick)).flatten()
    else:
        target_chan = np.sum(raw.get_data(outer_picks),axis = 1) - n_outer * np.squeeze(raw.get_data(inner_pick))
    return target_chan

def fir_coeffs(freq,fs=500,n_taps=825,validate=False):
    coeffs = firwin(numtaps=n_taps,cutoff=[freq-1,freq+1],fs=fs,pass_zero='bandpass')
    if validate:
        noise = np.random.rand(fs*60)
        noise_filt = filtfilt(coeffs,1,noise)
        pxx_noise,f = psd_array_welch(noise,sfreq=fs,fmin=1,fmax=20,n_fft=2**10)
        pxx_noise_filt,f = psd_array_welch(noise_filt,sfreq=fs,fmin=1,fmax=20,n_fft=2**10)
        plt.semilogy(f,pxx_noise)
        plt.semilogy(f,pxx_noise_filt)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (V**2)')
        plt.show()
    return coeffs


def find_bad_channels(raw,use_zscore=False):
    """Find channels whose broadband power deviates from the mean or are saturated"""
    tmp_raw = mne.set_eeg_reference(raw.copy())[0]
    ch_names = np.array(raw.ch_names)
    if use_zscore:
        vars = np.var(tmp_raw._data, 1)
        bads_zscore = ch_names[np.abs(zscore(vars)) > 1.645]
    else:
        bads_zscore = []
    bads_sat = ch_names[np.any(np.abs(tmp_raw._data) > 0.4,axis=1)]
    return np.unique(np.concatenate((bads_zscore,bads_sat)))

def sine_func(x,a,b, offset):
    # sine wave for curve fit
    return np.abs(a)*np.sin(2*np.pi*(1/8)*x+b) + offset
