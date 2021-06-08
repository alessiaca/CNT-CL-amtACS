# Help functions for Closed-Loop Stimulation

import numpy as np
import pandas
import mne
from mne.time_frequency import psd_multitaper, psd_welch
import matplotlib.pyplot as plt
from scipy import stats, linalg
from scipy.stats import median_absolute_deviation, zscore, wasserstein_distance
from mne.preprocessing import compute_current_source_density
from mne.time_frequency import psd_welch, psd_array_welch
from scipy.signal import hilbert
from scipy.signal import firwin, filtfilt
from scipy.fftpack import next_fast_len
from ordpy.ordpy import ordinal_distribution


def circ_mean(phases):
    return np.angle(np.exp(1j * phases).mean())

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def DFT(data, frequency, sfreq):
    """Computes the DFT at the target frequency, returns the amplitude and the phase"""
    data = data.flatten()
    n_samps = len(data)
    res = np.sum([np.exp(-2 * np.pi * 1j * frequency * i / sfreq) * data[i] for i in np.arange(n_samps-1)])
    return np.abs(res), np.angle(res)

def make_P_laplace(ch_names, inner, outer):
    """Returns a vector (spatial filter) with 1s at the index of the outer channels
     and -x (x=length of the outer picks) at the index of the inner channel"""
    n_outer = len(outer)
    chidx_outer = [ch_names.index(ch) for ch in outer]
    P_laplace = np.zeros((1, len(ch_names)))
    for i in range(n_outer):
        P_laplace[0, chidx_outer[i]] = 1
    P_laplace[0, ch_names.index(inner)] = -1 * n_outer
    return P_laplace


def find_n_nulls(A, B, D, M):
    N = A.shape[0]
    ses = np.zeros((N, N))
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        ses[n_nulls] = (np.diag(B) - np.diag(P.dot(A).dot(P.T))) ** 2
    return np.argmin(ses, axis=0)


def compute_P(A, B, n_nulls=None):
    '''n_nulls = vector or int specifiying the number of nulls per channel'''
    eigen_values, eigen_vectors = linalg.eig(A, B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:, ix].T
    M = linalg.pinv2(D)
    if n_nulls is None:
        n_nulls = find_n_nulls(A, B, D, M)
    else:
        if not hasattr(n_nulls, "__len__"): n_nulls = np.array([n_nulls] * len(M))
    Ps = []
    for i, n_null in enumerate(n_nulls):
        DI = np.ones(M.shape[0])
        DI[:n_null] = 0
        DI = np.diag(DI)
        Ps.append(M[i]@DI@D)
    P = np.array(Ps)
    return P, n_nulls


def SASS(raw_stim, raw_no_stim,SASS_lfreq=None,SASS_hfreq=None,filter_type='long',filter_coeffs=None, n_nulls=None):

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_no_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_no_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    else:
        data = raw_no_stim._data
    B = np.cov(data)

    if filter_type == 'short':
        data = filtfilt(filter_coeffs,1,raw_stim.copy()._data)
    elif filter_type == 'long':
        data = raw_stim.copy().filter(SASS_lfreq, SASS_hfreq)._data
    else:
        data=raw_stim._data
    A = np.cov(data)

    P, _ = compute_P(A, B, n_nulls)
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

    P, _ = compute_P(A, B)
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
        P, _ = compute_P(A, B)
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
        P, _ = compute_P(A, B,P_laplace)
        raw_stim._data[:,i:i+window_size] = P.dot(raw_stim._data[:,i:i+window_size])

    return raw_stim


def compute_phase_diff(signal1, signal2):
    signal_len = signal1.shape[-1]
    fft_len = next_fast_len(signal_len)
    return wrap(np.angle(hilbert(signal1,fft_len)) - np.angle(hilbert(signal2,fft_len)))[...,:signal_len]


def compute_phase(signal):
    """Compute the phase of a signal using the hilbert transform (with optimal length for faster computation)"""
    signal_len = signal.shape[-1]
    fft_len = next_fast_len(signal_len)
    return np.angle(hilbert(signal, fft_len)[..., :signal_len])


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
    ch_names = np.array(raw.ch_names)
    if use_zscore:
        vars = np.var(raw._data, 1)
        bads_zscore = ch_names[np.abs(zscore(vars)) > 1.645]
    else:
        bads_zscore = []
    bads_sat = ch_names[np.any(np.abs(raw._data) > 0.02, axis=1)]
    return np.unique(np.concatenate((bads_zscore, bads_sat)))


def sine_func(x,a,b, offset):
    # sine wave for curve fit
    return np.abs(a)*np.sin(2*np.pi*(1/8)*x+b) + offset


def get_flicker_events(raw, threshold=3, miniti_sec=0.4, name="audio"):
    # Get the time points of the flicker as well as the onset of one flicker set (separated by a ITI)
    audio_events = np.where(np.diff(
        (np.abs(stats.zscore(raw.copy().pick_channels([name])._data.flatten())) > threshold).astype(
            'int')) > 0)[0]
    miniti = raw.time_as_index(miniti_sec)[0]
    audio_events_onset = [audio_events[ix] for ix in range(1, audio_events.size)
                            if audio_events[ix] - audio_events[ix - 1] > miniti]
    return audio_events, audio_events_onset

def compute_ITC(phase_signal, events, trial_length=2, sfreq=500, frequency=10):
    '''Compute the inter trial coherence between a phase of a signal and the phase of a  flicker (determined by onset events with specific trial length)'''
    t = np.arange(0,trial_length,1/sfreq)
    phases_flicker_trial = wrap(2*np.pi*frequency*t)
    nsamp_trial = phases_flicker_trial.size
    nsamp_start = 0
    phasediffs = []
    for ev in events:
        if ev < len(phase_signal)-nsamp_trial:
            phasediffs.append(circ_mean(wrap(phase_signal[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
    return plv(np.array(phasediffs)), np.array(phasediffs)


def compare_rank_vector_distribution(signal1, signal2, dx, taux, all_rank_vectors, probs2_ordered=None):
    "Compare the rank vector distribution of the given signals, possibility to input one probability distribution"

    # Compute the ordinal distribution of signal1
    vectors1, probs1 = ordinal_distribution(signal1, dx=dx, taux=taux)
    # Order the probabilities in respect to the rank vectors (no all rank vectors are used and they are in
    # different orders)
    idx1 = np.array([[i for i, row in enumerate(all_rank_vectors) if (x == row).all()] for x in vectors1])
    probs1_ordered = np.zeros((all_rank_vectors.shape[0], 1))
    probs1_ordered[idx1.flatten()] = probs1[:, np.newaxis]

    # If the ordered probability distribution is not given generate it
    if not probs2_ordered:
        vectors2, probs2 = ordinal_distribution(signal2, dx=dx, taux=taux)
        idx2 = np.array([[i for i, row in enumerate(all_rank_vectors) if (x == row).all()] for x in vectors2])
        probs2_ordered = np.zeros((all_rank_vectors.shape[0], 1))
        probs2_ordered[idx2.flatten()] = probs2[:,np.newaxis]

    # Compute the distance between the distributions
    return wasserstein_distance(probs1_ordered.flatten(), probs2_ordered.flatten()), probs2_ordered
