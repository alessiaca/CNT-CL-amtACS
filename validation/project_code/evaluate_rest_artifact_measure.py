# Evaluate rest artifact measure with help of flicker and rivalry data

# Prepare environment
import numpy as np
import mne
from mne.time_frequency import psd_welch, psd_array_welch,tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import tkinter as tk
from tkinter import filedialog
from scipy.io import savemat, loadmat
from scipy import linalg, stats
from scipy.optimize import curve_fit
from main_code.utils import find_bad_channels, get_flicker_events, \
      SASS, compute_phase, compute_ITC
from scipy.signal import firwin, filtfilt

# Loop over the flicker dataset

# Set the folder path
folder_path = "C:/Users/alessia/Charité - Universitätsmedizin Berlin/Haslacher, David - SASS_data/"
n_par = 6
frequency = 10; lfreq = frequency - 1; hfreq = frequency + 1;

for i_par in range(1,n_par+1):

    # Load the data
    raw_no_stim = mne.io.read_raw_brainvision(folder_path + f'p{i_par}/no_stim.vhdr', preload=True)
    raw_stim = mne.io.read_raw_brainvision(folder_path + f'p{i_par}/open.vhdr', preload=True)

    # Get the flicker events
    no_stim_flicker_events, no_stim_flicker_event_onset = get_flicker_events(raw_stim)
    stim_flicker_events, stim_flicker_event_onset = get_flicker_events(raw_no_stim)

    # Drop non EEG channels
    non_EEG = ['env', 'sass_output', 'Flicker', 'audio', 'stim', 'ECG']
    raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in non_EEG])
    raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in non_EEG])

    # Find bad EEG channels
    bads_no_stim = find_bad_channels(raw_no_stim, use_zscore=True)
    bads_stim = find_bad_channels(raw_stim, use_zscore=True)

    # Drop bad channels
    raw_no_stim.drop_channels(bads_no_stim)
    raw_stim.drop_channels(bads_stim)
    n_chans = len(raw_stim.ch_names)

    # Filter the data
    raw_no_stim.filter(lfreq, hfreq)
    raw_stim.filter(lfreq, hfreq)

    # Get the target channel of the no stim data
    chidx = [raw_stim.ch_names.index(chname) for chname in raw_stim.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
    phase_no_stim = compute_phase(raw_no_stim._data[chidx].mean(0))
    phase_stim = compute_phase(raw_stim._data[chidx].mean(0))

    # Compute the ITC of the no stimulation data
    ITC_no_stim, phase_diffs_no_stim = compute_ITC(phase_no_stim, no_stim_flicker_event_onset)
    ITC_stim, phase_diffs_stim = compute_ITC(phase_stim, stim_flicker_event_onset)

    # Compute the ITC and the permutation entropy for each number of deleted nulls
    ITCs = np.zeros((n_chans, 1)); PEs = np.zeros((n_chans, 1))
    for n_null in range(n_chans):
        raw_stim_SASS = SASS(raw_stim.copy(), raw_no_stim.copy(), filter_type=None, n_nulls=n_null)

        # Compute the phase of the target channel for the different conditions
        phase_stim_SASS = compute_phase(raw_stim_SASS._data[chidx].mean(0))

        # Compute the Inter-Trial-Coherence
        ITC, phase_diffs_stim_SASS = compute_ITC(phase_stim_SASS, stim_flicker_event_onset)
        ITCs[n_null] = ITC

        # Plot the polar histogram of mean phase differences (for visual inspection)
        fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(18, 6))
        ax = axs[0]; ax.hist(phase_diffs_no_stim); plt.title("No Stim")
        ax = axs[1]; ax.hist(phase_diffs_stim); plt.title("Stim")
        ax = axs[2]; ax.hist(phase_diffs_stim_SASS); plt.title("Stim SASS")
        plt.show()

        # Compute the permutation entropy

    # Plot the relation between deleted nulls, permutation entropy and ITC
    plt.figure()
    plt.plot(range(n_chans), ITCs)
    plt.axhline(ITC_no_stim, color="r", label="No stim"); plt.axhline(ITC_stim, color="r", label="Sstim")
    plt.xlabel("#nulls"); plt.ylabel("ITC")
    plt.show()
    print("debug")