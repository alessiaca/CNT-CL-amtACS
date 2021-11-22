# Simulate the behaviour of the closed loop system

import numpy as np
import mne
from scipy.signal import hilbert, filtfilt
from main_code.utils import DFT, compute_P, find_bad_channels,\
      fir_coeffs,get_audio_events,get_audio_phases, plv

# Load the data used for simulation
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/validation/data/16_11_21/16_11_21/"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim/no_stim_1.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/corr/stim.vhdr', preload=True)
sfreq = raw_no_stim.info['sfreq']

# Get the events (switches of phase shift)
events_stim = mne.events_from_annotations(raw_stim)[0]
events_no_stim = mne.events_from_annotations(raw_no_stim)[0]

# Set some parameters
frequency = 6
lfreq = frequency - 1; hfreq = frequency + 1

buffer_size = sfreq * 1
buffer_SASS_size = sfreq * 25
buffer_SASS_step_size = sfreq * 1
coeffs = fir_coeffs(frequency, n_taps=84)
coeffs_SASS = fir_coeffs(frequency, n_taps=42)

# Get the audio channel
audio_no_stim = raw_no_stim.get_data("tones").flatten()
audio_stim = raw_stim.get_data(["tones"]).flatten()
audio_stim_events = get_audio_events(audio_stim,sfreq)
audio_stim_phases = get_audio_phases(audio_stim_events, len(audio_stim), sfreq)
audio_no_stim_events = get_audio_events(audio_no_stim, sfreq)
audio_no_stim_phases = get_audio_phases(audio_no_stim_events, len(audio_no_stim) ,sfreq)

# Drop non EEG and bad channels
inital_bad_channels = ['env', 'stim', 'tones', 'sass chan']
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in inital_bad_channels])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in inital_bad_channels])
bad_channels = find_bad_channels(raw_stim)
raw_no_stim.drop_channels(bad_channels)
raw_stim.drop_channels(bad_channels)

# Define the target channel
ch_names = raw_stim.ch_names
n_chans = len(ch_names)
chidx = ch_names.index("TP8")

# Get the data used for simulation
data_stim = raw_stim._data
n_samps = data_stim.shape[1]

# Create the covariance matrix of the data without stimulation (filtered around the target frequency)
no_stim_data = filtfilt(coeffs_SASS, 1, raw_no_stim._data.copy())
B = np.cov(no_stim_data)

# Loop over the samples and apply SASS just as in the real time system
steps = np.arange(buffer_SASS_size, n_samps)
envelope = np.zeros((n_samps, 1))
for i, i_samp in enumerate(steps):

    # Update the SASS Matrix
    if i_samp % buffer_SASS_step_size == 0:
        # Get the buffer
        buffer_long = data_stim[:, int(i_samp - buffer_SASS_size):int(i_samp)].copy()
        # Filter the buffer (with a longer filter)
        buffer_long_filt = filtfilt(coeffs_SASS, 1, buffer_long.copy())
        # Compute the covariance matrix
        A = np.cov(buffer_long_filt)
        # Compute the SASS Matrix
        P_SASS, n_nulls = compute_P(A, B)

    # Apply the matrix to the new sample

    # Get the buffer (1 sec)
    buffer = data_stim[:, int(i_samp - buffer_size):int(i_samp)].copy()
    # Detrend the buffer
    buffer -= buffer.mean(1)[:, np.newaxis]
    # Apply SASS
    buffer_SASS = P_SASS @ buffer
    # Compute the target channel
    target_chan = buffer_SASS(chidx)
    # Filter the target channel (with a short filter)
    target_chan_filt = filtfilt(coeffs, 1, target_chan.copy())
    # Compute the phase using the DFT at the target frequency using the last 3 cycles
    amplitude, phase = DFT(target_chan_filt[-int((sfreq / frequency) * 3):], frequency, sfreq)
    # Transform the phase into an envelope
    envelope[i] = np.cos(2 * np.pi * frequency * np.linspace(0, 1, 500) + phase)

# Analyze the results


