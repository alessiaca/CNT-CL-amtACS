# Simulate the behaviour of the closed loop system

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert, filtfilt
from scipy.io import loadmat
from scipy.stats import  zscore
from main_code.utils import DFT, compute_P, find_bad_channels,\
      fir_coeffs,get_audio_events,get_audio_phases, plv, compute_phase

# Load the data used for simulation
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/validation/data/16_11_21/16_11_21/"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim/no_stim_1.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/corr/stim.vhdr', preload=True)
initial_bad_channels = loadmat(folder_path+"/exclude_idx.mat")["exclude_idx"][0]
initial_bad_channels = [raw_stim.ch_names[i] for i in initial_bad_channels - 1]

# Downsample the data to the rates used in the RL system (neglect the 250 Hz used for SASS computation)
raw_stim.resample(500)
raw_no_stim.resample(500)
sfreq = raw_stim.info['sfreq']

# Crop the data for faster debugging
#raw_stim.crop(0,30*3)
#raw_no_stim.crop(0,30*3)

# Get the events (switches of phase shift)
events_stim = mne.events_from_annotations(raw_stim)[0]
events_no_stim = mne.events_from_annotations(raw_no_stim)[0]

# Set some parameters
frequency = 6
lfreq = frequency - 1
hfreq = frequency + 1
buffer_size = sfreq * 1
buffer_SASS_size_sec = 25
buffer_SASS_size = sfreq * buffer_SASS_size_sec
buffer_SASS_step_size = sfreq * 1
coeffs = fir_coeffs(frequency, n_taps=84)
phase_shifts = [i/6 * 360 for i in range(6)]

# Get the audio channel
raw_audio_stim = raw_stim.copy().pick_channels(["tones"])   
audio_no_stim = raw_no_stim.get_data("tones").flatten()
audio_stim = raw_stim.get_data(["tones"]).flatten()
audio_stim_phases = get_audio_phases(audio_stim)
audio_no_stim_phases = get_audio_phases(audio_no_stim)

# Get the envlopes
raw_stim_env = raw_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])
raw_no_stim_env = raw_no_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])

# Drop non EEG and bad channels
print(f"Bad channels initial {initial_bad_channels}")
inital_bad_channels = ['env', 'stim', 'tones', 'sass chan'] + initial_bad_channels
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in inital_bad_channels])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in inital_bad_channels])

# Define the target channel
ch_names = raw_stim.ch_names
n_chans = len(ch_names)
chidx = ch_names.index("TP8")

# Get the data used for simulation
data_stim = raw_stim._data
n_samps = data_stim.shape[1]

# Create the covariance matrix of the data without stimulation (filtered around the target frequency)
no_stim_data_filt = filtfilt(coeffs, 1, raw_no_stim._data.copy())
B = np.cov(no_stim_data_filt)

# Loop over the samples and apply SASS just as in the real time system
steps = np.arange(buffer_SASS_size, n_samps)
n_steps = len(steps)
envelope = np.zeros((n_samps, 1))
phase_shift = 0
for i, i_samp in enumerate(steps):

    # Print status update
    if i != 0 and i in np.round(np.linspace(0, n_steps, 10)): print(f"Done with {np.round((i / n_steps) * 100, 0)}%")

    # Check current phase shift
    if i_samp in events_stim[:,0]:
        i_phase_shift = events_stim[np.where(events_stim[:,0] == i_samp)[0],2]
        phase_shift = phase_shifts[int(i_phase_shift-1)]

    # Update the SASS Matrix
    if i_samp % buffer_SASS_step_size == 0:
        # Get the buffer from the last 25 sec
        buffer_long = data_stim[:, int(i_samp - buffer_SASS_size):int(i_samp)].copy()
        # Find bad channels and delete them
        vars = np.var(buffer_long, 1)
        idx_goods = np.abs(zscore(vars)) < 1.645
        buffer_long = buffer_long[idx_goods, :]
        # Detrend the buffer
        buffer_long -= buffer_long.mean(1)[:, np.newaxis]
        # Filter the buffer
        buffer_long_filt = filtfilt(coeffs, 1, buffer_long.copy())
        # Compute the covariance matrix
        A = np.cov(buffer_long_filt)
        # Compute the SASS Matrix
        tmp_B = B[np.ix_(idx_goods,idx_goods)]
        P_SASS_small, n_nulls = compute_P(A, tmp_B)
        # Embed in larger matrix (as channels were excluded)  ?? Correct ??
        P_SASS = np.zeros((B.shape))
        P_SASS[np.ix_(idx_goods,idx_goods)] = P_SASS_small

    # Apply the matrix to the new sample

    # Get the buffer (1 sec)
    buffer = data_stim[:, int(i_samp - buffer_size):int(i_samp)].copy()
    # Detrend the buffer
    buffer -= buffer.mean(1)[:, np.newaxis]
    # Apply SASS
    buffer_SASS = P_SASS @ buffer
    # Compute the target channel
    target_chan = buffer_SASS[chidx]
    # Filter the target channel (with a short filter)
    target_chan_filt = filtfilt(coeffs, 1, target_chan.copy())
    # Compute the phase using the DFT at the target frequency using the last cycle
    amplitude, phase = DFT(target_chan_filt[-int((sfreq / frequency) * 1):], frequency, sfreq)
    # Transform the phase into an envelope  ?? Not correct??
    envelope[int(i_samp)] = np.cos(2 * np.pi * frequency + phase - phase_shift)
    # Low-pass-filter of the envelope??

# Create a new raw object using the simulated envelope
raw_stim_env_sim = raw_stim_env.copy()
raw_stim_env_sim._data = envelope.T

# Crop the data as the first 25 sec cannot be used
raw_stim_env_sim.crop(tmin=buffer_SASS_size_sec)
raw_stim_env.crop(tmin=buffer_SASS_size_sec)
raw_audio_stim.crop(tmin=buffer_SASS_size_sec)
# Get the events (switches of phase shift)
events_stim = mne.events_from_annotations(raw_stim_env)[0]
# Include only events that are complete (20 sec)
events_stim = events_stim[events_stim[:,0] + sfreq * 20 < raw_stim_env.last_samp]
event_ids = np.unique(events_stim[:,2])

# Analyze the results: Plot the phase difference between audio and envelope for real and simulated envelope
fig_rt = plt.figure(1)
fig_sim = plt.figure(2)
for i,event_id in enumerate(event_ids):
    # Create epochs
    epochs_env = mne.Epochs(raw_stim_env, events_stim, event_id=event_id, tmin= 0, tmax=20, detrend=0, baseline=None,preload=True)
    epochs_env_sim = mne.Epochs(raw_stim_env_sim, events_stim, event_id=event_id, tmin=0, tmax=20, detrend=0, baseline=None,preload=True, reject=None)
    epochs_audio = mne.Epochs(raw_audio_stim, events_stim, event_id=event_id, tmin=0, tmax=20, detrend=0, baseline=None,preload=True)
    # Calculate the phases of the audio in the epochs
    phases_audio = get_audio_phases(epochs_audio._data.flatten())
    # Calculate the phases of the envelopes
    phases_env = compute_phase(epochs_env._data.flatten())
    phases_env_sim = compute_phase(epochs_env_sim._data.flatten())
    # Plot the phase differences
    plt.figure(1)
    plt.subplot(2, 3, i + 1, projection='polar')
    plt.hist(phases_env - phases_audio, color="green", alpha=0.5, label="RT")
    plt.title(event_id)
    plt.figure(2)
    plt.subplot(2, 3, i + 1, projection='polar')
    plt.hist(phases_env_sim - phases_audio, color="red", alpha=0.5, label="Sim")
    plt.title(event_id)

plt.figure(1)
plt.suptitle("Real Time")
plt.figure(2)
plt.suptitle("Simulation")

plt.show()
print("h")


