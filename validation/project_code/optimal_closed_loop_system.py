# Find the optimal SASS performance on the basis of the plv between audio and channel

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert, filtfilt
from main_code.utils import compute_P,fir_coeffs,get_audio_events,get_audio_phases, plv, find_bad_channels

# Load the data used for simulation
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/validation/data/16_11_21/16_11_21/"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim/no_stim_1.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/corr/stim.vhdr', preload=True)
sfreq = raw_no_stim.info['sfreq']

raw_stim.crop(0,60*3)
raw_no_stim.crop(0,60*3)

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
audio_no_stim_phases = get_audio_phases(audio_no_stim_events, len(audio_no_stim), sfreq)

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
steps = np.arange(buffer_SASS_size, n_samps,buffer_SASS_step_size)
n_steps = len(steps)
plvs_stim_final = np.zeros((n_steps,1))
n_nulls_final = np.zeros((n_steps,1))
for i, i_samp in enumerate(steps):

    # Print status update
    if i != 0 and i in np.round(np.linspace(0,n_steps,10)): print(f"Done with {np.round((i/n_steps)*100,2)}%")

    # Update the SASS Matrix
    # Get the buffer
    buffer_long = data_stim[:, int(i_samp - buffer_SASS_size):int(i_samp)].copy()
    # Filter the buffer (with a longer filter)
    buffer_long_filt = filtfilt(coeffs_SASS, 1, buffer_long.copy())
    # Compute the covariance matrix
    A = np.cov(buffer_long_filt)
    # Compute the SASS Matrix for all n_nulls
    P_SASS_all = [compute_P(A, B, n_null) for n_null in range(n_chans)]

    # Apply the matrix for the next second (update rate of SASS)

    # Get the buffer (1 sec)
    buffer = data_stim[:, int(i_samp - buffer_size):int(i_samp)].copy()
    # Detrend the buffer
    buffer -= buffer.mean(1)[:, np.newaxis]
    plv_all_SASS = np.zeros((n_chans,1))
    for P_SASS,n_nulls in P_SASS_all:
        # Apply SASS
        buffer_SASS = P_SASS @ buffer
        # Compute the target channel
        target_chan = buffer_SASS[chidx]
        # Filter the target channel (with a short filter)
        target_chan_filt = filtfilt(coeffs, 1, target_chan)

        # Compute the phase locking of the target chan with the audio channel
        phases = np.angle(hilbert(target_chan_filt))
        plv_all_SASS[n_nulls] = plv(phases - audio_stim_phases[int(i_samp - buffer_size):int(i_samp)])

    plvs_stim_final[i] = np.max(plv_all_SASS)
    n_nulls_final[i] = np.argmax(plv_all_SASS)

# What is the optimal plv? Without stimulation
data_no_stim = raw_no_stim._data
plvs_no_stim_final = np.zeros((len(steps),1))
for i, i_samp in enumerate(steps):
    buffer = data_no_stim[:, int(i_samp - buffer_size):int(i_samp)].copy()
    # Compute the target channel
    target_chan = buffer[chidx]
    # Filter the target channel (with a short filter)
    target_chan_filt = filtfilt(coeffs, 1, target_chan)
    # Compute the phase locking of the target chan with the audio channel
    phases = np.angle(hilbert(target_chan_filt))
    plvs_no_stim_final[i] = plv(phases - audio_no_stim_phases[int(i_samp - buffer_size):int(i_samp)])

plt.figure()
plt.hist(plvs_no_stim_final, label="No stim")
plt.hist(plvs_stim_final, label="Stim")
plt.legend()
plt.show()
