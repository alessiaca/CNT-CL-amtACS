import numpy as np
import mne
from mne.time_frequency import psd_welch, psd_array_welch, tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert,morlet
import tkinter as tk
from tkinter import filedialog
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit
from utils import get_target_chan, compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
import seaborn as sns

mne.set_log_level('CRITICAL')

# Set parameters

# Choose the dataset
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation")

# Load data
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/calib_stim.vhdr', preload=True)
initial_bads = loadmat(folder_path+ "/exclude_idx.mat")["exclude_idx"][0]
ch_names_array = np.array(raw_stim.ch_names)
initial_bads = ch_names_array[initial_bads - 1]
frequency = loadmat(folder_path+ "/frequency.mat")["frequency"][0][0]
lfreq = frequency-1
hfreq = frequency+1
sfreq = raw_no_stim.info['sfreq']

outer_picks = ["Afz","F3","F4","Cz"]
inner_pick = "Fz"

# Print the parameters for visual check
print(f"Bads: {initial_bads} based on the no stimulation data")
print(f"Outer channels: {outer_picks}")
print(f"Inner channel: {inner_pick}")
print(f"Frequency: {frequency}")

# Get the envelope
raw_no_stim_env = raw_no_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])
raw_stim_env = raw_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])

# Drop non EEG channels and those based on the non_stim dataset
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])\

# for ch_name,ch_vals in zip(raw_stim.ch_names,raw_stim._data):
#       plt.plot(raw_stim.times,ch_vals)
#       plt.title(ch_name)
#       plt.show()

# Identify bad channels from the stim dataset
bads = find_bad_channels(raw_stim,use_zscore=False)
#bads = initial_bads
print(f"{bads} would be dropped in addition based on the calibration data")
# Comment this out if you don't want to drop them
#raw_stim.drop_channels(bads)
#raw_no_stim.drop_channels(bads)
ch_names = raw_stim.ch_names

# Plot the power of the envelopes
fmin = 1
fmax = 30
plt.figure()
plt.subplot(1,2,1)
psds, freqs = psd_welch(raw_no_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="No Stim")
psds, freqs = psd_welch(raw_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="Stim")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Envelope")
plt.legend()

# Plot the power spectrum of the target channel
plt.subplot(1,2,2)
target_chan_no_stim = get_target_chan(raw_no_stim,outer_picks,inner_pick)
psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim")
target_chan_stim = get_target_chan(raw_stim,outer_picks,inner_pick)
psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim")

# Apply SASS and plot the power spectrum of the target channel

ch_names = raw_stim.ch_names
ix_outer = [ch_names.index(x) for x in outer_picks]
ix_inner = ch_names.index(inner_pick)
P_laplace = np.zeros((1,len(ch_names)))
P_laplace[0,ix_outer[0]] = 1
P_laplace[0,ix_outer[1]] = 1
P_laplace[0,ix_outer[2]] = 1
P_laplace[0,ix_outer[3]] = 1
P_laplace[0,ix_inner] = -4


raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace)
#raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq,hfreq,window_size=int(25*sfreq),P_laplace=P_laplace)

target_chan_stim_SASS = get_target_chan(raw_stim,outer_picks,inner_pick)
psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim after SASS")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
plt.legend()
plt.show()

# Filter the data around the target frequency
raw_stim.filter(lfreq,hfreq)

# Cut the data into epochs, get the mean amplitude for each condition and plot the phase locking in that condition
events = mne.events_from_annotations(raw_stim)[0]
conditions = ['0', '45', '90', '135', '180', '225','270','315']
event_ids = [2,3,4,5,6,7,8,9]
amplitudes_hil = np.zeros((len(event_ids),1))
stds_hil = np.zeros((len(event_ids),1))
amplitudes_welch = np.zeros((len(event_ids),1))
stds_welch = np.zeros((len(event_ids),1))
amplitudes_morlet = np.zeros((len(event_ids),1))
stds_morlet = np.zeros((len(event_ids),1))
plt.figure()
for i,event_id in enumerate(event_ids):

      print('Number of events for id {:d} found: {:d}'.format(event_id,sum(events[:,2]==event_id)))
      epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=0, tmax=4, detrend=0, baseline=None,preload=True)
      epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=0, tmax=4, detrend=0, baseline=None,preload=True)
      env = epochs_env._data.squeeze()

      # Get the target channel and its hilbert transform
      target_chan = get_target_chan(epochs,outer_picks,inner_pick)
      psds_welch, freqs = psd_array_welch(target_chan, fmin=frequency-0.2, fmax=frequency+0.2, sfreq=sfreq, n_fft=int(sfreq * 2))
      psds_morlet = tfr_array_morlet(target_chan[:,np.newaxis,:], sfreq, [frequency], n_cycles=4, output='power')
      n_samps = target_chan.shape[1]
      fft_len = next_fast_len(n_samps)
      hil = hilbert(target_chan,fft_len)[:,:n_samps]

      # Get the amplitude averaged over epochs
      amplitudes_hil[i] = np.mean(np.abs(hil))
      amplitudes_welch[i] = np.median(psds_welch.flatten())
      amplitudes_morlet[i] = np.mean(psds_morlet.flatten())
      stds_hil[i] = np.std(np.median(np.abs(hil),-1))
      stds_welch[i] = np.std(psds_welch.flatten())
      stds_morlet[i] = np.std(np.median(psds_morlet,-1).flatten())

      # Plot the phase locking of the target_channel with the envelope
      phase_diff = compute_phase_diff(target_chan, env)
      plt.subplot(2,4,i+1, projection='polar')
      plt.hist(phase_diff.flatten())
      plt.title(conditions[i])

plt.show()
stds_all = [stds_hil,stds_welch,stds_morlet]
amplitudes_all = [amplitudes_hil,amplitudes_welch,amplitudes_morlet]
measures = ["Hilbert","Welch","Morlet"]
plt.figure()
for i in range(3):
      plt.subplot(2,2,i+1)

      # Plot the amplitudes as bars
      amplitudes = amplitudes_all[i].flatten()
      x = np.arange(len(conditions))  # the label locations
      width = 0.35  # the width of the bars
      plt.bar(x,amplitudes,tick_label=conditions)
      plt.errorbar(x,amplitudes,stds_all[i],color="red")
      plt.title(measures[i])

      # Fit a sine wave to the bars
      x = np.linspace(0,7, num=8)
      param = curve_fit(sine_func, x, amplitudes)[0]
      y = sine_func(x, param[0], param[1], param[2])
      plt.plot(x,y,color="black")

      # ADD HERE ONLY PEAK_TROUGH WITH 180 DEGREE DIFFERENCE
      # Get the maximum and minimum, print and plot it
      print(measures[i])
      print("in-phase (peak): " + conditions[np.argmax(y)])
      print("anti-phase (trough): " + conditions[np.argmin(y)])
      plt.plot(x[np.argmax(y)], np.max(y), 'o', color='red', label="Peak");
      plt.plot(x[np.argmin(y)], np.min(y), 'o', color='red', label="Trough");
      plt.legend()
plt.show()

print("Which in-phase to choose?")
in_phase = input("> ")
in_phase = int(in_phase)
print("Which anti-phase to choose?")
anti_phase = input("> ")
anti_phase = int(anti_phase)

# Save the determined phase differences
data_dict = {"in_phase_diff": float(in_phase)}
savemat(folder_path + "/in_phase_diff.mat", data_dict)
data_dict = {"anti_phase_diff": float(anti_phase)}
savemat(folder_path + "/anti_phase_diff.mat", data_dict)

final_bads = np.unique(np.concatenate((bads,initial_bads)))
exclude_idx = [list(ch_names_array).index(ch) + 1 for ch in final_bads]
data_dict = {"exclude_idx": exclude_idx}
#savemat(folder_path + "/exclude_idx.mat", data_dict)


