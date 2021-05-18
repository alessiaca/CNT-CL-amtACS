import numpy as np
import mne
from mne.time_frequency import psd_welch, psd_array_welch,tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import tkinter as tk
from tkinter import filedialog
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit
from utils import get_target_chan, compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
from scipy.signal import firwin, filtfilt

# Test whether components are rejected if no stimulation occurs
# Set parameters

# Choose the dataset
#root = tk.Tk()
#root.withdraw()
#folder_path = filedialog.askdirectory(initialdir = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/WM-Task/data")
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/WM-Task/data/pilot/1"

# Load data
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
# Split the dataset into two
ch_names_array = np.array(raw_no_stim.ch_names)
initial_bads = ["Cz", "CPz", "CP1", "CP2", "C1", "C2","Fz"]
frequency = 28
lfreq = frequency-5
hfreq = frequency+5
sfreq = raw_no_stim.info['sfreq']
inner_pick = "Pz"
outer_picks = ["O1","O2","P3","P4"]

# Drop non EEG channels and those based on the non_stim dataset
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])

raw_no_stim_1 = raw_no_stim.copy()
raw_no_stim_2 = raw_no_stim.copy()
cut = int(raw_no_stim_1._data.shape[1]/2)
raw_no_stim_1._data = raw_no_stim_1._data[:,:cut]
raw_no_stim_2._data = raw_no_stim_2._data[:,cut:]

plt.figure()
fmin = 1; fmax = 50
target_chan_no_stim_1 = get_target_chan(raw_no_stim,inner_pick,outer_picks)
psds, freqs = psd_array_welch(target_chan_no_stim_1,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim 1")
target_chan_stim= get_target_chan(raw_stim,inner_pick,outer_picks)
psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim 2")

# Apply SASS and plot the power spectrum of the target channel
ch_names = raw_no_stim.ch_names
chidx = [ch_names.index(ch) for ch in outer_picks]
P_laplace = np.zeros((1,len(ch_names)))
P_laplace[0,chidx[0]] = 1
P_laplace[0,chidx[1]] = 1
P_laplace[0,chidx[2]] = 1
P_laplace[0,chidx[3]] = 1
P_laplace[0,ch_names.index(inner_pick)] = -4
coeffs = fir_coeffs(frequency, n_taps=96)
raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace,filter_type='long',filter_coeffs=coeffs)
target_chan_stim_SASS = get_target_chan(raw_stim,inner_pick,outer_picks)
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
# Delete the events of 1 that are not showing the delay (but after the delay)
cond_one = np.where(events[:,2] == 1)
smaller_200 = np.where(np.diff(events[:,0]) < 200)[0]
delete = np.intersect1d(smaller_200,cond_one)
events = np.delete(events, delete,axis=0)
#and np.diff(events[:,0]))
conditions = ['no stim','in phase', 'anti phase', 'open loop']
event_ids = [1,2,3,4]
amplitudes = np.zeros((len(event_ids),1))
stds = np.zeros((len(event_ids),1))
plt.figure()
for i,event_id in enumerate(event_ids):
      epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=0, tmax=3, detrend=0, baseline=None,preload=True)
      epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=0, tmax=3, detrend=0, baseline=None,preload=True)
      env = epochs_env._data.squeeze()

      # Get the target channel and its hilbert transform
      target_chan = get_target_chan(epochs, inner_pick, outer_picks)
      n_samps = target_chan.shape[1]
      fft_len = next_fast_len(n_samps)
      hil = hilbert(target_chan,fft_len)[:,:n_samps]

      psds = tfr_array_morlet(target_chan[:, np.newaxis, :], sfreq, [frequency], n_cycles=4, output='power')

      # Get the amplitude averaged over epochs
      #amplitudes[i] = np.median(np.abs(hil))
      amplitudes[i] = np.median(psds.flatten())
      #stds[i] = np.std(np.median(np.abs(hil),-1))
      stds[i] = np.std(psds.flatten())

      # Plot the phase locking of the target_channel with the envelope
      phase_diff = compute_phase_diff(target_chan, env)
      ax = plt.subplot(1, 4, i + 1, projection='polar')
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.hist(phase_diff.flatten(), color="blue", alpha=0.5)
      plt.title(f"{conditions[i]} °")

plt.show()

# Plot the amplitudes as bars
amplitudes = amplitudes.flatten()
stds = stds.flatten()
x = np.arange(len(conditions))  # the label locations
width = 0.35  # the width of the bars
plt.errorbar(x,amplitudes,stds,color="red")
plt.bar(x,amplitudes,tick_label=conditions)
plt.show()



