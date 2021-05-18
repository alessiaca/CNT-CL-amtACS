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
from utils import compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
import seaborn as sns

def get_target_chan(raw,inner_pick,outer_picks):
    """Computes the laplace over 3 input electrodes"""
    n_outer = len(outer_picks)
    if len(raw._data.shape) == 2:
        target_chan = (np.sum(raw.get_data(outer_picks),axis = 0) - n_outer * raw.get_data(inner_pick)).flatten()
    else:
        target_chan = np.sum(raw.get_data(outer_picks),axis = 1) - n_outer * np.squeeze(raw.get_data(inner_pick))
    return target_chan

mne.set_log_level('CRITICAL')

# Plot the amplitude modulation for each participant --> Choose one that looks nice for the proposal
# I left the phase difference polar plots etc. to check if the artifact was removed successfully

n_pars = 12

for i_par in [8]: #np.arange(1,n_pars+1):

      folder_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\eeg\\data\\{i_par}"

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

      # New spatial filter for the last 4 participants
      if i_par > 8:
            outer_picks = ["Afz","F3","F4","Cz"]
      else:
            outer_picks = ["F3", "F4"]
      inner_pick = "Fz"

      # Print the parameters for visual check
      print(f"Bads: {initial_bads} based on the no stimulation data")
      print(f"Frequency: {frequency}")

      # Get the envelope
      raw_no_stim_env = raw_no_stim.copy().pick_channels(["env"])
      raw_stim_env = raw_stim.copy().pick_channels(["env"])

      # Drop non EEG channels and those based on the non_stim dataset
      raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
      raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])\

      # Identify bad channels from the stim dataset
      bads = find_bad_channels(raw_stim,use_zscore=False)
      print(f"{bads} would be dropped in addition based on the calibration data")
      # Comment this out if you don't want to drop them
      raw_stim.drop_channels(bads)
      raw_no_stim.drop_channels(bads)
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
      target_chan_no_stim = get_target_chan(raw_no_stim,inner_pick,outer_picks)
      psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="No Stim")
      target_chan_stim = get_target_chan(raw_stim,inner_pick,outer_picks)
      psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="Stim")

      # Apply SASS and plot the power spectrum of the target channel
      ch_names = raw_stim.ch_names
      ix_outer = [ch_names.index(x) for x in outer_picks]
      ix_inner = ch_names.index(inner_pick)
      P_laplace = np.zeros((1, len(ch_names)))
      P_laplace[0, ix_inner] = -4
      for ix in ix_outer:
            P_laplace[0, ix] = 1

      raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace)
      #raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq,hfreq,window_size=int(25*sfreq),P_laplace=P_laplace)

      target_chan_stim_SASS = get_target_chan(raw_stim,inner_pick,outer_picks)
      psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="Stim after SASS")
      plt.xlabel("Frequency in Hz")
      plt.ylabel("Power Spectral Density in dB")
      plt.title("Target channel")
      plt.legend()

      # Filter the data around the target frequency
      raw_stim.filter(lfreq,hfreq)

      # Cut the data into epochs, get the mean amplitude for each condition and plot the phase locking in that condition
      events = mne.events_from_annotations(raw_stim)[0]
      conditions = ['0', '45', '90', '135', '180', '225','270','315']
      event_ids = [2,3,4,5,6,7,8,9]
      amplitudes_hil = {event_id:[] for event_id in event_ids}
      amplitudes_morlet = {event_id:[] for event_id in event_ids}

      plt.figure()
      for i,event_id in enumerate(event_ids):

            print('Number of events for id {:d} found: {:d}'.format(event_id,sum(events[:,2]==event_id)))
            epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=0, tmax=4, detrend=0, baseline=None,preload=True)
            epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=0, tmax=4, detrend=0, baseline=None,preload=True)
            env = epochs_env._data.squeeze()

            # Get the target channel and its hilbert transform
            target_chan = get_target_chan(epochs, inner_pick,outer_picks)

            # Change here if you only want one frequency
            psds_morlet = tfr_array_morlet(target_chan[:,np.newaxis,:], sfreq, [frequency-1, frequency, frequency+1], n_cycles=4, output='complex')

            n_samps = target_chan.shape[1]
            fft_len = next_fast_len(n_samps)
            hil = hilbert(target_chan,fft_len)[:,:n_samps]

            # Get the amplitude averaged over epochs
            amplitudes_hil[event_id].extend(np.abs(hil).flatten())
            amplitudes_morlet[event_id].extend(np.mean(np.abs(psds_morlet),2).flatten())

            # Plot the phase locking of the target_channel with the envelope
            phase_diff = compute_phase_diff(target_chan, env)
            ax = plt.subplot(2,4,i+1, projection='polar')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.hist(phase_diff.flatten(),color="red",alpha=0.5)
            plt.title(f"{conditions[i]} °")

      amplitudes = amplitudes_morlet
      xs = []
      ys = []
      for condition,event_id in zip(conditions,event_ids):
            xs.extend([condition]*len(amplitudes[event_id]))
            ys.extend(amplitudes[event_id])
      plt.figure()
      ax = sns.barplot(xs,ys,order=conditions,color="grey",alpha=.5)
      with plt.rc_context({'lines.linewidth': 2.5}):
            g = sns.pointplot(xs,ys,order=conditions,dodeg=True,color="red",plot_kws=dict(alpha=0.5),markers ="",ci=None)
            plt.setp(g.collections, alpha=.75,)  # for the markers
            plt.setp(g.lines, alpha=.75)
      ax.set_yticklabels([])
      ax.set_yticks([])
      ax.set_xticklabels([cond + " °" for cond in conditions])
      sns.despine()
      plt.title(f"Participant {i_par}")
      plt.xlabel("Phase difference")
      plt.ylabel("Amplitude")
      plt.show()