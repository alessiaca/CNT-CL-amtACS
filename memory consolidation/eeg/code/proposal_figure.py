import numpy as np
import mne
from mne.time_frequency import psd_welch, psd_array_welch, tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert,morlet
import tkinter as tk
import pandas
from tkinter import filedialog
from random import choices
from scipy.io import savemat, loadmat
import seaborn as sns
from mne.filter import  filter_data
from utils import get_target_chan, compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
from scipy.stats import norm
Z = norm.ppf

# Summarize the behavioural and neurophysiological results of one partiicpant
# - Power spectrum fo the target channel
# - Calibration performance
# - Recognition accuracy on day 1 and 2
mne.set_log_level('CRITICAL')

# Set parameters
n_par = 8
# Choose the dataset
for i_par in [1,3,4,5,6,7,8]:
      #root = .Tk()
      #root.withdraw()
      #folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation")
      folder_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\eeg\\data\\{i_par}"

      # Load data
      raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
      raw_stim = mne.io.read_raw_brainvision(folder_path + '/encoding.vhdr', preload=True)
      in_phase_diff = loadmat(folder_path+ "/in_phase_diff.mat")["in_phase_diff"][0]
      initial_bads = loadmat(folder_path+ "/exclude_idx.mat")["exclude_idx"][0]
      ch_names_array = np.array(raw_stim.ch_names)
      initial_bads = ch_names_array[initial_bads - 1]
      chidx = loadmat(folder_path+ "/chidx.mat")["chidx"][0]
      ch_picks = ch_names_array[chidx - 1]
      frequency = loadmat(folder_path+ "/frequency.mat")["frequency"][0][0]
      lfreq = frequency-1
      hfreq = frequency+1
      sfreq = raw_no_stim.info['sfreq']

      # Print the parameters for visual check
      print(f"Bads: {initial_bads} based on the no stimulation data")
      print(f"Channel picks: {ch_picks}")
      print(f"Frequency: {frequency}")

      # Get the envelope
      raw_no_stim_env = raw_no_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])
      raw_stim_env = raw_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])
      raw_no_stim_env_no_filt = raw_no_stim.copy().pick_channels(["env"])
      raw_stim_env_no_filt = raw_stim.copy().pick_channels(["env"])

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
      fmin = 2
      fmax = 20
      plt.figure()
      plt.subplot(2,4,1)
      psds, freqs = psd_welch(raw_no_stim_env_no_filt, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
      plt.semilogy(freqs, psds.T, label="No Stim")
      psds, freqs = psd_welch(raw_stim_env_no_filt, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
      plt.semilogy(freqs, psds.T, label="Stim")
      plt.legend(loc=2, prop={'size': 6})
      plt.title(f"Envelope at {frequency} Hz",fontsize=8)

      # Plot the power spectrum of the target channel
      plt.subplot(2,4,2)
      target_chan_no_stim = get_target_chan(raw_no_stim,ch_picks)
      psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="No Stim")
      target_chan_stim = get_target_chan(raw_stim,ch_picks)
      psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="Stim")

      # Apply SASS and plot the power spectrum of the target channel

      ch_names = raw_stim.ch_names
      chidx = [ch_names.index(ch) for ch in ch_picks]
      P_laplace = np.zeros((1,len(ch_names)))
      P_laplace[0,chidx[0]] = 1
      P_laplace[0,chidx[1]] = -2
      P_laplace[0,chidx[2]] = 1

      raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace)
      #raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq,hfreq,window_size=int(25*sfreq),P_laplace=P_laplace)

      target_chan_stim_SASS = get_target_chan(raw_stim,ch_picks)
      psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
      plt.semilogy(freqs,psds,label="Stim after SASS")
      plt.legend(loc=2, prop={'size': 6})
      plt.title("Target channel",fontsize=8)

      # Filter the data around the target frequency
      raw_stim.filter(lfreq,hfreq)
      raw_no_stim.filter(lfreq,hfreq)

      # Plot the phase locking in the no stimulation condition
      target_chan_no_stim = filter_data(target_chan_no_stim,sfreq,frequency-1,frequency+1)
      phase_diff = compute_phase_diff(target_chan_no_stim, raw_no_stim_env._data.flatten())
      ax = plt.subplot(2,4,3, projection='polar')
      plt.hist(phase_diff,color="red",alpha=0.5)
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.title("No stim",fontsize=8)

      # Cut the data into epochs, get the mean amplitude for each condition and plot the phase locking in that condition
      events = mne.events_from_annotations(raw_stim)[0]
      conditions = ['no stim',f"In phase: {in_phase_diff} °", 'open loop']
      event_ids = [1,2,3]
      bins_amplitude = 100
      amplitudes = np.zeros((len(event_ids),bins_amplitude-1))
      amplitudes_short = np.zeros((len(event_ids), 1))
      tmin = 25
      tmax = 5.5*60
      for i,event_id in enumerate(event_ids):

            epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=tmin, tmax=tmax, detrend=0, baseline=None,preload=True)
            epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=tmin, tmax=tmax, detrend=0, baseline=None,preload=True)
            env = epochs_env._data.squeeze()

            # Get the target channel, its hilbert transform and morlet power at the target frequency
            target_chan = get_target_chan(epochs, ch_picks)
            psds_morlet = tfr_array_morlet(target_chan[:, np.newaxis, :], sfreq, [frequency], n_cycles=4,
                                           output='power')
            n_samps = target_chan.shape[1]
            fft_len = next_fast_len(n_samps)
            hil = hilbert(target_chan, fft_len)[:, :n_samps]

            # Get the amplitude from 100 time bins
            #psds_morlet = psds_morlet.flatten()
            psds_morlet = np.abs(hil.flatten())
            bins = np.linspace(0, len(psds_morlet), bins_amplitude)
            amplitudes[i, :] = [np.mean(psds_morlet[int(bins[j]):int(bins[j+1])]) for j in np.arange(len(bins)-1)]
            amplitudes_short[i] = np.mean(psds_morlet)

            # Plot the phase locking of the target_channel with the envelope
            if event_id > 1:
                  phase_diff = compute_phase_diff(target_chan, env)
                  ax = plt.subplot(2,4,i+3, projection='polar')
                  ax.set_yticklabels([])
                  ax.set_xticklabels([])
                  plt.hist(phase_diff.flatten(),color="red",alpha=0.5)
                  plt.title(conditions[i],fontsize=8)

      # Plot the amplitudes for each condition
      plt.subplot(2,4,6)
      ys = list(amplitudes[0,:])+list(amplitudes[1,:])+list(amplitudes[2,:])
      xs = ['No stim']*len(amplitudes.T)+['In phase']*len(amplitudes.T)+['Open loop']*len(amplitudes.T)
      ax = sns.boxplot(xs,ys,boxprops=dict(alpha=.5),showfliers=False)
      #ax = plt.bar([1,2,3],amplitudes_short.flatten())
      ax.set_xticklabels(ax.get_xticklabels(), size=5)
      plt.yticks(fontsize=5)
      plt.title(f"Amplitudes", fontsize=8)

      # Plot the recognition accuracy for each condition and day
      def bootstrap_data(data):
          """From the presented/not presented images samples new ones"""
          new_data = np.empty((0,data.shape[1]), int)
          for thres in [13,33]:
              data_tmp = data[(data[:,4] < thres+20) & (data[:,4] >= thres),:]
              n_trials = len(data_tmp)
              new_data_tmp = np.array(choices(data_tmp, k=n_trials))
              new_data = np.vstack((new_data,new_data_tmp))
          return np.array(new_data)


      # Compute the recognition accuracy
      n_boots = 100
      conditions_names = ["No stim", "In phase", "Open loop"]

      for day in [1,2]:
            plt.subplot(2,4,6+day)
            accuracies = np.zeros((len(conditions_names), n_boots))
            file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\behaviour\\data\\{i_par}\\{i_par}_recognition{day}.csv"
            data = pandas.read_csv(file_path)
            data = pandas.DataFrame.to_numpy(data)
            conditions = np.unique(data[:, 1])
            for cond in np.unique(conditions):

                  # Get the data from the condition
                  data_cond = data[data[:,1] == cond]

                  for j in np.arange(n_boots):

                        data_boot = bootstrap_data(data_cond)

                        # Get a first measure of the recognition accuracy (HIT - FALSE ALARMS)
                        # But neglects the bias to say yes or no
                        correct_pressed = data_boot[data_boot[:,5] == 1,6]
                        hit_rate = np.sum(correct_pressed)/len(correct_pressed)
                        fa_rate = 1 - hit_rate
                        acc = Z(hit_rate) - Z(fa_rate) # 0 % Chance level
                        accuracies[int(cond-1),j] = acc

            ys = list(accuracies[0,:]) + list(accuracies[1,:]) + list(accuracies[2,:])
            xs = ['No stim'] * len(accuracies.T) + ['In phase'] * len(accuracies.T) + ['Open loop'] * len(accuracies.T)
            ax = sns.boxplot(xs, ys, boxprops=dict(alpha=.5),showfliers=False)
            ax.set_xticklabels(ax.get_xticklabels(), size=5)
            plt.yticks(fontsize=5)
            plt.ylabel("d prime",fontsize=8)
            plt.title(f"Recognition day {day}", fontsize=8)


      plt.subplots_adjust(wspace=0.4,hspace=0.4)
      plt.suptitle(f"Participant {i_par}",fontsize=10)
      plt.show()