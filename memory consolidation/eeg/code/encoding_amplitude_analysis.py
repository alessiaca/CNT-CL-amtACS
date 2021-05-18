import mne 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from os import listdir
from scipy.io import loadmat
from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
import seaborn as sns
from pandas import DataFrame
from scipy.stats import ttest_ind
from utils import get_target_chan, compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
mne.set_log_level('CRITICAL')
from matplotlib import ticker


sns.set_context('talk')
mne.set_log_level('CRITICAL')


base_path = '/Users/asmitanarang/Desktop/Debugging_folder/analysis_scripts/data'
# names = listdir(base_path)
paths = [base_path+'/'+x for x in names]
conditions_codes = [1,2,3]
all_participants_amplitude_morlet = {key:[] for key in conditions_codes}
all_participants_amplitude_hilb = {key:[] for key in conditions_codes}
conditions = ['no stim','in phase', 'open loop']
names = ['1','2','3','4','5','6','7','8']

for idx, path,name in zip(range(len(names)),paths,names):
    
    # Load data
    raw_no_stim = mne.io.read_raw_brainvision(path + '/no_stim.vhdr', preload=True)
    if name =='2':
        data_paths = [path+'/encoding1.vhdr',path+'/encoding2.vhdr']
        raw_stim =  mne.io.concatenate_raws([mne.io.read_raw_brainvision(path,preload=True) for path in data_paths])
    else:
        raw_stim = mne.io.read_raw_brainvision(path + '/encoding.vhdr', preload=True)

    ch_names_array = np.array(raw_stim.ch_names)
    initial_bads = loadmat(path+ "/exclude_idx.mat")["exclude_idx"][0]
    initial_bads = ch_names_array[initial_bads - 1]
    chidx = loadmat(path+ "/chidx.mat")["chidx"][0]
    ch_picks = ch_names_array[chidx - 1]
    frequency = loadmat(path+ "/frequency.mat")["frequency"][0][0]
    lfreq = frequency-1
    hfreq = frequency+1
    sfreq = raw_no_stim.info['sfreq']


    # drop channels
    raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
    raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])

    # z-score
    # Identify bad channels from the stim dataset
    bads = find_bad_channels(raw_stim,use_zscore=True)
    # print(f"{bads} would be dropped")
    # Comment this out if you don't want to drop them
    raw_stim.drop_channels(bads)
    raw_no_stim.drop_channels(bads)
    ch_names = raw_stim.ch_names

    # SASS
    
    ch_names = raw_stim.ch_names
    chidx = [ch_names.index(ch) for ch in ch_picks]
    P_laplace = np.zeros((1,len(ch_names)))
    P_laplace[0,chidx[0]] = 1
    P_laplace[0,chidx[1]] = -2
    P_laplace[0,chidx[2]] = 1
    raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace)


    # filter data
    raw_no_stim.filter(lfreq,hfreq)
    raw_stim.filter(lfreq,hfreq)

    # events extraction
    events = mne.events_from_annotations(raw_stim)[0]
    conditions = ['no stim','in phase', 'open loop']
    event_ids = [1,2,3]
    bins_amplitude = 100
    amplitudes = np.zeros((len(event_ids),bins_amplitude-1))
    amplitude_mean = np.zeros((3,1))
    amplitude_std = np.zeros((3,1))
    amplitudes_hilb = np.zeros((len(event_ids),bins_amplitude-1))

    for i,event_id in enumerate(conditions_codes):
        epochs = mne.Epochs(raw_stim, events, event_id=int(event_id), tmin=25, tmax=60*5, detrend=0, baseline=None,preload=True)
        
      # Get the target channel and its hilbert transform
        target_chan = get_target_chan(epochs, ch_picks)
        if target_chan.ndim==1:
            n_samps = target_chan.shape[0]
        else:
            n_samps = target_chan.shape[1]
        fft_len = next_fast_len(n_samps)
        if target_chan.ndim==1:
            hil = hilbert(target_chan,fft_len)[:n_samps]
        else:
            hil = hilbert(target_chan,fft_len)[:,:n_samps]
        
        if target_chan.ndim==1:
            psds = tfr_array_morlet(target_chan[np.newaxis, np.newaxis, :], sfreq, [frequency-1,frequency,frequency+1], n_cycles=4, output='power')
            psds = np.median(psds,axis=2)
            psds = psds.flatten()
        else:
            psds = tfr_array_morlet(target_chan[:, np.newaxis, :], sfreq, [frequency-1,frequency,frequency+1], n_cycles=4, output='power')
            psds = np.median(psds,axis=2)
            psds = psds.flatten()
      # Get the amplitude averaged over epochs
      
        # all_participants_amplitude_hilb[event_id].append(np.mean(np.abs(hil)))
        bins = np.linspace(0, len(psds), bins_amplitude)
        amplitudes[i, :] = [np.median(psds[int(bins[j]):int(bins[j+1])]) for j in np.arange(len(bins)-1)]
        hil = hil.flatten()
        amplitudes_hilb[i, :] = [np.median(hil[int(bins[j]):int(bins[j+1])]) for j in np.arange(len(bins)-1)]
        # all_participants_amplitude_morlet[event_id].append(np.median((psds.flatten())))
        all_participants_amplitude_morlet[event_id].append(np.median(amplitudes[i,:]))
        amplitude_mean[i] = np.median(np.abs(amplitudes[i,:]))
        # amplitude_mean[i] = np.median(amplitudes[i,:])
        # amplitude_std[i] = np.std(np.abs(hil))
   
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.subplot(2,4,idx+1)
    x = np.arange(len(conditions))
    plt.ticklabel_format(style='scientific',useMathText=True, axis='y',useOffset=True,scilimits=(0,0))
    plt.tick_params(axis='y', bottom = False, top = False)
    sns.barplot(data = DataFrame(amplitude_mean.reshape(-1,len(amplitude_mean))))
    plt.xticks(np.arange(3),['no stim', ' in phase', 'open loop'])
    plt.locator_params(axis = 'y',nbins=4)
    plt.tick_params(axis='x', bottom = False, top = False)
    plt.title('P'+name , fontdict={'fontsize': 10})
    

plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.show()
        

data1 = DataFrame(all_participants_amplitude_hilb)
data2 = DataFrame(all_participants_amplitude_morlet)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
linewidth=5
labelpad=10
sns.boxplot(data=data1)
plt.xticks(np.arange(3),['no stimulation', ' in phase', 'open loop'])
plt.ylabel('amplitude')

plt.figure()
sns.boxplot(data=data2)         
c = "\u00b0"
plt.xticks(np.arange(3),['no stimulation', ' in phase', 'open loop'])
plt.ylabel('amplitude')
plt.show()




   



