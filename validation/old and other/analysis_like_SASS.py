# Visualize the performance of SASS using the gamma parameter found with the bayesian optimizer

import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg
from scipy.io import savemat, loadmat
from mne.preprocessing import compute_current_source_density
from mne.channels import make_standard_montage
import matplotlib
from scipy.signal import hilbert
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, mannwhitneyu
from pycircstat.tests import watson_williams
from scipy.stats import circstd
import time
from scipy.signal import firwin, filtfilt
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch

def fir_coeffs(freq,fs=500,n_cycles=3,validate=False):
    num_taps = int((fs/freq)*n_cycles)
    num_taps |= 1
    coeffs = firwin(numtaps=num_taps,cutoff=[freq-1,freq+1],fs=fs,pass_zero='bandpass')
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

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_context('poster')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

stepsize_sec = 0.5
winsize_sec = 25

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def plv_unbiasedz(phases):
    N = phases.size
    return (1/(N-1))*((plv(phases)**2)*N-1)

def circ_mean(phases):
    return np.angle(np.exp(1j*phases).mean())

def robust_z_score(ys):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.array(modified_z_scores)

def circ_detrend(phases):
    return wrap(phases-np.angle(np.exp(1j*phases).mean()))

def wallraff_dep(A,B,alternative):
    return wilcoxon(np.abs(circ_detrend(A)),np.abs(circ_detrend(B)),alternative=alternative)[1]

def wallraff_ind(A,B,alternative):
    return mannwhitneyu(np.abs(circ_detrend(A)),np.abs(circ_detrend(B)),alternative=alternative)[1]

def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        mses.append(np.mean((np.diag(B)-np.diag(P.dot(A).dot(P.T)))**2))
    return np.argmin(mses)

def compute_P(A, B):
    eigen_values, eigen_vectors = linalg.eig(A, B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:, ix].T
    M = linalg.pinv2(D)
    n_nulls = find_n_nulls(A, B, D, M)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    return P


def sliding_SASS(raw_open, raw_no_stim, start_window, SASS_step):
    B = np.cov(raw_no_stim._data)
    # Compute the first Covariance matrix
    old_A = np.cov(raw_open._data[:, :start_window])
    P = compute_P(old_A, B)
    raw_open._data[:, :start_window] = P.dot(raw_open._data[:, :start_window])
    raw_data = raw_open._data.copy()
    # Samples at which the filter should be updated and applied
    for i in range(start_window, raw_open.n_times,SASS_step):
        new_A = np.cov(raw_data[:, i-start_window:i:2])
        # Update the covariance matrix at each sample
        P = compute_P(new_A, B)
        raw_open._data[:, i:i+SASS_step] = P.dot(raw_open._data[:, i:i+SASS_step])
    return raw_open

def SASS(raw_open, raw_no_stim):
    B = np.cov(raw_no_stim._data)
    # Compute the first Covariance matrix
    A = np.cov(raw_open._data)
    P = compute_P(A, B)
    raw_open._data = P.dot(raw_open._data)
    return raw_open

def compute_phase_dft(data):
    # Get the last 4 cycles --> 200 samples
    phases_dft = np.zeros((1,len(data)))
    for i in np.arange(200,len(data)):
        data_use = data[i-200:i]
        dft = np.sum([np.exp((-2*np.pi*1j) * (j * 10)/(len(data_use)+1)) * data_use[j] for j in np.arange(len(data_use))])
        phases_dft[:,i] = np.angle(dft)
    return phases_dft.flatten()

def compute_envelope_dft(data):
    # Get the last 4 cycles --> 200 samples
    env_dft = np.zeros((1,len(data)))
    for i in np.arange(200,len(data)):
        data_use = data[i-200:i]
        dft = np.sum([np.exp((-2*np.pi*1j) * (j * 10)/(len(data_use)+1)) * data_use[j] for j in np.arange(len(data_use))])
        phase_dft = np.angle(dft)
        env_dft[:,i] = np.sin(2 * np.pi * 10 + phase_dft)
    return env_dft.flatten()

participant = '4'
base_path = "C:/Users/alessia/Documents/Praktika/Berlin/CL-amtACS/CL-Validation/data/pilot/"
bads = ["CPz"]
#bads = np.load('bads.npy',allow_pickle=True).item()
binmethod = 10
n_trials = 200
path = base_path+participant+'/no_stim.vhdr'
raw_no_stim = mne.io.read_raw_brainvision(path,preload=True)

threshold = 3
audio_events = np.where(np.diff((np.abs(stats.zscore(raw_no_stim.copy().pick_channels(['Flicker'])._data.flatten())) > threshold).astype('int')) > 0)[0]

miniti = raw_no_stim.time_as_index(0.4)[0]
audio_events_no_stim = [audio_events[ix] for ix in range(1,audio_events.size)
                if audio_events[ix]-audio_events[ix-1] > miniti]

coeffs = fir_coeffs(10)
raw_no_stim._data = filtfilt(coeffs,1, raw_no_stim._data)

hil = raw_no_stim.copy().pick_channels(['env']).apply_hilbert(envelope=False).get_data()
#chidxs = [raw_no_stim.ch_names.index(chname) for chname in raw_no_stim.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
#data = raw_no_stim._data[chidxs].mean(0)
#hil = hilbert(data)
phases_brain = np.angle(hil).flatten()
amplitudes_brain = np.abs(hil).flatten()

t = np.arange(0,2,1/500)
phases_flicker_trial = wrap(2*np.pi*10*t)
nsamp_trial = phases_flicker_trial.size
nsamp_start = 0
phasediffs = []
plvs = []
amplitudes = []
for ev in audio_events_no_stim:
    if ev < len(raw_no_stim.times)-nsamp_trial:
        phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
        plvs.append(plv(phases_brain[ev + nsamp_start:ev + nsamp_trial] - phases_flicker_trial[nsamp_start:]))
        amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

phasediffs_no_tacs = circ_detrend(np.array(phasediffs)[:n_trials])
amplitudes_no_tacs = np.array(amplitudes)[:n_trials]
plvs_no_tacs = np.array(plvs)[:n_trials]

path = base_path+'/'+participant+'/stim_0_1.vhdr'
raw_open = mne.io.read_raw_brainvision(path,preload=True)
threshold = 3
audio_events = np.where(np.diff((np.abs(stats.zscore(raw_open.copy().pick_channels(['Flicker'])._data.flatten())) > threshold).astype('int')) > 0)[0]
miniti = raw_open.time_as_index(0.4)[0]
audio_events_open = [audio_events[ix] for ix in range(1,audio_events.size)
                if audio_events[ix]-audio_events[ix-1] > miniti]

raw_open._data = filtfilt(coeffs,1, raw_open._data)
hil = raw_open.copy().pick_channels(['env']).apply_hilbert(envelope=False).get_data()
phases_brain_online = np.angle(hil).flatten()
phases_brain = phases_brain_online
amplitudes_brain = np.abs(hil).flatten()

t = np.arange(0,2,1/500)
phases_flicker_trial = wrap(2*np.pi*10*t)
nsamp_trial = phases_flicker_trial.size
nsamp_start = 0
phasediffs = []
amplitudes = []
for ev in audio_events_open:
    if ev < len(raw_open.times)-nsamp_trial:
        phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
        amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

phasediffs_tacs_without_sass = circ_detrend(np.array(phasediffs)[:n_trials])
amplitudes_tacs_without_sass = np.array(amplitudes)[:n_trials]

# #data_online = raw_open.get_data(["sass_output"])
# raw_open.drop_channels(["env", "Flicker","sass_output"])
# raw_no_stim.drop_channels(["env", "Flicker","sass_output"])
# raw_open = sliding_SASS(raw_open.copy(),raw_no_stim,raw_open.time_as_index(winsize_sec)[0],raw_open.time_as_index(stepsize_sec)[0])
# #raw_open = SASS(raw_open.copy(),raw_no_stim)
#
# chidxs = [raw_open.ch_names.index(chname) for chname in raw_open.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
#
# # data_sliding = raw_open_sliding._data[chidxs].mean(0)
# # data = raw_open._data[chidxs].mean(0)
# # diff = data - data_sliding
# # plt.figure()
# # plt.plot(data- data_sliding)
# # plt.show()
#
# data = raw_open._data[chidxs].mean(0)
# #phases_offline_dft = compute_phase_dft(data)
# #env_dft = compute_envelope_dft(data)
# hil = hilbert(data.flatten())
# #hil = raw_open.copy().pick_channels([ch for ch in raw_no_stim.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
# phases_brain_offline = np.angle(hil)
# phases_brain = phases_brain_offline
# amplitudes_brain = np.abs(hil)
#
# t = np.arange(0,2,1/500)
# phases_flicker_trial = wrap(2*np.pi*10*t)
# nsamp_trial = phases_flicker_trial.size
# nsamp_start = 0
# phasediffs = []
# plvs = []
# amplitudes = []
# for ev in audio_events_open:
#     if ev < len(raw_open.times)-nsamp_trial:
#         phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
#         plvs.append(plv(phases_brain[ev + nsamp_start:ev + nsamp_trial] - phases_flicker_trial[nsamp_start:]))
#         amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))
#
# phasediffs_tacs_with_sass = np.array(phasediffs)[:n_trials]
# amplitudes_tacs_with_sass = np.array(amplitudes)[:n_trials]
# plvs_tacs_with_sass = np.array(plvs)[:n_trials]

fig,axs = plt.subplots(1,2,subplot_kw=dict(polar=True),figsize=(18,6))
ax = axs[0]
ax.tick_params(pad=10)
ax.yaxis.grid(False)
ax.xaxis.grid(False)
ax.get_yaxis().set_visible(False)
color = '#ffffc7'
border='black'
ax.hist(phasediffs_no_tacs,color=color,edgecolor=border,bins=binmethod)
ax.tick_params(axis='x', labelsize=25)
#ax.spines.set_color(border)
ax.set_title('No tACS',y=1.2)
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
    tick.set_color(border)
# plt.title(str(plv(phasediffs_no_tacs)))
ax = axs[1]
ax.tick_params(pad=10)
ax.yaxis.grid(False)
ax.xaxis.grid(False)
ax.get_yaxis().set_visible(False)
ax.hist(phasediffs_tacs_without_sass,color=color,edgecolor=border,bins=binmethod)
ax.tick_params(axis='x', labelsize=25)
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
    tick.set_color(border)
ax.set_title('tACS',y=1.2)
#plt.title(str(plv(phasediffs_tacs_without_sass)))
# ax = axs[2]
# ax.tick_params(pad=10)
# ax.yaxis.grid(False)
# ax.xaxis.grid(False)
# ax.get_yaxis().set_visible(False)
# ax.hist(phasediffs_tacs_with_sass,edgecolor='black',bins=binmethod)
# ax.tick_params(axis='x', labelsize=15)
# ax.set_title('offline SASS',y=1.2)
# # plt.title(str(plv(phasediffs_tacs_with_sass)))
fig.tight_layout()


plt.show()