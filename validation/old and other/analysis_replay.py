# Analyse the performance of the Closed-Loop System (phase locking of envelope with visual flicker)

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert
from scipy import stats
from scipy.io import loadmat
from circular_hist import circular_hist
from scipy import stats, linalg
from mne.time_frequency import psd_welch

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi


def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        diff = (B-P.dot(A).dot(P.T))**2
        mses.append(np.mean(diff))
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

def SASS_combined(raw_open1,raw_open2, raw_no_stim):
    B = np.cov(raw_no_stim.copy().filter(freq-1,freq+1)._data)
    # Compute the first Covariance matrix
    combined_data = np.hstack((raw_open1.copy().filter(freq-1,freq+1)._data,raw_open2.copy().filter(freq-1,freq+1)._data))
    A = np.cov(combined_data)
    P = compute_P(A, B)
    raw_open1._data = P.dot(raw_open1._data)
    raw_open2._data = P.dot(raw_open2._data)
    return raw_open1, raw_open2

def window_SASS(raw_open, raw_no_stim, window):
    B = np.cov(raw_no_stim._data)
    # Samples at which the filter should be updated and applied
    for i in range(0, raw_open.n_times-window,window):
        A = np.cov(raw_open._data[:, i:i+window])
        # Update the covariance matrix at each sample
        P = compute_P(A, B)
        raw_open._data[:, i:i+window] = P.dot(raw_open._data[:, i:i+window])
    return raw_open


i_par = 5
data_path = "C:/Users/alessia/Documents/Praktika/Berlin/CL-amtACS/CL-Validation/data/pilot/"
bads = ["env","sass_output","Flicker","CPz","Cz"]
freq = 12
# Load the data without stimulation
file_path = data_path + f'{i_par}/'+'no_stim.vhdr'
no_stim = mne.io.read_raw_brainvision(file_path,preload=True)
sfreq = no_stim.info["sfreq"]
no_stim_env = no_stim.get_data("env").flatten()
no_stim.drop_channels(bads)

# psds, freqs = psd_welch(inst=no_stim, fmin=1, fmax=30, picks=["env"], n_fft=int(sfreq*5))
# plt.subplot(1,3,1)
# plt.semilogy(freqs, psds.T)
# plt.xlabel("Frequency in Hz")
# plt.ylabel("Power Spectral Density in dB")
# plt.title("No stim")

#no_stim.filter(freq-1,freq+1)
ch_names = no_stim.ch_names

# Load the data with closed loop stimulation
file_path = data_path+'/'+ f'/{i_par}/'+'stim.vhdr'
cl_stim = mne.io.read_raw_brainvision(file_path, preload=True)
cl_env = cl_stim.get_data("env").flatten()
cl_stim.drop_channels(bads)

# psds, freqs = psd_welch(inst=cl_stim, fmin=1, fmax=30, picks=["env"], n_fft=int(sfreq*5))
# plt.subplot(1,3,2)
# plt.semilogy(freqs, psds.T)
# plt.xlabel("Frequency in Hz")
# plt.ylabel("Power Spectral Density in dB")
# plt.title("CL stim")

#cl_stim.filter(freq-1,freq+1)


# Load the data with replayed stimulation
file_path = data_path+'/'+ f'/{i_par}/'+'replay.vhdr'
replay_stim = mne.io.read_raw_brainvision(file_path, preload=True)
replay_env = replay_stim.get_data("env").flatten()
replay_stim.drop_channels(bads)

# psds, freqs = psd_welch(inst=replay_stim, fmin=1, fmax=30, picks=["env"], n_fft=int(sfreq*5))
# plt.subplot(1,3,3)
# plt.semilogy(freqs, psds.T)
# plt.xlabel("Frequency in Hz")
# plt.ylabel("Power Spectral Density in dB")
# plt.title("Replay stim")
#plt.show()

#replay_stim.filter(freq-1,freq+1)

# Filter out the artifact for closed loop and replay stim
winsize_sec = cl_stim.time_as_index(25)[0]
ix_left = ["O2"]
ix_center = ["PO8"]
ix_right = ["P8"]

cl_stim_SASS, replay_stim = SASS_combined(cl_stim.copy(),replay_stim,no_stim)
#cl_stim = window_SASS(cl_stim,no_stim,winsize_sec)
#replay_stim = window_SASS(replay_stim,no_stim,winsize_sec)

for ch in ch_names:

    psds, freqs = psd_welch(inst=cl_stim, fmin=1, fmax=60, picks=ch, n_fft=int(sfreq*5))
    plt.semilogy(freqs, psds.T,label="CL")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Power Spectral Density in dB")


    psds, freqs = psd_welch(inst=cl_stim_SASS, fmin=1, fmax=60, picks=ch, n_fft=int(sfreq*5))
    plt.semilogy(freqs, psds.T,label="CL SASS")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Power Spectral Density in dB")

    psds, freqs = psd_welch(inst=no_stim, fmin=1, fmax=60, picks=ch, n_fft=int(sfreq*5))
    plt.semilogy(freqs, psds.T,label="No Stim")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Power Spectral Density in dB")
    plt.legend()
    plt.suptitle(ch)
    plt.show()

# Compute the phase locking between the envelope and the filtered channels of interest
cl_stim_chan = cl_stim.get_data(ix_left) + cl_stim.get_data(ix_right) - 2* cl_stim.get_data(ix_center)
cl_hil = hilbert(cl_stim_chan.flatten())
cl_diff = wrap(np.angle(hilbert(cl_env))-np.angle(cl_hil))

replay_stim_chan = replay_stim.get_data(ix_left) + replay_stim.get_data(ix_right) - 2* replay_stim.get_data(ix_center)
replay_hil = hilbert(replay_stim_chan.flatten())
replay_diff = wrap(np.angle(hilbert(replay_env))-np.angle(replay_hil))

no_stim_chan = no_stim.get_data(ix_left) + no_stim.get_data(ix_right) - 2* no_stim.get_data(ix_center)
no_stim_hil = hilbert(no_stim_chan.flatten())
no_stim_diff = wrap(np.angle(hilbert(no_stim_env))-np.angle(no_stim_hil))


# Plot the phases for stim and no stim
fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))

# Visualise by area of bins
circular_hist(ax[0], no_stim_diff)
ax[0].set_title("No Stim")
# Visualise by radius of bins
circular_hist(ax[1], cl_diff)
ax[1].set_title("Closed-Loop")
circular_hist(ax[2], replay_diff)
ax[2].set_title("Replay")
plt.suptitle("1 mA")
#plt.savefig('1 mA 180 °.png')
plt.show()

