# Analyse the performance of the Closed-Loop System (phase locking of envelope with visual flicker)

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert
from scipy import stats
from scipy.io import loadmat
from circular_hist import circular_hist

i_par = 6
data_path = "C:/Users/alessia/Documents/Praktika/Berlin/CL-amtACS/CL-Validation/data/pilot/"

# Load the data without stimulation
file_path = data_path + f'{i_par}/'+'no_stim.vhdr'
raw_no_stim = mne.io.read_raw_brainvision(file_path,preload=True)

# Get the flicker events (when the grating appears)
threshold = 3
audio_events_no_stim = np.where(np.diff((np.abs(stats.zscore(raw_no_stim.copy().pick_channels(['Flicker'])._data.flatten())) > threshold).astype('int')) > 0)[0]
#audio_events_no_stim = [audio_events_no_stim[i] for i in range(1,len(audio_events_no_stim)) if (audio_events_no_stim[i]-audio_events_no_stim[i-1]) > 5 ]
plt.plot(stats.zscore(raw_no_stim.get_data(["Flicker"]).flatten()))
for event in audio_events_no_stim: plt.axvline(event,color="red")
#plt.show()
#audio_events_no_stim = np.random.randint(0,len(raw_no_stim.get_data(["Flicker"]).flatten()),len(audio_events_no_stim))
# Compute the phase locking between the envelope and the flicker
raw_no_stim.filter(9,11)
raw_no_stim.apply_hilbert()
env = raw_no_stim.get_data(picks=['env'])
phases_no_stim = np.angle(env[:,audio_events_no_stim])

# Load the data with stimulation
file_path = data_path+'/'+ f'/{i_par}/'+'stim.vhdr'
raw_stim = mne.io.read_raw_brainvision(file_path, preload=True)

# Get the flicker events (when the grating appears)
threshold = 3
audio_events_stim = np.where(np.diff((np.abs(stats.zscore(raw_stim.copy().pick_channels(['Flicker'])._data.flatten())) > threshold).astype('int')) > 0)[0]
#audio_events_stim = [audio_events_stim[i] for i in range(1,len(audio_events_stim)) if (audio_events_stim[i]-audio_events_stim[i-1]) > 5 ]

#audio_events_stim = np.random.randint(0,len(raw_no_stim.get_data(["Flicker"]).flatten()),len(audio_events_no_stim))

plt.plot(stats.zscore(raw_stim.get_data(["Flicker"]).flatten()))
for event in audio_events_stim: plt.axvline(event,color="red")
#plt.show()

# Compute the phase locking between the envelope and the flicker
raw_stim.filter(9, 11)
raw_stim.apply_hilbert()
env = raw_stim.get_data(picks=['env'])
phases_stim = np.angle(env[:, audio_events_stim])

# Plot the phases for stim and no stim
fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))

# Visualise by area of bins
circular_hist(ax[0], phases_no_stim)
ax[0].set_title("No stim")
# Visualise by radius of bins
circular_hist(ax[1], phases_stim)
ax[1].set_title("stim")
plt.suptitle("1 mA 0 °")
plt.savefig('1 mA 0 °.png')
plt.show()

