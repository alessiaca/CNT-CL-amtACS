import mne
from scipy.io import savemat
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Load data
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data")
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)

# Save the stimulation envelope
data = raw_stim.filter(1,30).get_data(['env']).flatten()*(1/0.3)
savemat(folder_path+'/replay_signal.mat',dict(replay_signal=data[:,np.newaxis]))
plt.plot(raw_stim.times,data)
plt.xlabel('Time (s)')
plt.show()