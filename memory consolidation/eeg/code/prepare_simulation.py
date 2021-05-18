import mne
from scipy.io import savemat
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from mne.time_frequency import psd_array_welch

# Load data that should be used for simulation
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\validation")
txt = input("Do you want no_stim or stim? Type stim or no_stim")
raw =  mne.io.read_raw_brainvision(folder_path + '/' + txt + '.vhdr', preload=True)
raw.drop_channels([ch for ch in raw.ch_names if ch in ['env','sass_output','Flicker']])

# Save the data in matlab format
savemat(folder_path+'/signal_in_' + txt +'.mat',dict(signal_in=raw._data.T))