import numpy as np
import mne
from scipy.io import savemat
from mne.time_frequency import psd_welch, psd_array_welch
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.stats import zscore
from utils import  get_target_chan, fir_coeffs
import tkinter as tk
from tkinter import filedialog


# After the run without stimulation use this script to determine the individual frequency and to generate the covariance
# matrix of the data without stimulation

# Load the data without stimulation
path = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation\\debugging\\"
raw_calib = mne.io.read_raw_brainvision(path+'triggers_calibration.vhdr',preload=True)
raw_encoding = mne.io.read_raw_brainvision(path+'triggers_encoding.vhdr',preload=True)
triggers_calib = np.load(path+"trigger_calibration.npy")
triggers_encod = np.load(path+"trigger_encoding.npy")
events_calib = mne.events_from_annotations(raw_calib)[0]
events_encoding = mne.events_from_annotations(raw_encoding)[0]
events_calib_cut = events_calib[1:,2]
events_encoding_cut = events_encoding[1:,2]
print("h")