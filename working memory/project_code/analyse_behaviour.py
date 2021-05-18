import numpy as np
import pandas
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Read the data
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/WM-Task/data")
data = pandas.read_csv(file_path)
#print(data)

# Get the responses and reaction times for each conditions
conditions = [1,2,3,4]
condition_names = ["No Stim", "In Phase", "Anti Phase","Closed Loop"]
accuracies = np.zeros((len(conditions),1))
reaction_times = np.zeros((len(conditions),1))
for cond in conditions:
    data_res = data.loc[data["Condition"] == cond, ["Response"]]
    data_rt = data.loc[data["Condition"] == cond, ["ReactionTime"]]
    if len(data_res) > 0:
        accuracies[cond-1] = data_res.mean()[0]*100
        reaction_times[cond-1] = data_rt.mean()[0]

# Plot the results
plt.figure()
plt.subplot(1,2,1)
plt.bar(conditions,accuracies.flatten(), tick_label=condition_names)
plt.ylabel("Accuracy in %")
plt.subplot(1,2,2)
plt.bar(conditions,reaction_times.flatten(), tick_label=condition_names)
plt.ylabel("RT in s")
plt.show()
