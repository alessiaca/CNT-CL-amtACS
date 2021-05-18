import numpy as np
import pandas
import matplotlib.pyplot as plt
from random import choices
from scipy.stats import mannwhitneyu
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats import norm
Z = norm.ppf

# Take same for each condition
def bootstrap_data(data):
    """From the presented/not presented images samples new ones"""
    new_data = np.empty((0,data.shape[1]), int)
    for thres in [13,33]:
        data_tmp = data[(data[:,4] < thres+20) & (data[:,4] >= thres),:]
        n_trials = len(data_tmp)
        new_data_tmp = np.array(choices(data_tmp, k=n_trials))
        new_data = np.vstack((new_data,new_data_tmp))
    return np.array(new_data)

# Analyse the behavior: Recognition accuracy

n_pars = 12
n_conds = 3
recognition = 1 # Day 1 or 2
n_boots = 100
for day in [1,2]:
    plt.figure()
    for i_par in range(1,n_pars+1):

        # Read the data
        file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\behaviour\\data\\{i_par}\\{i_par}_recognition{day}.csv"
        data = pandas.read_csv(file_path)
        data = pandas.DataFrame.to_numpy(data)

        # Compute the recognition accuracy
        conditions = np.unique(data[:,1])
        conditions_names = ["No stim", "In phase", "Open loop"]
        accuracies = np.zeros((n_conds, n_boots))
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
                acc = hit_rate - fa_rate # 0 % Chance level
                accuracies[int(cond-1),j] = acc

        plt.subplot(3, 4, i_par)
        ys = list(accuracies[0,:]) + list(accuracies[1,:]) + list(accuracies[2,:])
        xs = ['No stim'] * len(accuracies.T) + ['In phase'] * len(accuracies.T) + ['Open loop'] * len(accuracies.T)
        ax = sns.boxplot(xs, ys, boxprops=dict(alpha=.5),showfliers=False)
        ax.set_xticklabels(ax.get_xticklabels(), size=5)
        plt.yticks(fontsize=5)
        #ax.set_yticklabels(ax.get_yticklabels(), size=5)
        if i_par ==1 or i_par == 5: plt.ylabel("Hit-False alarm in %",fontsize=8)
        plt.title(f"Participant {i_par}",fontsize=8)

        # add_stat_annotation(ax, x=xs, y=ys,
        #                     box_pairs=[("No stim", "In phase"), ("No stim", "Open loop"), ("In phase", "Open loop")],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)
        plt.subplots_adjust(hspace=0.3,wspace=0.3)

    plt.suptitle(f"Recognition day {day}")
    plt.show()

