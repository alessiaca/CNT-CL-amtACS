import numpy as np
import pandas
import matplotlib.pyplot as plt
from random import choices, randint
from scipy.stats import mannwhitneyu
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats import ttest_ind
from scipy.stats import norm
Z = norm.ppf


# Analyse the behavior: Recognition accuracy
plot_lines = True # Whether to plot lines for individual participants
n_pars = 12
n_conds = 3
titles = ["Recognition after 20 min", "Recognition on next day"]

colors = []
for i in range(n_pars):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

for day in [1,2]:
    all_order = []
    accuracies = np.zeros((n_pars,n_conds))
    for i_par in range(1,n_pars+1):

        # Read the data
        file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\behaviour\\data\\{i_par}\\{i_par}_recognition{day}.csv"
        data = pandas.read_csv(file_path)
        data = pandas.DataFrame.to_numpy(data)

        # Compute the recognition accuracy
        conditions = np.unique(data[:,1])
        conditions_names = ["No stim", "In phase", "Open loop"]
        for cond in np.unique(conditions):

            # Get the data from the condition
            data_cond = data[data[:,1] == cond]

            # Get a first measure of the recognition accuracy (HIT - FALSE ALARMS)
            # But neglects the bias to say yes or no
            correct_pressed = data_cond[data_cond[:,5] == 1,6]
            hit_rate = np.sum(correct_pressed)/len(correct_pressed)
            fa_rate = 1 - hit_rate
            acc = Z(hit_rate) - Z(fa_rate) # 0 % Chance level
            accuracies[i_par-1,int(cond-1)] = acc


    ys = list(accuracies[:,0])+list(accuracies[:,1])+list(accuracies[:,2])
    xs = ['No stim']*len(accuracies)+['In phase']*len(accuracies)+['Open loop']*len(accuracies)
    plt.subplot(1, 2, day)
    ax = sns.boxplot(xs,ys,color ="grey",boxprops=dict(alpha=.5),showfliers=False)
    if plot_lines:
        for i_par in np.arange(n_pars):
            plt.plot([0,1,2],accuracies[i_par,:],color=colors[i_par],alpha=1,marker = "o")

    add_stat_annotation(ax, x = xs,y=ys,
                        box_pairs=[("No stim","In phase"), ("No stim", "Open loop"), ("In phase", "Open loop")],
                        test='Kruskal', text_format='simple', loc='inside', verbose=2)

    #plt.ylim([-0.50, 1.75])
    plt.title(titles[day-1])
    plt.ylabel("d' (Z(hit)-Z(FA))")


plt.subplots_adjust(wspace=0.4)
plt.show()
