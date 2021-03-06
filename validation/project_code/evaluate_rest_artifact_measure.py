# Evaluate rest artifact measure with help of flicker and rivalry data

# Prepare environment
import numpy as np
import mne
import matplotlib.pyplot as plt
import itertools
from main_code.utils import find_bad_channels, get_flicker_events, \
      SASS, compute_phase, compute_ITC, compare_rank_vector_distribution
from ordpy.ordpy import permutation_entropy, ordinal_distribution, ordinal_sequence
from pyriemann.utils.distance import distance_riemann, distance_kullback, distance_euclid

# Loop over the flicker dataset

# Set the folder path and some other parameters
folder_path = "C:/Users/alessia/Charité - Universitätsmedizin Berlin/Haslacher, David - SASS_data/"
n_par = 6
frequency = 10; lfreq = frequency - 1; hfreq = frequency + 1
plot_polar = False
dx = 5; taux = 25
all_rank_vectors = np.array(list(itertools.permutations(range(dx), dx))) # Array of all possible rank vectors

for i_par in range(1, n_par+1):

    # Load the data
    raw_no_stim = mne.io.read_raw_brainvision(folder_path + f'p{i_par}/no_stim.vhdr', preload=True)
    raw_stim = mne.io.read_raw_brainvision(folder_path + f'p{i_par}/open.vhdr', preload=True)

    # Get the flicker events
    no_stim_flicker_events, no_stim_flicker_event_onset = get_flicker_events(raw_no_stim)
    stim_flicker_events, stim_flicker_event_onset = get_flicker_events(raw_stim)

    # Drop non EEG channels
    non_EEG = ['env', 'sass_output', 'Flicker', 'audio', 'stim', 'ECG']
    raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in non_EEG])
    raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in non_EEG])

    # Find bad EEG channels
    bads_no_stim = find_bad_channels(raw_no_stim, use_zscore=True)
    bads_stim = find_bad_channels(raw_stim, use_zscore=True)
    bads = np.unique(np.concatenate((bads_no_stim, bads_stim)))

    # Drop bad channels
    raw_no_stim.drop_channels(bads)
    raw_stim.drop_channels(bads)
    n_chans = len(raw_stim.ch_names)

    # Filter the data
    raw_no_stim.filter(lfreq, hfreq)
    raw_stim.filter(lfreq, hfreq)

    # Get the target channel
    chidx = raw_stim.ch_names.index("O2")
    phase_no_stim = compute_phase(raw_no_stim._data[chidx])
    phase_stim = compute_phase(raw_stim._data[chidx])

    # Compute the measures for the data without and with stimulation (before artifact rejection)
    # Compute the ITC
    ITC_no_stim, phase_diffs_no_stim = compute_ITC(phase_no_stim, no_stim_flicker_event_onset)
    ITC_stim, phase_diffs_stim = compute_ITC(phase_stim, stim_flicker_event_onset)

    # Compute the covariance
    Cov_no_stim = np.median(np.cov(raw_no_stim._data))
    Cov_stim = np.median(np.cov(raw_stim._data))

    # Compute the permutation entropy
    PE_no_stim = permutation_entropy(raw_no_stim._data[chidx], dx=dx, taux=taux)
    PE_stim = permutation_entropy(raw_stim._data[chidx], dx=dx, taux=taux)

    # Compute the distance between the rank vector distributions
    RVDD_stim, probs_no_stim_ordered = compare_rank_vector_distribution(raw_stim._data[chidx], raw_no_stim._data[chidx],
                                                                        dx=dx, taux=taux, all_rank_vectors=all_rank_vectors)

    # Compute the distance between the covaraince matrices
    CD_stim = distance_euclid(np.cov(raw_no_stim._data), np.cov(raw_stim._data))

    # Compute the measures for each number of deleted nulls
    ITCs = np.zeros((n_chans, 1)); PEs = np.zeros((n_chans, 1)); Covs = np.zeros((n_chans, 1))
    CDs = np.zeros((n_chans, 1)); RVDDs = np.zeros((n_chans, 1))
    for n_null in range(n_chans):

        print(f"{n_null} out of {n_chans}")

        raw_stim_SASS = SASS(raw_stim.copy(), raw_no_stim.copy(), filter_type=None, n_nulls=n_null)

        # Compute the phase of the target channel for the different conditions
        phase_stim_SASS = compute_phase(raw_stim_SASS._data[chidx])

        # Compute the measures for the current n_null
        ITC, phase_diffs_stim_SASS = compute_ITC(phase_stim_SASS, stim_flicker_event_onset)
        ITCs[n_null] = ITC
        Covs[n_null] = np.median(np.cov(raw_stim_SASS._data))
        PEs[n_null] = permutation_entropy(raw_stim_SASS._data[chidx], dx=dx, taux=taux)
        RVDDs[n_null], _ = compare_rank_vector_distribution(raw_stim_SASS._data[chidx], raw_no_stim._data[chidx],
                                                     dx=dx, taux=taux, all_rank_vectors=all_rank_vectors,
                                                     probs2_ordered=probs_no_stim_ordered)
        CDs[n_null] = distance_euclid(np.cov(raw_no_stim._data), np.cov(raw_stim_SASS._data))

        # Plot the polar histogram of mean phase differences (for visual inspection)
        if plot_polar:
            fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(18, 6))
            ax = axs[0]; ax.hist(phase_diffs_no_stim); ax.set_title("No Stim")
            ax = axs[1]; ax.hist(phase_diffs_stim); ax.set_title("Stim")
            ax = axs[2]; ax.hist(phase_diffs_stim_SASS); ax.set_title("Stim SASS")
            plt.suptitle(f"n_nulls = {n_null}"); plt.show()

    # Plot the relation between deleted nulls and the measures
    plt.figure()
    plt.subplot(2,3,1)
    plt.plot(range(n_chans), ITCs)
    plt.axhline(ITC_no_stim, color="g", label="No stim"); plt.axhline(ITC_stim, color="r", label="Stim")
    n_nulls = np.argmin(np.abs(ITCs-ITC_no_stim))
    plt.annotate(n_nulls, (n_nulls, ITCs[n_nulls]))
    plt.plot(n_nulls, ITCs[n_nulls], 'o')
    plt.xlabel("#nulls"); plt.ylabel("ITC")
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(range(n_chans), PEs)
    plt.axhline(PE_no_stim, color="g", label="No stim"); plt.axhline(PE_stim, color="r", label="Stim")
    n_nulls = np.argmin(np.abs(PEs - PE_no_stim))
    plt.annotate(n_nulls, (n_nulls, PEs[n_nulls]))
    plt.plot(n_nulls, PEs[n_nulls], 'o')
    plt.xlabel("#nulls"); plt.ylabel("Permutation Entropy")
    plt.legend()

    plt.subplot(2,3,3)
    plt.plot(range(n_chans), Covs)
    plt.axhline(Cov_no_stim, color="g", label="No stim"); plt.axhline(Cov_stim, color="r", label="Stim")
    n_nulls = np.argmin(np.abs(Covs-Cov_no_stim))
    plt.annotate(n_nulls, (n_nulls, Covs[n_nulls]))
    plt.plot(n_nulls, Covs[n_nulls], 'o')
    plt.xlabel("#nulls"); plt.ylabel("Mean Covariance")
    plt.legend()

    plt.subplot(2,3,4)
    plt.plot(range(n_chans), CDs)
    plt.axhline(CD_stim, color="g", label="stim")
    plt.xlabel("#nulls"); plt.ylabel("Distance no_stim-stim")
    n_nulls = np.argmin(CDs)
    plt.annotate(n_nulls, (n_nulls, CDs[n_nulls]))
    plt.plot(n_nulls, CDs[n_nulls], 'o')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(range(n_chans), RVDDs)
    plt.axhline(RVDD_stim, color="g", label="stim")
    plt.xlabel("#nulls"); plt.ylabel("Distance rank vector distribution")
    n_nulls = np.argmin(RVDDs)
    plt.annotate(n_nulls, (n_nulls, RVDDs[n_nulls]))
    plt.plot(n_nulls, RVDDs[n_nulls], 'o')
    plt.legend();
print("debug")