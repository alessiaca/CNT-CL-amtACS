% Load the data needed to run the "closed-loop amtacs model"
clear all;

load('C:\Users\David\Documents\Closed-Loop amtacs\Data\ch_names.mat')

% Decide what experiment you are going to perform 
prompt = 'What experiment to perform? No Stimulation = 1, Calibration = 2, Stimulation = 3 \n';
cond = input(prompt);

%% No Stimulation
if cond == 1
    % Define parameters that won't be used (needed for SASS)
    C_B_64 = ones(64,64); 
    stim_amplitude=0;initial_phase_diff = 0;in_phase_diff = 0; anti_phase_diff = 0; is_trigger = 0;
    
    prompt = 'Do you want to use the default memory consolidation? \n';
    default_memory = input(prompt);
    if default_memory
        frequency = 6; exclude_idx = [13,63,55,29,57,31]; ix_left = 3; ix_center = 18; ix_right = 4;
    else
        % Get information from the user
        prompt = 'What is the target frequency? \n';
        frequency = input(prompt);
        prompt = 'Which electrodes do you want to exclude? ["xx","xx"..] \n';
        exclude_channels = input(prompt);
        exclude_idx = find(any(exclude_channels == ch_names,2));
        prompt = 'Which electrodes do use left? "xx" \n';
        ix_left = find(any(ch_names == input(prompt),2));
        prompt = 'Which electrodes do use center? "xx" \n';
        ix_center = find(any(ch_names == input(prompt),2));
        prompt = 'Which electrodes do use right? "xx" \n';
        ix_right = find(any(ch_names == input(prompt),2));
    end
    disp(ch_names(exclude_idx)+" will be excluded");
    disp(ch_names([ix_left ix_center ix_right])+" will be used for " + ["left";"center";"right"]);
    
%% Calibration
else
    % Set the path to the participant as we already have data 
    path = uigetdir("C:\Users\David\Documents\Closed-Loop amtacs\Data");
    % Load the information from that participant
    load(strcat(path,"\exclude_idx.mat"));
    load(strcat(path,"\C_B.mat"));
    load(strcat(path,"\frequency.mat"));
    load(strcat(path,"\chidx.mat"));
    ix_left = chidx(1); ix_center=chidx(2); ix_right=chidx(3);
    disp(ch_names([ix_left ix_center ix_right])+" will be used for " + ["left";"center";"right"]);

    % Get more input
    prompt = 'What is the stimulation amplitude? \n';
    stim_amplitude = input(prompt);
    prompt = 'Do you send trigger to manage stimulation? Yes = 1, No = 0 \n';
    is_trigger = input(prompt);
    if ~is_trigger
        prompt = 'What phase difference to you want then?\n';
        initial_phase_diff = input(prompt);
    else
        initial_phase_diff = 0;
    end
    
    %% Stimulation
    if cond == 3 && is_trigger
        load(strcat(path,"\in_phase_diff.mat"));
        load(strcat(path,"\anti_phase_diff.mat"));
    else
        in_phase_diff = 0; anti_phase_diff = 0;
    end
    
    disp(ch_names(exclude_idx)+" will be excluded");
end

%% Make the filter coefficients given the frequency

if frequency < 10
    n_cycles = 1;
else
    n_cycles = 2;
end

n_cut = 0;
frequency_filt = 6;
n_samp = floor((500/frequency_filt)*n_cycles);
h_freq = frequency_filt+1;
l_freq = frequency_filt-1;
filter_coeffs = fir1(n_samp,[l_freq/250,h_freq/250]);

n_samp_env = 400;
lowpass_freq = 15;
filter_coeffs_env = fir1(n_samp_env,lowpass_freq/10000);

n_samp = 825;
h_freq = frequency+1;
l_freq = frequency-1;
filter_coeffs_sass = fir1(n_samp,[l_freq/125,h_freq/125]);

dft_abs_min = 1e-6;
dft_abs_max = 1e-2;

carrier_frequency = 40;