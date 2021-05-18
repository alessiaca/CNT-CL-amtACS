% Load the data needed to simulate the "closed-loop amtacs model"
clear all;

load('C:\Users\David\Documents\Closed-Loop amtacs\Data\ch_names.mat')

% Decide what experiment you are going to perform 
prompt = 'What experiment are you going to simulate? No Stimulation = 1, Calibration = 2, Stimulation = 3 \n';
cond = input(prompt);

% Set the path to the dataset that you want to sue for simulation
path = uigetdir("C:\Users\David\Documents\Closed-Loop amtacs\Data");
if cond == 1
    signal_in = load(strcat(path,"\signal_in_no_stim.mat"));
    signal_in = signal_in.signal_in;
else
    signal_in = load(strcat(path,"\signal_in_stim.mat"));
    signal_in = signal_in.signal_in;
end

%% No Stimulation
if cond == 1
    % Define parameters that won't be used (needed for SASS)
    C_B_64 = ones(64,64); 
    stim_amplitude =0 ;initial_phase_diff = 0;in_phase_diff = 0; anti_phase_diff = 0;
    
    % Get information from the user
    prompt = 'What is the target frequency? \n';
    frequency = input(prompt);
    prompt = 'Which electrodes do you want to exclude? ["xx","xx"..] \n';
    exclude_channels = input(prompt);
    exclude_idx = find(any(exclude_channels == ch_names,2));
    disp(ch_names(exclude_idx)+" will be excluded");
    prompt = 'Which electrodes do use left? "xx" \n';
    ix_left = find(any(ch_names == input(prompt),2));
    prompt = 'Which electrodes do use center? "xx" \n';
    ix_center = find(any(ch_names == input(prompt),2));
    prompt = 'Which electrodes do use right? "xx" \n';
    ix_right = find(any(ch_names == input(prompt),2));
    disp(ch_names([ix_left ix_center ix_right])+" will be used for " + ["left";"center";"right"]);
    
%% With Stimulation (2 or 3)
else
    % Load the information from that participant
    load(strcat(path,"\exclude_idx.mat"));
    load(strcat(path,"\C_B.mat"));
    load(strcat(path,"\frequency.mat"));
    load(strcat(path,"\chidx.mat"));
    ix_left = chidx(1); ix_center = chidx(2); ix_right = chidx(3);
    
    % Get more input
    stim_amplitude = 1; is_trigger = 0; in_phase_diff = 0; anti_phase_diff = 0;
    prompt = 'What phase difference to you want?\n';
    initial_phase_diff = input(prompt);
  
end

%% Make the filter coefficients given the frequency 
n_cycles = 3;
n_samp = floor((500/frequency)*n_cycles);
h_freq = frequency+1;
l_freq = frequency-1;
filter_coeffs = fir1(n_samp,[l_freq/250,h_freq/250]);

n_samp_env = 400;
lowpass_freq = 15;
filter_coeffs_env = fir1(n_samp_env,lowpass_freq/10000);

n_cycles = 3;
n_samp = floor((250/frequency)*n_cycles);
h_freq = frequency+1;
l_freq = frequency-1;
filter_coeffs_sass = fir1(n_samp,[l_freq/125,h_freq/125]);
