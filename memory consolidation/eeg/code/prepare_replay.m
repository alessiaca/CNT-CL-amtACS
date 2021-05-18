% Load the data needed to replay amtacs (from a closed-loop session)
clear all;
path = uigetdir("C:\Users\David\Documents\Closed-Loop amtacs\Data");
load(strcat(path,"\replay_signal.mat"));
prompt = 'What is the stimulation amplitude? \n';
stim_amplitude = input(prompt);