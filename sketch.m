clc,clear all,close all
load('finalresults.mat')

[x,y]=find(score<0.70);  % this 0.70 can be changed to any value in rage 0-1

eeg_data=squeeze(eeg);
emg_data=squeeze(emg);
ne_data=squeeze(ne);

flag=zeros(1,size(eeg_data,1));
flag(y)=1;

EEG=[];
for i=1:size(eeg_data,1)
    EEG=[EEG,squeeze(eeg_data(i,:))];
end
EMG=[];
for i=1:size(emg_data,1)
    EMG=[EMG,squeeze(emg_data(i,:))];
end
NE=[];
for i=1:size(ne_data,1)
    NE=[NE,squeeze(ne_data(i,:))];
end

wake_binary_vector_cut=substitute_vector(final_pred,0);
sws_binary_vector_cut=substitute_vector(final_pred,1);
REM_binary_vector_cut=substitute_vector(final_pred,2);
flag_vector=substitute_vector(flag,1);
EEG_time = (1:length(EEG))/512;
ds_sec_signal=(1:length(NE))/1017;


sleepscore_time_cut = 0:length(wake_binary_vector_cut)-1; % should be same length for wake/sws/REM

fig = figure; % This shows what model predicts
a = subplot(3,1,1);
    plot_sleep(ds_sec_signal, NE, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut);
    title('NE2m');
b = subplot (3,1,2);
    ds_EEG_time = downsample(EEG_time_cut, 10);
    ds_EMG_rawtrace = downsample(EMG, 10);
    plot_sleep(ds_EEG_time, ds_EMG_rawtrace, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut);
    xlabel('time (s)');
    ylabel('EMG (V)');
c = subplot(3,1,3);
    ds_EEG_rawtrace = downsample(EEG, 10);
    plot_sleep(ds_EEG_time,ds_EEG_rawtrace, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut);
    xlabel('time (s)');
    ylabel('EEG (V)');
linkaxes([a,b,c],'x');

h = datacursormode(fig);
    h.UpdateFcn = @DataCursor_custom;
    h.SnapToDataVertex = 'on';
    datacursormode on



fig = figure; % This shows the trials with lower scores in yellow
a = subplot(3,1,1);
    plot_sleep(ds_sec_signal, NE, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut,flag_vector);
    title('NE2m');
b = subplot (3,1,2);
    ds_EEG_time = downsample(EEG_time_cut, 10);
    ds_EMG_rawtrace = downsample(EMG, 10);
    plot_sleep(ds_EEG_time, ds_EMG_rawtrace, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut,flag_vector);
    xlabel('time (s)');
    ylabel('EMG (V)');
c = subplot(3,1,3);
    ds_EEG_rawtrace = downsample(EEG, 10);
    plot_sleep(ds_EEG_time,ds_EEG_rawtrace, sleepscore_time_cut, wake_binary_vector_cut, sws_binary_vector_cut, REM_binary_vector_cut,flag_vector);
    xlabel('time (s)');
    ylabel('EEG (V)');
linkaxes([a,b,c],'x');

h = datacursormode(fig);
    h.UpdateFcn = @DataCursor_custom;
    h.SnapToDataVertex = 'on';
    datacursormode on