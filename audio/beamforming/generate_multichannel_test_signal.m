clc; clear; close all;

%% ================= USER PARAMETERS =================
voicePath = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav';   % <-- change later
noisePath = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/female_piano_14s.wav';   % <-- change later

theta_voice = 0;    % speaker DOA (deg)
theta_noise = 40;   % noise DOA (deg)
SNR_dB = 0;          % desired SNR at reference mic

fs_expected = 16000;

%% ================= ARRAY PARAMETERS =================
c = 343;             % speed of sound (m/s)
micSpacing = 0.08;   % 8 cm spacing
micPos = [0; micSpacing];
numMics = 2;

%% ================= LOAD AUDIO =================
[voice, fs_v] = audioread(voicePath);
[noise, fs_n] = audioread(noisePath);

if fs_v ~= fs_expected || fs_n ~= fs_expected
    error('Sampling rate mismatch');
end

voice = mean(voice,2);   % force mono
noise = mean(noise,2);

%% ================= LENGTH ALIGNMENT =================
L = min(length(voice), length(noise));
voice = voice(1:L);
noise = noise(1:L);

%% ================= SNR SCALING =================
voice = voice / rms(voice);
noise = noise / rms(noise);

noise = noise * 10^(-SNR_dB/20);

%% ================= DELAY COMPUTATION =================
tau_voice = micPos * sin(deg2rad(theta_voice)) / c;
tau_noise = micPos * sin(deg2rad(theta_noise)) / c;

%% ================= MULTICHANNEL MIXTURE =================
x = zeros(L, numMics);

for m = 1:numMics
    v_delayed = delayseq(voice, tau_voice(m), fs_expected);
    n_delayed = delayseq(noise, tau_noise(m), fs_expected);
    x(:,m) = v_delayed + n_delayed;
end

%% ================= NORMALIZATION =================
x = x / max(abs(x(:)));

%% ================= LISTENING CHECK =================
disp('Playing Mic 1...');
% soundsc(x(:,1), fs_expected); pause(3);

disp('Playing Mic 2...');
% soundsc(x(:,2), fs_expected);

%% ================= SAVE FOR PIPELINE =================
audiowrite('Test_audio/multichannel_maleClean_femalePiano.wav', x, fs_expected);

disp('Multichannel test signal created and saved.');
