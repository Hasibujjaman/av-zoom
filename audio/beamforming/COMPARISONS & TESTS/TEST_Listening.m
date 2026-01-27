clc; clear; close all;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

%% ================= USER CONFIG =================
audioPath = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/multichannel_maleClean_femalePiano.wav';   % <-- CHANGE LATER
theta_target = 0;                      % target DOA (deg)
fs_expected = 16000;                    % expected sampling rate

%% ================= ARRAY CONFIG =================
c = 343;
micSpacing = 0.08;
micPos = [0; micSpacing];
numMics = 2;

%% ================= STFT CONFIG =================
N = 512;
hop = 256;
nfft = 512;
win = hann(N,'symmetric');
alpha = 0.98;
delta = 1e-3;

%% ================= LOAD AUDIO =================
[x, fs] = audioread(audioPath);

if fs ~= fs_expected
    
    error('Sampling rate mismatch');
end

if size(x,2) ~= numMics
    error('Audio must have exactly 2 channels');
end

sigLen = size(x,1);

%% ================= STFT =================
x_pad = [zeros(N,numMics); x; zeros(N,numMics)];
X = stft_multichannel(x_pad, win, hop, nfft);
[numFreqs, numFrames, ~] = size(X);
freqs = linspace(0, fs/2, numFreqs);

%% ================= COVARIANCE =================
Rxx = init_covariance(numFreqs, numMics);
for n = 1:numFrames
    Rxx = update_covariance(Rxx, squeeze(X(:,n,:)), alpha);
end

%% ================= MVDR =================
d = compute_steering_vector(theta_target, freqs, micPos, c);
W = compute_mvdr_weights(Rxx, d, delta);
Y = apply_mvdr(X, W);

%% ================= ISTFT =================
% y_mvdr = istft(Y, ...
%                'Window', win, ...
%                'OverlapLength', length(win)-hop, ...
%                'FFTLength', nfft);
y_mvdr = istft(Y, ...
    'Window', win, ...
    'OverlapLength', length(win)-hop, ...
    'FFTLength', nfft, ...
    'FrequencyRange', 'onesided');

y_mvdr = y_mvdr(1:sigLen);

%% ================= BASELINES =================
% Single mic
y_mic1 = x(:,1);

% Delay-and-sum
y_das = delay_and_sum(x, theta_target, micPos, fs, c);

%% ================= NORMALIZATION =================
y_mic1 = y_mic1 / max(abs(y_mic1));
y_das  = y_das  / max(abs(y_das));
y_mvdr = y_mvdr / max(abs(y_mvdr));

%% ================= LISTENING =================
disp('Playing single mic...');
% soundsc(y_mic1, fs); pause(15);
audiowrite('../Test_output/TEST_Listening2_mic1.wav', y_mic1, fs);


disp('Playing delay-and-sum...');
% soundsc(y_das, fs); pause(15);
% audiowrite('Test_output/TEST_Listening2_das.wav', y_das, fs);


disp('Playing MVDR...');
y_mvdr = real(y_mvdr);
% soundsc(y_mvdr, fs);
audiowrite('../Test_output/TEST_Listening2_MVDR_Emon.wav', y_mvdr, fs);


%% ================= METRICS =================
fprintf('\n==== OBJECTIVE METRICS ====\n');
y_mvdr_taki = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/mvdr_taki.wav');
% Output power
P_mic  = mean(y_mic1.^2);
P_das  = mean(y_das.^2);
P_mvdr = mean(y_mvdr.^2);
P_mvdr_taki = mean(y_mvdr_taki.^2);

fprintf('Output Power:\n');
% fprintf('Mic1: %.4f | DAS: %.4f | MVDR: %.4f\n | MVDR(Taki): %.4f\n', P_mic, P_das, P_mvdr, P_mvdr_taki);
fprintf('Mic1: %.4f | DAS: %.4f | MVDR: %.4f\n', P_mic, P_das, P_mvdr);


% Spectral energy comparison
spec_mic  = mean(abs(stft(y_mic1, 'Window',win,'OverlapLength',length(win)-hop,'FFTLength',nfft)).^2,'all');
spec_das  = mean(abs(stft(y_das,  'Window',win,'OverlapLength',length(win)-hop,'FFTLength',nfft)).^2,'all');
spec_mvdr = mean(abs(stft(y_mvdr, 'Window',win,'OverlapLength',length(win)-hop,'FFTLength',nfft)).^2,'all');
% spec_mvdr_taki = mean(abs(stft(y_mvdr_taki, 'Window',win,'OverlapLength',length(win)-hop,'FFTLength',nfft)).^2,'all');


fprintf('Spectral Energy:\n');
% fprintf('Mic1: %.2f | DAS: %.2f | MVDR: %.2f\n | MVDR(Taki): %.2f\n', spec_mic, spec_das, spec_mvdr, spec_mvdr_taki);
fprintf('Mic1: %.2f | DAS: %.2f | MVDR: %.2f\n', spec_mic, spec_das, spec_mvdr);


%% ================= VISUAL COMPARISON =================
figure;
subplot(4,1,1);
plot(y_mic1); title('Single Mic'); grid on;

subplot(4,1,2);
plot(y_das); title('Delay-and-Sum'); grid on;

subplot(4,1,3);
plot(y_mvdr); title('MVDR Output'); grid on;

% subplot(4,1,4);
% plot(y_mvdr_taki); title('MVDR Output (Taki)'); grid on;


%% ================= OPTIONAL: SPECTROGRAM =================
figure;
subplot(3,1,1);
spectrogram(y_mic1, win, length(win)-hop, nfft, fs, 'yaxis');
title('Input audio Spectrogram');
colorbar;

subplot(3,1,2);
spectrogram(y_mvdr, win, length(win)-hop, nfft, fs, 'yaxis');
title('MVDR Spectrogram');
colorbar;

% subplot(3,1,3);
% spectrogram(y_mvdr_taki, win, length(win)-hop, nfft, fs, 'yaxis');
% title('MVDR Spectrogram (Taki)');
% colorbar;


audioPath = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav';
y_clean = audioread(audioPath);
subplot(3,1,3);
spectrogram(y_clean, win, length(win)-hop, nfft, fs, 'yaxis');
title('Target Male clean audio');
colorbar;
