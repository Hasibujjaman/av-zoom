clc; clear;

 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');


fs = 16000;
c = 343;
micSpacing = 0.08;
micPos = [0; micSpacing];

theta_true = 20;

% Synthetic source
t = (0:fs-1)'/fs;
s = sin(2*pi*1000*t);

% Delays
tau = micPos*sin(deg2rad(theta_true))/c;
x = zeros(length(s),2);
for m = 1:2
    x(:,m) = delayseq(s, tau(m), fs);
end


% STFT
N = 512; hop = 256; nfft = 512;
win = hann(N,'symmetric');
x_pad = [zeros(N,2); x; zeros(N,2)];
X = stft_multichannel(x_pad, win, hop, nfft);

[numFreqs, numFrames, ~] = size(X);
freqs = linspace(0, fs/2, numFreqs);

% Covariance
Rxx = init_covariance(numFreqs, 2);
alpha = 0.98;
for n = 1:numFrames
    Rxx = update_covariance(Rxx, squeeze(X(:,n,:)), alpha);
end

% Steering + MVDR
d = compute_steering_vector(theta_true, freqs, micPos, c);
W = compute_mvdr_weights(Rxx, d, 1e-3);

% Beamform
Y = apply_mvdr(X, W);

disp("MVDR output power:");
disp(mean(abs(Y(:)).^2));





%%%%%%%%%%% Additional Tests %%%%%%%%%%%%
%% Test 1
function test_distortionless(W, d)

[~, numFreqs] = size(W);
vals = zeros(numFreqs,1);

for k = 1:numFreqs
    vals(k) = abs(W(:,k)' * d(:,k));
end

figure;
plot(vals);
yline(1,'--');
xlabel('Frequency bin');
ylabel('|w^H d|');
title('Distortionless Constraint Check');
grid on;

disp(['Mean constraint value: ', num2str(mean(vals))]);
end

%% Test 2, V1
function magnitudedB = plot_mvdr_beampattern(W, freqs, micPos, c)

angles = -90:1:90;
numFreqs = length(freqs);
numAngles = length(angles);

k0 = round(numFreqs*0.3);   % mid-band freq
response = zeros(numAngles,1);

for i = 1:numAngles
    d_test = compute_steering_vector(angles(i), ...
                                     freqs(k0), micPos, c);
    response(i) = abs(W(:,k0)' * d_test(:,1));
end
magnitudedB = 20*log10(response/max(response));

figure;
plot(angles, 20*log10(response/max(response)));
xlabel('Angle (deg)');
ylabel('Magnitude (dB)');
title(['MVDR Beampattern @ ', num2str(freqs(k0)), ' Hz']);
ylim([-40 5]);
grid on;

end

%% Test 2, V2
function response_db = plot_mvdr_beampattern_multifreq(W, freqs, micPos, c)

angles = -90:1:90;
numAngles = length(angles);

% Speech-relevant frequency range
fmin = 300;
fmax = 4000;

freq_idx = find(freqs >= fmin & freqs <= fmax);
numUsedFreqs = length(freq_idx);

response = zeros(numAngles,1);

for i = 1:numAngles
    theta = angles(i);
    power_sum = 0;

    for k = freq_idx
        d = compute_steering_vector(theta, freqs(k), micPos, c);
        power_sum = power_sum + abs(W(:,k)' * d(:,1))^2;
    end

    response(i) = power_sum / numUsedFreqs;
end

% Normalize
response_db = 10*log10(response / max(response));

figure;
plot(angles, response_db, 'LineWidth', 2);
xlabel('Angle (deg)');
ylabel('Normalized Power (dB)');
title('Multi-Frequency Averaged MVDR Beampattern (300–4000 Hz)');
ylim([-40 3]);
grid on;

end

%% Test 3
function [target_gain,interference_gain] = test_interference_rejection(W, freqs, micPos, c)

theta_target = 20;
theta_interf = -40;

d_t = compute_steering_vector(theta_target, freqs, micPos, c);
d_i = compute_steering_vector(theta_interf, freqs, micPos, c);

gain_t = zeros(length(freqs),1);
gain_i = zeros(length(freqs),1);

for k = 1:length(freqs)
    gain_t(k) = abs(W(:,k)' * d_t(:,k));
    gain_i(k) = abs(W(:,k)' * d_i(:,k));
end
target_gain = 20*log10(gain_t);
interference_gain = 20*log10(gain_i);
figure;
plot(target_gain,'b'); hold on;
plot(interference_gain,'r');
legend('Target','Interference');
xlabel('Frequency bin');
ylabel('Gain (dB)');
title('Target vs Interference Gain');
grid on;


end

%% Test 4
function power = test_doa_sweep(X, Rxx, freqs, micPos, c)

angles = -60:2:60;
power = zeros(length(angles),1);

for i = 1:length(angles)
    d = compute_steering_vector(angles(i), freqs, micPos, c);
    W = compute_mvdr_weights(Rxx, d, 1e-3);
    Y = apply_mvdr(X, W);
    power(i) = mean(abs(Y(:)).^2);
end

figure;
plot(angles, power);
xlabel('Steering angle (deg)');
ylabel('Output power');
title('MVDR Output Power vs DOA');
grid on;

end


%% Test 5
function conditional_vals_log10 = test_cov_condition(Rxx)

numFreqs = size(Rxx,3);
cond_vals = zeros(numFreqs,1);

for k = 1:numFreqs
    cond_vals(k) = cond(Rxx(:,:,k));
end

conditional_vals_log10 = log10(cond_vals);
figure;
plot(log10(cond_vals));
xlabel('Frequency bin');
ylabel('log10(condition number)');
title('Covariance Conditioning');
grid on;

disp(['Median condition number: ', num2str(median(cond_vals))]);
end




%%% test driver

clear; close all;

%% ------------------- PARAMETERS -------------------
fs = 16000;
c = 343;
micSpacing = 0.05;
micPos = [0; micSpacing];
numMics = 2;

theta_target = 20;     % true DOA
theta_interf = -40;   % interference DOA
alpha = 0.98;
delta = 1e-3;

N = 512;
hop = 256;
nfft = 512;
win = hann(N,'symmetric');

%% ------------------- SIGNAL GENERATION -------------------
t = (0:fs-1)'/fs;
s_target = sin(2*pi*1000*t);
s_interf = sin(2*pi*600*t);

tau_t = micPos * sin(deg2rad(theta_target)) / c;
tau_i = micPos * sin(deg2rad(theta_interf)) / c;

x = zeros(length(t), numMics);
for m = 1:numMics
    x(:,m) = delayseq(s_target, tau_t(m), fs) ...
           + 0.7 * delayseq(s_interf, tau_i(m), fs);
end

x_pad = [zeros(N,numMics); x; zeros(N,numMics)];

%% ------------------- STFT -------------------
X = stft_multichannel(x_pad, win, hop, nfft);
[numFreqs, numFrames, ~] = size(X);
freqs = linspace(0, fs/2, numFreqs);

%% ------------------- COVARIANCE -------------------
Rxx = init_covariance(numFreqs, numMics);
for n = 1:numFrames
    Rxx = update_covariance(Rxx, squeeze(X(:,n,:)), alpha);
end

%% ------------------- STEERING + MVDR -------------------
d_target = compute_steering_vector(theta_target, freqs, micPos, c);
W = compute_mvdr_weights(Rxx, d_target, delta);

%% ================== TEST 1 ==================
disp("TEST 1: Distortionless Constraint");
test_distortionless(W, d_target);

%% ================== TEST 2 ==================
disp("TEST 2: Beampattern");
magitude_dB = plot_mvdr_beampattern(W, freqs, micPos, c); %v1
% response_dB = plot_mvdr_beampattern_multifreq(W, freqs, micPos, c); %v2

%% ================== TEST 3 ==================
disp("TEST 3: Interference Rejection");
[target_gain,interference_gain] = test_interference_rejection(W, freqs, micPos, c);

%% ================== TEST 4 ==================
disp("TEST 4: DOA Sweep");
output_power = test_doa_sweep(X, Rxx, freqs, micPos, c);

%% ================== TEST 5 ==================
disp("TEST 5: Covariance Conditioning");
conditional_vals_log10 = test_cov_condition(Rxx);



disp("ALL MVDR VALIDATION TESTS COMPLETED.");
