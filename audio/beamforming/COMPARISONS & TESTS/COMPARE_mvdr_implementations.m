addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
clear; clc;

%% ================== USER CONFIG ==================
fs = 16000;
theta_target = 0;             % degrees
mic_pos = [0; 0.08];           % 2-mic linear array (meters)

% Replace paths as needed
clean_path = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav';
noise_path = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/female_piano_14s.wav';

%% ================== LOAD SIGNALS ==================
[s_clean, fs1] = audioread(clean_path);
[v_noise, fs2] = audioread(noise_path);

assert(fs1 == fs && fs2 == fs, 'Sampling rate mismatch');

L = min(length(s_clean), length(v_noise));
s_clean = s_clean(1:L);
v_noise = v_noise(1:L);

% Mixture (single-channel reference)
% x_mono = s_clean + v_noise;
% 
% % Two-mic mixture (plane wave assumption)
% delay = round((mic_pos(2) * sind(theta_target) / 340) * fs);
% x(:,1) = x_mono;
% x(:,2) = [zeros(delay,1); x_mono(1:end-delay)];

theta_noise = 40;
SNR_dB = 0;
x = generate_multichannel_test_signal_v2(clean_path, noise_path, theta_target, theta_noise, SNR_dB);
x_mono = x(:,1);
%% ================== RUN YOUR MVDR ==================
    
    % ================= params =================
N = 512;
hop = 256;
nfft = 512;
win = hann(N,'symmetric');
alpha = 0.98;
delta = 1e-3;

c = 343;
micSpacing = 0.08;
micPos = [0; micSpacing];
numMics = 2;

sigLen = size(x,1);
% ================= STFT =================
x_pad = [zeros(N,numMics); x; zeros(N,numMics)];
X = stft_multichannel(x_pad, win, hop, nfft);
[numFreqs, numFrames, ~] = size(X);
freqs = linspace(0, fs/2, numFreqs);
    % ================= COVARIANCE =================
Rxx = init_covariance(numFreqs, numMics);
for n = 1:numFrames
    Rxx = update_covariance(Rxx, squeeze(X(:,n,:)), alpha);
end
    % ================= MVDR =================
d = compute_steering_vector(theta_target, freqs, micPos, c);
W = compute_mvdr_weights(Rxx, d, delta);
Y = apply_mvdr(X, W);
% ================= ISTFT =================
y_mvdr = istft(Y, ...
    'Window', win, ...
    'OverlapLength', length(win)-hop, ...
    'FFTLength', nfft, ...
    'FrequencyRange', 'onesided');

y_mvdr = y_mvdr(1:sigLen);
y_mvdr_yours = real(y_mvdr);

%% ================== RUN TAKI MVDR ==================
    % =====================================================
% mic_pos_taki = [-0.08/2; 0.08/2];
mic_pos_taki = [0; 0.08];
Y_mvdr = mvdr_beamformer(x, fs, theta_target, mic_pos_taki);

    % ---------- ISTFT ----------
[~, params] = compute_stft(s_clean, fs);
mvdr_time = compute_istft(Y_mvdr, params);

mvdr_time = mvdr_time / (rms(mvdr_time)+1e-6) * rms(s_clean);
L = min(length(s_clean), length(mvdr_time));
y_mvdr_taki = mvdr_time(1:L);



%% trim BOTH signals to the minimum common length %%
% ======== FINAL ALIGNMENT FOR METRICS ========
L = min([
    length(s_clean), ...
    length(x_mono), ...
    length(y_mvdr_yours), ...
    length(y_mvdr_taki)
]);

s_clean_trim        = s_clean(1:L);
x_mono_trim  = x_mono(1:L);
y_mvdr_yours_trim  = y_mvdr_yours(1:L);
y_mvdr_taki_trim  = y_mvdr_taki(1:L);






%% ================== METRIC: OSINR ==================
osinr_mic  = compute_osinr(s_clean_trim, v_noise);
osinr_your = compute_osinr(s_clean_trim, y_mvdr_yours_trim - s_clean_trim);
osinr_taki = compute_osinr(s_clean_trim, y_mvdr_taki - s_clean_trim);

%% ================== METRIC: PESQ ==================
% MATLAB does not have a built-in PESQ function
% pesq_mic  = pesq(fs, s_clean_trim, x_mono, 'wb');
% pesq_your = pesq(fs, s_clean_trim, y_mvdr_yours_trim, 'wb');
% pesq_taki = pesq(fs, s_clean_trim, y_mvdr_taki_trim, 'wb');

%% ================== METRIC: STOI ==================
stoi_mic  = stoi(s_clean_trim, x_mono_trim, fs);
stoi_your = stoi(s_clean_trim, y_mvdr_yours_trim, fs);
stoi_taki = stoi(s_clean_trim, y_mvdr_taki_trim, fs);

%% ================== DISPLAY RESULTS ==================
fprintf('\n==== OBJECTIVE COMPARISON ====\n');

fprintf('\nOSINR (dB):\n');
fprintf('Mic:   %.2f\n', osinr_mic);
fprintf('Yours: %.2f\n', osinr_your);
fprintf('Taki:  %.2f\n', osinr_taki);

% fprintf('\nPESQ:\n');
% fprintf('Mic:   %.2f\n', pesq_mic);
% fprintf('Yours: %.2f\n', pesq_your);
% fprintf('Taki:  %.2f\n', pesq_taki);

fprintf('\nSTOI:\n');
fprintf('Mic:   %.3f\n', stoi_mic);
fprintf('Yours: %.3f\n', stoi_your);
fprintf('Taki:  %.3f\n', stoi_taki);





%% $$$$$$$$$$$$$ save outputs $$$$$$$$$$$$$$$$$$$$$$$
audiowrite('../Test_output/COMPARE1_mvdr_mic1_output.wav', s_clean, fs);
audiowrite('../Test_output/COMPARE1_mvdr_emon_output.wav', y_mvdr_yours, fs);
audiowrite('../Test_output/COMPARE1_mvdr_taki_output.wav', y_mvdr_taki, fs);
