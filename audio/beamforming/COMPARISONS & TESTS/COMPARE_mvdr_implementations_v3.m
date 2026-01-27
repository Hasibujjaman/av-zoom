%% ############ FAIR MVDR COMPARISON SCRIPT ############
clear; clc;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');

%% ================= CONFIG =================
fs = 16000;
theta_target = 0;
theta_noise  = 40;

theta_target_test = 40;  

SNR_dB       = 0;

mic_pos = [0; 0.08];
c = 343;

clean_path = 'male_clean_15s.wav';
noise_path = 'female_piano_14s.wav';

%% ================= LOAD SIGNALS =================
[s_clean, fs1] = audioread(clean_path);
[v_noise, fs2] = audioread(noise_path);
assert(fs1==fs && fs2==fs);

L = min(length(s_clean), length(v_noise));
s_clean = s_clean(1:L);
v_noise = v_noise(1:L);

%% ================= MULTICHANNEL MIXTURE =================
x = generate_multichannel_test_signal_v2( ...
        clean_path, noise_path, ...
        theta_target, theta_noise, ...
        SNR_dB);

x = x(1:L,:);
x_mono = x(:,1);

%% ================= YOUR MVDR =================
%% ================= YOUR MVDR (FIXED) =================

alpha = 0.98;
eps   = 1e-3;

[numSamples, numMics] = size(x);
sigLen = numSamples;

% --- STFT (EXACTLY like Taki) ---
for m = 1:numMics
    if m == 1
        [Xf(:,:,m), params] = compute_stft(x(:,m), fs);
    else
        [Xf(:,:,m), ~] = compute_stft(x(:,m), fs);
    end
end

% --- MVDR ---
Y = mvdr_time_adaptive( ...
        Xf, fs, theta_target_test, mic_pos, c, alpha, eps );

% --- ISTFT (NO extra cropping) ---
y_mvdr_yours = real(compute_istft(Y, params));


%% ================= TAKI MVDR =================
N = 512;
sigLen = size(x,1);
x_pad = [zeros(N,numMics); x; zeros(N,numMics)];

Y_taki = mvdr_beamformer(x_pad, fs, theta_target_test, mic_pos);

[~, params] = compute_stft(s_clean, fs);
y_mvdr_taki = compute_istft(Y_taki, params);
% ---------- Crop to original length ----------
y_mvdr_taki = y_mvdr_taki(N+1 : N+sigLen);
%% ================= ALIGN SIGNALS =================
Lmin = min([length(s_clean), length(x_mono), ...
            length(y_mvdr_yours), length(y_mvdr_taki)]);

s  = s_clean(1:Lmin);
x0 = x_mono(1:Lmin);
y1 = y_mvdr_yours(1:Lmin);
y2 = y_mvdr_taki(1:Lmin);

%% ================= METRICS =================

% ---- SI-SDR (CORRECT) ----
sisdr_mic   = si_sdr(x0, s);
sisdr_yours = si_sdr(y1, s);
sisdr_taki  = si_sdr(y2, s);

% ---- STOI (FAIR) ----
stoi_mic   = stoi(s, x0, fs);
stoi_yours = stoi(s, y1, fs);
stoi_taki  = stoi(s, y2, fs);

% ================= OSINR =================
% residuals
res_mic   = x0 - s;
res_yours = y1 - s;
res_taki  = y2 - s;

osinr_mic   = 10*log10( sum(s.^2) / (sum(res_mic.^2)   + 1e-12) );
osinr_yours = 10*log10( sum(s.^2) / (sum(res_yours.^2) + 1e-12) );
osinr_taki  = 10*log10( sum(s.^2) / (sum(res_taki.^2)  + 1e-12) );

%% ================= DISPLAY =================
fprintf('\n===== FAIR MVDR COMPARISON =====\n');

fprintf('\nSI-SDR (dB):\n');
fprintf('Mic:   %.2f\n', sisdr_mic);
fprintf('Yours: %.2f\n', sisdr_yours);
fprintf('Taki:  %.2f\n', sisdr_taki);

fprintf('\nSTOI:\n');
fprintf('Mic:   %.3f\n', stoi_mic);
fprintf('Yours: %.3f\n', stoi_yours);
fprintf('Taki:  %.3f\n', stoi_taki);


fprintf('\nOSINR (dB):\n');
fprintf('Mic:   %.2f\n', osinr_mic);
fprintf('Yours: %.2f\n', osinr_yours);
fprintf('Taki:  %.2f\n', osinr_taki);





%% $$$$$$$$$$$$$ save outputs $$$$$$$$$$$$$$$$$$$$$$$
audiowrite('../Test_output/COMPARE4_V2_mvdr_mic1_output.wav', s_clean, fs);
audiowrite('../Test_output/COMPARE4_V2_mvdr_emon_output.wav', y_mvdr_yours, fs);
audiowrite('../Test_output/COMPARE4_V2_mvdr_taki_output.wav', y_mvdr_taki, fs);