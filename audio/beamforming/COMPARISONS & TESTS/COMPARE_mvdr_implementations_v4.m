%% ############ FAIR MVDR COMPARISON SCRIPT ############
clear; 

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation');
%% ================= CONFIG =================
fs = 16000;
theta_target = 90;     % broadside (MATLAB convention)
theta_noise  = 40;     % 40° off broadside → 50° azimuth

theta_target_test = 90;  % ← MUST match theta_target (steer at the target!)

SNR_dB       = 5;  %5 dB

c = 340;
d = 0.08; % 8 cm

mic_pos = [-d/2; d/2];


clean_path = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/reverb_demo/1_clean_dry.wav';
% clean_path = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/prepared_dataset/test/clean/1_part3_A_female_only.wav';
noise_path = 'female_piano_14s.wav';

%% ================= LOAD SIGNALS =================
[s_clean, fs1] = audioread(clean_path);
[v_noise, fs2] = audioread(noise_path);
assert(fs1==fs && fs2==fs);

L = min(length(s_clean), length(v_noise));
s_clean = s_clean(1:L);
v_noise = v_noise(1:L);

%% ================= MULTICHANNEL MIXTURE =================
[x, target_mc, interf_mc, noise_mc] = create_mixture_v2( s_clean, v_noise, ...
        theta_target, theta_noise, fs, ...
        SNR_dB, c, d);
%% ******
x = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/reverb_demo/3_mixture_stereo.wav');
%% *******
x = x(1:L,:);
x_mono = x(:,1);

% verify mixture conditoins
sir_check = 10*log10( ...
    mean(target_mc(:,1).^2) / mean(interf_mc(:,1).^2) );

noise = x_mono - (target_mc(:,1) + interf_mc(:,1));
snr_check = 10*log10( ...
    mean((target_mc(:,1)+interf_mc(:,1)).^2) / mean(noise.^2) );

fprintf('\n\nSIR = %.2f dB\n', sir_check);
fprintf('SNR = %.2f dB\n', snr_check);

% -------- Convention sanity check --------
tau_target = mic_pos * cos(deg2rad(theta_target)) / c;
tau_noise  = mic_pos * cos(deg2rad(theta_noise))  / c;
tau_steer  = mic_pos * cos(deg2rad(theta_target_test)) / c;
fprintf('\n--- Convention Check ---\n');
fprintf('Target  delays (s): [%.6e, %.6e]  (should be ~0 for broadside)\n', tau_target(1), tau_target(2));
fprintf('Noise   delays (s): [%.6e, %.6e]  (should be non-zero)\n', tau_noise(1), tau_noise(2));
fprintf('Steer   delays (s): [%.6e, %.6e]  (must match target delays)\n', tau_steer(1), tau_steer(2));
fprintf('Inter-mic delay diff target: %.2e s\n', diff(tau_target));
fprintf('Inter-mic delay diff noise:  %.2e s\n', diff(tau_noise));

% -------- Check multichannel signal correlation --------
corr_ch = corrcoef(target_mc(:,1), target_mc(:,2));
fprintf('Target ch1-ch2 corr: %.4f  (should be ~1.0 for broadside)\n', corr_ch(1,2));
corr_ch_n = corrcoef(interf_mc(:,1), interf_mc(:,2));
fprintf('Interf ch1-ch2 corr: %.4f  (should be < 1.0)\n\n', corr_ch_n(1,2));


%% ================= YOUR MVDR =================
% -------- STFT params --------
N = 512;
hop = 128;
nfft = 512;
window = sqrt(hann(N,'periodic'));

%% Apply mvdr
X = stft_multichannel(x, window, hop, nfft);
[numFreqs, numFrames, numMics] = size(X);

freqs = (0:numFreqs-1)' * fs / nfft;

% -------- Batch sample covariance (uniform weighting) --------
% Better than EMA for offline: every frame contributes equally.
Rxx = zeros(numMics, numMics, numFreqs);
for n = 1:numFrames
    X_frame = squeeze(X(:,n,:));          % [numFreqs x numMics]
    for k = 1:numFreqs
        xk = X_frame(k,:).';             % [numMics x 1]
        Rxx(:,:,k) = Rxx(:,:,k) + (xk * xk');
    end
end
Rxx = Rxx / numFrames;

% -------- Steering vector --------
dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

% -------- MVDR weights --------
% Delta sweep showed 1e-6 to 1e-2 are equivalent (2.01 dB).
% Use 1e-3 as safe default; larger values push toward delay-and-sum.
delta = 1e-3;
W = compute_mvdr_weights(Rxx, dvec, delta);

% -------- Apply MVDR --------
Y = apply_mvdr(X, W);

% -------- MVDR + Wiener Post-Filter --------
Y_pf = mvdr_postfilter(X, W, dvec);

% -------- ISTFT --------
y_mvdr = istft_single_channel(Y, window, hop, nfft, L);
y_mvdr_pf = istft_single_channel(Y_pf, window, hop, nfft, L);

% Crop to original signal length
y_mvdr_yours = real(y_mvdr(1:L));
y_mvdr_postfilt = real(y_mvdr_pf(1:L));

sisdr_pf = si_sdr(y_mvdr_postfilt, s_clean(1:L));
fprintf('\n  MVDR only:         SI-SDR = %.2f dB\n', si_sdr(y_mvdr_yours, s_clean(1:L)));
fprintf('  MVDR + post-filter: SI-SDR = %.2f dB\n\n', sisdr_pf);


%% ================= TAKI MVDR =================
x_pad = [zeros(N,numMics); x; zeros(N,numMics)];
Y_taki = mvdr_beamformer(x_pad, fs, theta_target_test, mic_pos);

[~, params] = compute_stft(s_clean, fs);
y_mvdr_taki = compute_istft(Y_taki, params);
% ---------- Crop to original length ----------
y_mvdr_taki = y_mvdr_taki(N+1 : N+L);


%% ================= MATLAB MVDR =================
% ---- MATLAB Subband MVDR Beamformer ----
sa = phased.ULA( ...
        'NumElements', 2, ...
        'ElementSpacing', abs(diff(mic_pos)));

bm = phased.SubbandMVDRBeamformer( ...
    'SensorArray', sa, ...
    'OperatingFrequency', fs/2+1, ...
    'SampleRate', fs, ...
    'Direction', [theta_target_test; 0], ...
    'DiagonalLoadingFactor', delta, ...
    'WeightsOutputPort', false);

% MATLAB expects time-domain multichannel signal
y_mvdr_matlab = bm(x);

% Ensure column vector
y_mvdr_matlab = y_mvdr_matlab(:);

% Trim to match signal length
y_mvdr_matlab = y_mvdr_matlab(1:L);


%% ================= ML Post Enchancement =================
[y_model, ~] = audioread("/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/custom_model_1/model_enhanced_output/v2_2/COMPARE_V4_mvdr_emon_output_enhanced.wav");

% Energy match to MVDR output
y_model = y_model / rms(y_model) * rms(x_mono);


%% ================= ALIGN SIGNALS =================
Lmin = min([length(s_clean), length(x_mono), ...
            length(y_mvdr_yours), length(y_mvdr_taki), ...
            length(y_mvdr_matlab), length(y_model)]);

s  = s_clean(1:Lmin);
x0 = x_mono(1:Lmin);
% ---------- Align YOUR MVDR ----------
[y_mvdr_yours, lag1] = alignsignals(y_mvdr_yours, s);

% ---------- Align TAKI MVDR ----------
[y_mvdr_taki, lag2] = alignsignals(y_mvdr_taki, s);

% ---------- Align ML Enhacer output ----------
[y_model_aligned, lag3] = alignsignals(y_model, y_mvdr_yours);



y1 = y_mvdr_yours(1:Lmin);
y2 = y_mvdr_taki(1:Lmin);
y3 = real(y_mvdr_matlab(1:Lmin));
y4 = y_model_aligned(1:Lmin);


fprintf('\n\nAlignment lag (mean): Yours=%d, Taki=%d\n, RNNoise=%d\n',  mean(lag1), mean(lag2), mean(lag3));


%% ================= METRICS =================

% ================= SI-SDR =================
sisdr_mic   = si_sdr(x0, s);
sisdr_yours = si_sdr(y1, s);
sisdr_taki  = si_sdr(y2, s);
sisdr_matlab = si_sdr(y3, s);
sisdr_model = si_sdr(y4, s);

% ================= STOI =================
stoi_mic   = stoi(s, x0, fs);
stoi_yours = stoi(s, y1, fs);
stoi_taki  = stoi(s, y2, fs);
stoi_matlab = stoi(s, y3, fs);
stoi_model = stoi(s, y4, fs);

% ================= OSINR =================
% residuals
res_mic   = x0 - s;
res_yours = y1 - s;
res_taki  = y2 - s;
res_matlab = y3 - s;
res_model = y4 - s;
% 
% osinr_mic   = 10*log10( sum(s.^2) / (sum(res_mic.^2)   + 1e-12) );
% osinr_yours = 10*log10( sum(s.^2) / (sum(res_yours.^2) + 1e-12) );
% osinr_taki  = 10*log10( sum(s.^2) / (sum(res_taki.^2)  + 1e-12) );
% osinr_matlab = 10*log10( sum(s.^2) / (sum(res_matlab.^2) + 1e-12) );
% osinr_model = 10*log10( sum(s.^2) / (sum(res_model.^2) + 1e-12) );

% ================= ViSQOL =================
[visqol_mic,ftable_mic,ttable_mic] = visqol(x0,s,fs,mode='speech', OutputMetric="MOS and NSIM");
[visqol_yours,ftable_yours,ttable_yours] = visqol(y1,s,fs,mode='speech', OutputMetric="MOS and NSIM");
[visqol_taki,ftable_taki,ttable_taki] = visqol(y2,s,fs,mode='speech', OutputMetric="MOS and NSIM");
[visqol_matlab,ftable_matlab,ttable_matlab] = visqol(y3,s,fs,mode='speech', OutputMetric="MOS and NSIM");
[visqol_model,ftable_model,ttable_model] = visqol(y4,s,fs,mode='speech', OutputMetric="MOS and NSIM");


%% ================= DISPLAY =================
fprintf('\n===== FAIR MVDR COMPARISON =====\n');

fprintf('\nSI-SDR (dB):\n');
fprintf('Mic:     %.2f\n', sisdr_mic);
fprintf('Yours:   %.2f\n', sisdr_yours);
fprintf('Taki:    %.2f\n', sisdr_taki);
fprintf('MATLAB:  %.2f\n', sisdr_matlab);
fprintf('Model:  %.2f\n', sisdr_model);
% fprintf('Model (ref: MVDR output):  %.2f\n', sisdr_model);

fprintf('\nSTOI:\n');
fprintf('Mic:     %.3f\n', stoi_mic);
fprintf('Yours:   %.3f\n', stoi_yours);
fprintf('Taki:    %.3f\n', stoi_taki);
fprintf('MATLAB:  %.3f\n', stoi_matlab);
fprintf('Model:  %.3f\n', stoi_model);
% 
% fprintf('\nOSINR (dB):\n');
% fprintf('Mic:     %.2f\n', osinr_mic);
% fprintf('Yours:   %.2f\n', osinr_yours);
% fprintf('Taki:    %.2f\n', osinr_taki);
% fprintf('MATLAB:  %.2f\n', osinr_matlab);
% fprintf('Model:  %.2f\n', osinr_model);

fprintf('\nViSQOL (MOS -->[1,5], (NSIM) --> [-1,1]):\n');
fprintf('Mic:     MOS = %.2f, NSIM = %.2f\n', visqol_mic(1),visqol_mic(2));
fprintf('Yours:   MOS = %.2f, NSIM = %.2f\n', visqol_yours(1),visqol_yours(2));
fprintf('Taki:    MOS = %.2f, NSIM = %.2f\n', visqol_taki(1),visqol_taki(2));
fprintf('MATLAB:  MOS = %.2f, NSIM = %.2f\n', visqol_matlab(1),visqol_matlab(2));
fprintf('Model:   MOS = %.2f, NSIM = %.2f\n\n\n', visqol_model(1),visqol_model(2));







%% $$$$$$$$$$$$$ save outputs $$$$$$$$$$$$$$$$$$$$$$$
audiowrite('../Test_output/COMPARE_V4_mvdr_mic1_output.wav', x0, fs);
audiowrite('../Test_output/COMPARE_V4_mvdr_emon_output.wav', y_mvdr_yours, fs);
% audiowrite('../Test_output/COMPARE_V4_mvdr_taki_output.wav', y_mvdr_taki, fs);
% audiowrite('../Test_output/COMPARE_V4_mvdr_matlab_output.wav', real(y_mvdr_matlab), fs);
audiowrite('../Test_output/COMPARE_V4_model_output.wav', y_model, fs);

audiowrite('../Test_output/COMPARE_V4_mixture.wav', x, fs);

