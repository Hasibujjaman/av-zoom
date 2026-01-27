%% ===============================
%  MVDR END-TO-END TEST SCRIPT
%  Uses ONLY your implementation
% ===============================
clear;
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');


%% -------- Load audio --------
[target, fs] = audioread('male_clean_15s.wav');
[interf, fs2] = audioread('female_piano_14s.wav');
assert(fs == fs2);

L = min(length(target), length(interf));
target = target(1:L);
interf = interf(1:L);

%% -------- Geometry --------
c = 340;
d = 0.08;                         % 8 cm spacing
micPos = [-d/2; d/2];             % x-axis (meters)

theta_target = 0;                 % broadside
theta_target_test = 0;
theta_interf = 40;                % to the right
SNR_dB = Inf;

apply_delay = @(x, tau) interp1( ...
    (0:length(x)-1)', x, ...
    (0:length(x)-1)' - tau*fs, ...
    'linear', 0);

% Plane-wave model
make_multichannel = @(sig, theta) ...
    [ apply_delay(sig, micPos(1)*sin(deg2rad(theta))/c), ...
      apply_delay(sig, micPos(2)*sin(deg2rad(theta))/c) ];

target_mc = make_multichannel(target, theta_target);
interf_mc = make_multichannel(interf, theta_interf);

mixture = target_mc + interf_mc;

mixture = create_mixture( target, interf, ...
        theta_target, theta_interf, fs, ...
        SNR_dB, c, d);

% 4. Visualization & Verification

% Plot Time Domain
t = (0:L-1)' / fs;

figure('Name', 'Mixture Channels', 'Color', 'w');
subplot(2,1,1);
plot(t, mixture(:,1));
title('Channel 1 (Left Mic)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t, mixture(:,2));
title('Channel 2 (Right Mic)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

% Plot Spectrogram (Optional check for white noise floor)
figure('Name', 'Spectrogram Channel 1', 'Color', 'w');
spectrogram(mixture(:,1), 256, 128, 512, fs, 'yaxis');
title(sprintf('Spectrogram (Channel 1) - SNR: %d dB', SNR_dB));


disp('Done.');

% Write mixute audio
audiowrite('../Test_output/FIX_MVDR_V2_mixture_ch1.wav', mixture(:,1), fs);
% audiowrite('FIX_MVDR_V2_mixture_ch2.wav', mixture(:,2), fs);

%% -------- STFT params --------
N = 256;
hop = 128;
nfft = 512;
window = sqrt(hann(N,'periodic'));

X = stft_multichannel(mixture, window, hop, nfft);
[numFreqs, numFrames, numMics] = size(X);

freqs = (0:numFreqs-1)' * fs / nfft;

%% -------- Covariance --------
Rxx = init_covariance(numFreqs, numMics);
alpha = 0.98;

for n = 1:numFrames
    X_frame = squeeze(X(:,n,:));   % [freq x mic]
    Rxx = update_covariance(Rxx, X_frame, alpha);
end

%% -------- Steering vector --------
dvec = compute_steering_vector(theta_target_test, freqs, micPos, c);

%% -------- MVDR weights --------
delta = 1e-3;
W = compute_mvdr_weights(Rxx, dvec, delta);

%% -------- Apply MVDR --------
Y = apply_mvdr(X, W);

%% -------- ISTFT --------
y = istft_single_channel(Y, window, hop, nfft, L);
audiowrite('../Test_output/FIX_MVDR_V2_mvdr_output.wav', y, fs);

%% ===============================
%          METRICS
% ===============================

% Remove DC
target = target - mean(target);
y = y - mean(y);
x_ref = mixture(:,1) - mean(mixture(:,1));

%% ---- SI-SDR ----
si_sdr = @(s, shat) ...
    10*log10( sum(((s'*shat)/(s'*s)*s).^2) / ...
              sum((shat - (s'*shat)/(s'*s)*s).^2) );

si_sdr_in = si_sdr(target, x_ref);
si_sdr_out = si_sdr(target, y);

%% ---- OSINR ----
proj = (target'*y)/(target'*target) * target;
noise = y - proj;
osinr_out = 10*log10(sum(proj.^2)/sum(noise.^2));

proj_in = (target'*x_ref)/(target'*target) * target;
noise_in = x_ref - proj_in;
osinr_in = 10*log10(sum(proj_in.^2)/sum(noise_in.^2));

%% ---- STOI ----
stoi_in = stoi(x_ref, target, fs);
stoi_out = stoi(y, target, fs);

%% -------- Display --------
fprintf('\n===== MVDR RESULTS =====\n');
fprintf('SI-SDR  in : %.2f dB\n', si_sdr_in);
fprintf('SI-SDR out : %.2f dB\n', si_sdr_out);
fprintf('Gain       : %.2f dB\n\n', si_sdr_out - si_sdr_in);

fprintf('OSINR  in : %.2f dB\n', osinr_in);
fprintf('OSINR out : %.2f dB\n', osinr_out);
fprintf('Gain      : %.2f dB\n\n', osinr_out - osinr_in);

fprintf('STOI  in : %.3f\n', stoi_in);
fprintf('STOI out : %.3f\n', stoi_out);
