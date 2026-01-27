clear;
 
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
 addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');

%% =========================
% 1. Load audio
%% =========================
[target, fs] = audioread('male_clean_15s.wav');
[interf, fs2] = audioread('female_piano_14s.wav');

assert(fs == fs2, 'Sampling rates must match');

% Truncate to same length
L = min(length(target), length(interf));
target = target(1:L);
interf = interf(1:L);

%% =========================
% 2. Geometry (YOUR assumptions)
%% =========================
c = 343;                 % speed of sound
d = 0.08;                % 8 cm spacing
micPos = [-d/2; d/2];    % x-axis
theta_target = 0;        % degrees (broadside = +y)
theta_interf = 40;       % degrees to the right

%% =========================
% 3. Plane-wave propagation
%% =========================
apply_delay = @(x, tau) interp1( ...
    (0:length(x)-1)', ...
    x, ...
    (0:length(x)-1)' - tau*fs, ...
    'linear', 0);

plane_wave = @(x, theta) ...
    arrayfun(@(m) ...
        apply_delay(x, micPos(m)*sind(theta)/c), ...
        1:2, 'UniformOutput', false);

% Target
tmp = plane_wave(target, theta_target);
target_mc = [tmp{1}, tmp{2}];

% Interference
tmp = plane_wave(interf, theta_interf);
interf_mc = [tmp{1}, tmp{2}];

%% =========================
% 4. Mixture
%% =========================
mixture = target_mc + interf_mc;
% Normalize ONLY for saving
mix_norm = mixture / max(abs(mixture(:)) + eps);

audiowrite('FIX_MVDR_mixture_ch1.wav', mix_norm(:,1), fs);
audiowrite('FIX_MVDR_mixture_ch2.wav', mix_norm(:,2), fs);

fprintf('Mixture written.\n');

%% =========================
% 5. STFT parameters
%% =========================
nfft = 512;
N = 256;
hop = 128;
win = sqrt(hann(N,'periodic'));

%% =========================
% 6. STFT (multichannel)
%% =========================
[X1, f] = stft(mixture(:,1), fs, ...
    'Window', win, ...
    'OverlapLength', N-hop, ...
    'FFTLength', nfft);

[X2, ~] = stft(mixture(:,2), fs, ...
    'Window', win, ...
    'OverlapLength', N-hop, ...
    'FFTLength', nfft);

% X: [freq x frame x mic]
X = cat(3, X1, X2);

[numFreqs, numFrames, numMics] = size(X);

%% =========================
% 7. Covariance estimation (EMA, mixture)
%% =========================
alpha = 0.98;
Rxx = zeros(numMics, numMics, numFreqs);

for k = 1:numFreqs
    Rxx(:,:,k) = eye(numMics);
end

for n = 1:numFrames
    for k = 1:numFreqs
        xk = squeeze(X(k,n,:));
        Rxx(:,:,k) = alpha * Rxx(:,:,k) + ...
            (1-alpha) * (xk * xk');
    end
end

%% =========================
% 8. Steering vector (broadside = 0°)
%% =========================
dvec = zeros(numMics, numFreqs);

for k = 1:numFreqs
    omega = 2*pi*f(k);
    tau = micPos * sind(theta_target) / c;
    dvec(:,k) = exp(-1j * omega * tau);
end

%% =========================
% 9. MVDR weights
%% =========================
delta = 1e-3;
W = zeros(numMics, numFreqs);

for k = 1:numFreqs
    R = Rxx(:,:,k);
    R = R + delta * trace(R)/numMics * eye(numMics);
    
    W(:,k) = (R \ dvec(:,k)) / ...
        (dvec(:,k)' * (R \ dvec(:,k)));
end

%% =========================
% 10. Apply MVDR
%% =========================
Y = zeros(numFreqs, numFrames);

for n = 1:numFrames
    for k = 1:numFreqs
        Y(k,n) = W(:,k)' * squeeze(X(k,n,:));
    end
end

%% =========================
% 11. ISTFT
%% =========================
% y = istft(Y, fs, ...
%     'Window', win, ...
%     'OverlapLength', N-hop, ...
%     'FFTLength', nfft);
% 
% y = y(1:L);

y = istft(Y, fs, ...
    'Window', win, ...
    'OverlapLength', N-hop, ...
    'FFTLength', nfft);

y = y(:);

% Align lengths safely
Lout = length(y);
Lmin = min(L, Lout);

y = y(1:Lmin);
target = target(1:Lmin);
mixture = mixture(1:Lmin,:);

y_norm = y / max(abs(y) + eps);
audiowrite('FIX_MVDR_mvdr_output.wav', y_norm, fs);


fprintf('MVDR output written.\n');

%% =========================
% 12. Metrics
%% =========================

% ---- SI-SDR ----
si_sdr = @(s, shat) ...
    10*log10( ...
        sum(((shat'*s)/(s'*s)*s).^2) / ...
        sum((shat - (shat'*s)/(s'*s)*s).^2) );

si_sdr_in  = si_sdr(target, mixture(:,1));
si_sdr_out = si_sdr(target, y);

% ---- OSINR ----
proj_out = (y'*target)/(target'*target) * target;
noise_out = y - proj_out;
osinr_out = 10*log10(sum(proj_out.^2)/sum(noise_out.^2));

proj_in = (mixture(:,1)'*target)/(target'*target) * target;
noise_in = mixture(:,1) - proj_in;
osinr_in = 10*log10(sum(proj_in.^2)/sum(noise_in.^2));

% ---- STOI proxy (correlation-based) ----
stoi_in  = corr(target, mixture(:,1));
stoi_out = corr(target, y);

%% =========================
% 13. Display results
%% =========================
fprintf('\n===== RESULTS =====\n');
fprintf('SI-SDR  in  : %.2f dB\n', si_sdr_in);
fprintf('SI-SDR  out : %.2f dB\n', si_sdr_out);
fprintf('OSINR   in  : %.2f dB\n', osinr_in);
fprintf('OSINR   out : %.2f dB\n', osinr_out);
fprintf('STOI   in   : %.3f\n', stoi_in);
fprintf('STOI   out  : %.3f\n', stoi_out);
