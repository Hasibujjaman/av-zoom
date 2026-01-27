addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

clc; clear;

fs = 16000;
x = randn(fs*2, 2);   % 2 sec, 2 mics

N = 512;
hop = 256;
nfft = 512;
window = hann(N,'symmetric');

% Padding (from Step 2.1)
pad = N;
x_pad = [zeros(pad,2); x; zeros(pad,2)];

% STFT
X = stft_multichannel(x_pad, window, hop, nfft);

[numFreqs, numFrames, numMics] = size(X);

% Initialize covariance
Rxx = init_covariance(numFreqs, numMics);

alpha = 0.98;

% Track one frequency bin for inspection
k0 = round(numFreqs/4);  % mid-low frequency

trace_vals = zeros(numFrames,1);

for n = 1:numFrames
    X_frame = squeeze(X(:,n,:));   % [numFreqs x numMics]
    Rxx = update_covariance(Rxx, X_frame, alpha);
    trace_vals(n) = trace(Rxx(:,:,k0));
end

figure;
plot(trace_vals);
xlabel('Frame');
ylabel('Trace of Rxx');
title('Covariance Energy Evolution');
grid on;


%test 2
norm(Rxx(:,:,k0) - Rxx(:,:,k0)', 'fro') % should be ~0

%test 3
eig(Rxx(:,:,k0))  % should be +ve
