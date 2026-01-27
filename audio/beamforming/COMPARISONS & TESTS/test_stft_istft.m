addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

fs = 16000;
x = randn(fs*2, 2);

N = 512;
hop = 256;
nfft = 512;
window = hann(N,'symmetric');

pad = N;
x_pad = [zeros(pad,2); x; zeros(pad,2)];

X = stft_multichannel(x_pad, window, hop, nfft);
y_pad = istft_single_channel(X(:,:,1), window, hop, nfft, size(x_pad,1));

y = y_pad(pad+1 : pad+size(x,1));

snr_val = 20*log10(norm(x(:,1)) / norm(x(:,1) - y));
disp(['STFT Reconstruction SNR: ', num2str(snr_val), ' dB']);
