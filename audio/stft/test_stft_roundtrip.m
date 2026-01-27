clc; clear;

fs = 16000;

% Original signal
t = (0:1/fs:2).';
x = 0.6*sin(2*pi*440*t) + 0.4*randn(length(t),1);

% STFT parameters
frame_len = round(0.02 * fs);
hop_len   = round(0.01 * fs);

% --- PAD signal to full frames ---
pad_len = frame_len;
x_pad = [zeros(pad_len,1); x; zeros(pad_len,1)];

% STFT
[X, params] = compute_stft(x_pad, fs);

% iSTFT
x_rec_pad = compute_istft(X, params);

% --- REMOVE padding ---
x_rec = x_rec_pad(pad_len+1 : pad_len+length(x));

% Error
err = x - x_rec;
snr_val = 10*log10(sum(x.^2) / sum(err.^2));

fprintf('Reconstruction SNR: %.2f dB\n', snr_val);



% Plot
figure;
subplot(2,1,1);
plot(x); title('Original Signal');

subplot(2,1,2);
plot(x_rec); title('Reconstructed Signal');
