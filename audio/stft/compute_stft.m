function [X, stft_params] = compute_stft(x, fs)
 
     % --- Parameters ---
     frame_len = round(0.02 * fs);   % 320 samples for 20 ms @16kHz
     hop_len   = round(0.01 * fs);   % 10 ms
     nfft      = 512;
     window    = hann(frame_len, 'periodic');
     % window    = sqrt(hann(frame_len, 'periodic'));


    % --- STFT ---
    X = stft(x, ...
        'Window', window, ...
        'OverlapLength', frame_len - hop_len, ...
        'FFTLength', nfft);

    % --- Store params for inverse ---
    stft_params.fs        = fs;
    stft_params.frameLen = frame_len;
    stft_params.hopLen   = hop_len;
    stft_params.nfft     = nfft;
    stft_params.window  = window;
end

