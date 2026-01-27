 function [X, params] = compute_stft(x, fs)
 
     frame_len = round(0.02 * fs);    %20 ms
     hop_len   = round(0.01 * fs);    %10 ms
     nfft      = 2^nextpow2(frame_len);
 
     win = sqrt(hann(frame_len, 'periodic'));
 
     X = stft(x, ...
         'Window', win, ...
         'OverlapLength', frame_len - hop_len, ...
         'FFTLength', nfft);
 
     params.window   = win;
     params.frameLen = frame_len;
     params.hopLen   = hop_len;
     params.nfft     = nfft;
 end




