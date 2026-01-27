% function y = reconstruct_mvdr_audio(Y, window, hop, nfft, sigLen)
% % Y       : [numFreqs x numFrames]
% % sigLen : desired output length (samples)
% 
% y = istft(Y, ...
%           'Window', window, ...
%           'OverlapLength', length(window)-hop, ...
%           'FFTLength', nfft);
% 
% % Trim to original length
% y = y(1:sigLen);
% end


function y = reconstruct_mvdr_audio(Y_half, window, hop, nfft, sigLen)
% Y_half : [nfft/2+1 x numFrames]

[numFreqs, numFrames] = size(Y_half);

% Rebuild full spectrum (Hermitian symmetry)
Y_full = zeros(nfft, numFrames);

Y_full(1:numFreqs, :) = Y_half;
Y_full(numFreqs+1:end, :) = conj( ...
    Y_half(end-1:-1:2, :) ...
);

% Overlap-add ISTFT (manual)
y = zeros((numFrames-1)*hop + length(window), 1);

win = window(:);

idx = 1;
for n = 1:numFrames
    frame = real(ifft(Y_full(:,n), nfft));
    frame = frame(1:length(window)) .* win;
    y(idx:idx+length(window)-1) = ...
        y(idx:idx+length(window)-1) + frame;
    idx = idx + hop;
end

% Trim to original signal length
y = y(1:sigLen);
end
