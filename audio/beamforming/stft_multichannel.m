function X = stft_multichannel(x, window, hop, nfft)

[T, M] = size(x);
N = length(window);
numFrames = floor((T - N) / hop) + 1;
numFreqs = nfft/2 + 1;

X = zeros(numFreqs, numFrames, M);

for m = 1:M
    for n = 1:numFrames
        idx = (n-1)*hop + (1:N);
        frame = x(idx, m) .* window;
        Xfull = fft(frame, nfft);
        X(:, n, m) = Xfull(1:numFreqs);
    end
end

end
