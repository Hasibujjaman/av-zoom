function y = istft_single_channel(Y, window, hop, nfft, outLength)

[~, numFrames] = size(Y);
N = length(window);

y = zeros(outLength, 1);
winNorm = zeros(outLength, 1);

for n = 1:numFrames
    idx = (n-1)*hop + (1:N);
    
    % Reconstruct full spectrum
    Yfull = [Y(:,n); conj(Y(end-1:-1:2,n))];
    frame = real(ifft(Yfull, nfft));
    
    y(idx) = y(idx) + frame(1:N) .* window;
    winNorm(idx) = winNorm(idx) + window.^2;
end

% Normalize for COLA
nz = winNorm > eps;
y(nz) = y(nz) ./ winNorm(nz);

end
