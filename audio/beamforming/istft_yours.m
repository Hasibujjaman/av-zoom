function y = istft_yours(Y, params)
% COMPUTE_ISTFT
% Inverse STFT matching custom stft_multichannel implementation
%
% INPUT:
%   Y      : [numFreqs x numFrames] complex STFT (onesided)
%   params :
%       .win   : analysis window (Nx1)
%       .hop   : hop size
%       .nfft  : FFT length
%
% OUTPUT:
%   y : reconstructed time-domain signal (column vector)

    win  = params.win(:);
    hop  = params.hop;
    nfft = params.nfft;

    N = length(win);
    [numFreqs, numFrames] = size(Y);

    % ---- Sanity checks ----
    expectedFreqs = nfft/2 + 1;
    assert(numFreqs == expectedFreqs, ...
        'compute_istft: STFT dimension mismatch');

    % ---- Output length ----
    sigLen = (numFrames-1)*hop + N;
    y = zeros(sigLen,1);

    % ---- Overlap-add normalization ----
    win_norm = zeros(sigLen,1);

    % ---- Loop over frames ----
    for t = 1:numFrames
        % ----- Reconstruct full spectrum -----
        Y_frame = Y(:,t);

        % Mirror (exclude DC and Nyquist)
        Y_full = [ ...
            Y_frame; ...
            conj(Y_frame(end-1:-1:2)) ...
        ];

        % ----- IFFT -----
        x_frame = real(ifft(Y_full, nfft));

        % ----- Window -----
        x_frame = x_frame(1:N) .* win;

        % ----- Overlap-add -----
        idx = (t-1)*hop + (1:N);
        y(idx) = y(idx) + x_frame;
        win_norm(idx) = win_norm(idx) + win.^2;
    end

    % ---- Normalize to undo window overlap ----
    nonzero = win_norm > 1e-8;
    y(nonzero) = y(nonzero) ./ win_norm(nonzero);

end
