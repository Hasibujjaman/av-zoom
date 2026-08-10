function Y_pf = mvdr_postfilter(X, W, dvec)
% MVDR_POSTFILTER  Apply MVDR beamforming + Zelinski-style Wiener post-filter.
%
% The MVDR nulls directional interference. The post-filter then estimates
% and suppresses residual diffuse/white noise using inter-channel coherence.
%
% Inputs:
%   X    : [numFreqs x numFrames x numMics] STFT tensor
%   W    : [numMics x numFreqs] MVDR weights
%   dvec : [numMics x numFreqs] steering vector (for target PSD estimate)
%
% Output:
%   Y_pf : [numFreqs x numFrames] post-filtered STFT output
%
% Reference: Zelinski (1988), Simmer et al. (2001)

[numFreqs, numFrames, numMics] = size(X);

% Step 1: Apply MVDR
Y_mvdr = zeros(numFreqs, numFrames);
for n = 1:numFrames
    for k = 1:numFreqs
        xk = squeeze(X(k,n,:));
        Y_mvdr(k,n) = W(:,k)' * xk;
    end
end

% Step 2: Estimate post-filter gain per TF bin
%
% For each frequency bin, estimate:
%   - Target PSD: from cross-spectral density between mic pairs
%   - Total PSD:  from auto-spectral density
%
% Wiener gain = target_PSD / total_PSD

alpha_pf = 0.92;    % smoothing for PSD estimates
beta     = 0.02;    % spectral floor (prevents musical noise)

Phi_xx  = zeros(numFreqs, 1);   % smoothed auto-PSD (average across mics)
Phi_x1x2 = zeros(numFreqs, 1); % smoothed cross-PSD

G       = ones(numFreqs, 1);    % gain
Y_pf    = zeros(numFreqs, numFrames);

for n = 1:numFrames
    x1 = squeeze(X(:,n,1));   % mic 1
    x2 = squeeze(X(:,n,2));   % mic 2

    % Auto-PSD: average of |X1|^2 and |X2|^2
    auto_psd = 0.5 * (abs(x1).^2 + abs(x2).^2);

    % Cross-PSD: Re{X1 * conj(X2)}
    % For spatially-white noise, cross-PSD ≈ 0 (uncorrelated between mics)
    % For target (broadside), cross-PSD ≈ auto-PSD
    cross_psd = real(x1 .* conj(x2));

    % Smooth
    Phi_xx   = alpha_pf * Phi_xx   + (1 - alpha_pf) * auto_psd;
    Phi_x1x2 = alpha_pf * Phi_x1x2 + (1 - alpha_pf) * cross_psd;

    % Wiener gain: ratio of correlated (target) to total power
    G_raw = Phi_x1x2 ./ (Phi_xx + eps);

    % Clamp to [beta, 1]
    G = max(min(G_raw, 1), beta);

    % Apply gain to MVDR output
    Y_pf(:,n) = G .* Y_mvdr(:,n);
end

end
