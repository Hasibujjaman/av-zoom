function W = compute_mvdr_weights(Rxx, d, delta, subtract_noise_floor)
% COMPUTE_MVDR_WEIGHTS  Frequency-domain MVDR beamformer weights.
%
% Rxx   : [numMics x numMics x numFreqs] spatial covariance
% d     : [numMics x numFreqs] steering vector
% delta : diagonal loading factor (e.g., 1e-4)
% subtract_noise_floor : (optional, default false) if true, subtract the
%         minimum eigenvalue (isotropic noise estimate) from Rxx per bin
%         before computing weights. This removes spatially-white noise and
%         restores directional interference structure. Critical for
%         performance when additive white/diffuse noise is present.

if nargin < 4, subtract_noise_floor = false; end

[numMics, ~, numFreqs] = size(Rxx);
W = zeros(numMics, numFreqs);

for k = 1:numFreqs
    R = Rxx(:,:,k);

    % ---- Noise floor subtraction (per-bin) ----
    if subtract_noise_floor
        eigvals = eig(R);
        lambda_min = max(real(min(eigvals)), 0);
        % Subtract 95% of isotropic component; keep 5% to avoid singularity
        R = R - 0.95 * lambda_min * eye(numMics);
    end

    % ---- Diagonal loading ----
    R = R + delta * trace(R)/numMics * eye(numMics);
    dk = d(:,k);

    W(:,k) = R \ dk / (dk' * (R \ dk));
end

end
