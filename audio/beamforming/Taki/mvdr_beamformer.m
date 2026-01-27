function Y = mvdr_beamformer(X, fs, theta_deg, mic_pos)

    M = size(X,2);

    % Compute STFT for each channel explicitly (robust)
    for m = 1:M
        [Xf(:,:,m), params] = compute_stft(X(:,m), fs);
    end

    [F, T, ~] = size(Xf);

    freqs = (0:F-1).' * fs / params.nfft;

    % Initialize covariance matrices safely
    Rxx = repmat(eye(M), [1 1 F]);

    alpha = 0.98;
    eps   = 1e-3;

    Y = zeros(F, T);

    for t = 1:T
        for f = 2:F   % skip DC bin

            xf = squeeze(Xf(f,t,:));

            Rxx(:,:,f) = alpha * Rxx(:,:,f) + ...
                         (1 - alpha) * (xf * xf');

            R = Rxx(:,:,f) + eps * eye(M);

            a = steering_vector(freqs(f), mic_pos, theta_deg);

            w = (R \ a) / (a' * (R \ a));

            Y(f,t) = w' * xf;
        end
    end
end
