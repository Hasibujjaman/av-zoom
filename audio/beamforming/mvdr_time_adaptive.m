function Y = mvdr_time_adaptive(Xf, fs, theta_deg, mic_pos, c, alpha, eps)

[F, T, M] = size(Xf);
freqs = (0:F-1).' * fs / (2*(F-1));   % consistent with compute_stft

Rxx = repmat(eye(M), [1 1 F]);
Y = zeros(F, T);

for t = 1:T
    for f = 2:F   % ❗ skip DC (CRITICAL FIX)

        xf = squeeze(Xf(f,t,:));   % [M x 1]

        % Recursive covariance
        Rxx(:,:,f) = alpha * Rxx(:,:,f) + ...
                     (1-alpha) * (xf * xf');

        R = Rxx(:,:,f) + eps * eye(M);

        % Steering vector (MATCH Taki)
        a = steering_vector(freqs(f), mic_pos, theta_deg);

        % MVDR weights
        w = (R \ a) / (a' * (R \ a));

        Y(f,t) = w' * xf;
    end
end
end
