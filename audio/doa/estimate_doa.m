function theta = estimate_doa(X_frame, fs, mic_dist)

    c = 343; % speed of sound

    X1 = X_frame(:,1);
    X2 = X_frame(:,2);

    tau = gcc_phat(X1, X2, fs);

    % Clamp physically valid delay
    tau_max = mic_dist / c;
    tau = max(min(tau, tau_max), -tau_max);

    theta = asind(c * tau / mic_dist);
end
