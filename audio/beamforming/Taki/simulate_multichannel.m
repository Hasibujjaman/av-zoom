function X = simulate_multichannel(noisy, fs, mic_pos, theta_deg)

    c = 343;
    theta = deg2rad(theta_deg);

    delays = mic_pos * sin(theta) / c;
    M = length(mic_pos);

    X = zeros(length(noisy), M);

    for m = 1:M
        X(:,m) = fractional_delay(noisy, delays(m), fs);
    end
end
