function X = simulate_multichannel(noisy, fs, mic_pos, azimuth_deg)
% azimuth_deg: MATLAB convention (0°=endfire, 90°=broadside)

    c = 343;
    az = deg2rad(azimuth_deg);

    delays = mic_pos * cos(az) / c;
    M = length(mic_pos);

    X = zeros(length(noisy), M);

    for m = 1:M
        X(:,m) = fractional_delay(noisy, delays(m), fs);
    end
end
