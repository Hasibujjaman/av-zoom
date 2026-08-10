function a = steering_vector(freq, mic_pos, azimuth_deg)
% MATLAB convention: azimuth 0°=endfire (along array), 90°=broadside (perpendicular)

    c = 340;
    az = deg2rad(azimuth_deg);

    % Time delay for each microphone
    tau = mic_pos * cos(az) / c;

    % Steering vector
    a = exp(-1j * 2*pi * freq .* tau);
end
