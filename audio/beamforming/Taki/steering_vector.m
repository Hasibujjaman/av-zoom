function a = steering_vector(freq, mic_pos, theta_deg)

    c = 340;
    theta = deg2rad(theta_deg);

    % Time delay for each microphone
    tau = mic_pos * sin(theta) / c;

    % Steering vector
    a = exp(-1j * 2*pi * freq .* tau);
end
