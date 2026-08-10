function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_reverberant_mixture(target, interf, theta_target, theta_interf, ...
                               fs, SNR_dB, c, d, varargin)
%CREATE_REVERBERANT_MIXTURE Create reverberant mixture with specified acoustic conditions
%
%   [mixture, target_mc, interf_mc, noise_mc] = ...
%       create_reverberant_mixture(target, interf, theta_target, theta_interf, ...
%                                  fs, SNR_dB, c, d, options)
%
%   INPUTS:
%   - target: Target source signal (mono)
%   - interf: Interference source signal (mono)
%   - theta_target: Azimuth of target source (degrees)
%   - theta_interf: Azimuth of interference source (degrees)
%   - fs: Sampling rate (Hz)
%   - SNR_dB: Signal-to-Noise Ratio (dB)
%   - c: Speed of sound (m/s)
%   - d: Microphone spacing (m)
%   - options: (Optional) Name-value pairs:
%       * 'RT60': Reverberation time (default: 0.5 s)
%       * 'RoomDims': Room dimensions [Lx, Ly, Lz] (default: [4.9, 4.9, 4.9])
%       * 'MicPositions': Microphone positions (N x 3 matrix, default: array at room center)
%       * 'ReflectionCoefficient': Wall reflection coefficient (default: 0.75)
%       * 'PlotResults': Flag to plot results (default: false)
%       * 'SaveAudio': Flag to save audio files (default: false)
%       * 'Verbose': Flag for progress output (default: true)
%
%   OUTPUTS:
%   - mixture: Final mixture signal (N-channel)
%   - target_mc: Target signal at each microphone
%   - interf_mc: Interference signal at each microphone
%   - noise_mc: Noise signal at each microphone
%
%   EXAMPLE:
%   [mix, target_mc, interf_mc, noise] = create_reverberant_mixture(...
%       target_sig, interf_sig, 0, 40, 16000, 5, 340, 0.08, ...
%       'RT60', 0.5, 'PlotResults', true);

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'RT60', 0.5, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'RoomDims', [4.9, 4.9, 4.9], @(x) isnumeric(x) && numel(x) == 3);
    addParameter(p, 'MicPositions', [], @(x) ismatrix(x));
    addParameter(p, 'ReflectionCoefficient', 0.75, @(x) isnumeric(x) && x >= 0 && x <= 1);
    addParameter(p, 'PlotResults', false, @islogical);
    addParameter(p, 'SaveAudio', false, @islogical);
    addParameter(p, 'Verbose', false, @islogical);
    parse(p, varargin{:});
    
    % Extract parameters
    RT60 = p.Results.RT60;
    room_dims = p.Results.RoomDims;
    R_wall = p.Results.ReflectionCoefficient;
    plot_results = p.Results.PlotResults;
    save_audio = p.Results.SaveAudio;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== Creating Reverberant Mixture ===\n');
        fprintf('RT60: %.2f s | SNR: %.1f dB | SIR: 0 dB\n', RT60, SNR_dB);
    end
    
    %% 1. Set up geometry
    % Define microphone array (if not provided)
    if isempty(p.Results.MicPositions)
        % Center of room
        room_center = room_dims / 2;
        % 2-microphone linear array with specified spacing
        mic_pos = [room_center(1) - d/2, room_center(2), 1.5;  % Mic 1
                   room_center(1) + d/2, room_center(2), 1.5]; % Mic 2
    else
        mic_pos = p.Results.MicPositions;
    end
    
    % Convert azimuth angles to positions
    % Assuming sources are at same height as microphones (1.5m)
    % and at a fixed distance of 1.5m from array center
    array_center = mean(mic_pos, 1);
    source_distance = 1.5; % meters
    
    % MATLAB convention: azimuth from +X axis, so x=r*cos(az), y=r*sin(az)
    src1_pos = [array_center(1) + source_distance * cosd(theta_target), ...
                array_center(2) + source_distance * sind(theta_target), ...
                1.5];
    
    src2_pos = [array_center(1) + source_distance * cosd(theta_interf), ...
                array_center(2) + source_distance * sind(theta_interf), ...
                1.5];
    
    if verbose
        fprintf('Microphone positions:\n');
        fprintf('  Mic 1: [%.2f, %.2f, %.2f] m\n', mic_pos(1,:));
        fprintf('  Mic 2: [%.2f, %.2f, %.2f] m\n', mic_pos(2,:));
        fprintf('Source positions:\n');
        fprintf('  Target (θ=%.0f°): [%.2f, %.2f, %.2f] m\n', theta_target, src1_pos);
        fprintf('  Interference (θ=%.0f°): [%.2f, %.2f, %.2f] m\n', theta_interf, src2_pos);
    end
    
    %% 2. Generate Room Impulse Responses using Image Source Method
    if verbose, fprintf('\nGenerating room impulse responses...\n'); end
    
    % Generate RIR for target source
    rir_target = acousticRoomResponse_ISM(room_dims, src1_pos, mic_pos, R_wall, fs, RT60);
    
    % Generate RIR for interference source
    rir_interf = acousticRoomResponse_ISM(room_dims, src2_pos, mic_pos, R_wall, fs, RT60);
    
    %% 3. Verify RT60 (optional)
    if verbose
        rt60_target = estimate_rt60_schroeder(rir_target(:,1), fs);
        rt60_interf = estimate_rt60_schroeder(rir_interf(:,1), fs);
        fprintf('RT60 achieved: Target RIR = %.3f s, Interf RIR = %.3f s\n', ...
                rt60_target, rt60_interf);
    end
    
    %% 4. Apply RIRs to signals
    if verbose, fprintf('Applying reverberation...\n'); end
    
    % Ensure signals are same length
    L = min(length(target), length(interf));
    target = target(1:L);
    interf = interf(1:L);
    
    % Convolve each channel
    n_mics = size(mic_pos, 1);
    target_mc = zeros(L + size(rir_target,1) - 1, n_mics);
    interf_mc = zeros(L + size(rir_interf,1) - 1, n_mics);
    
    for m = 1:n_mics
        target_mc(:,m) = conv(target, rir_target(:,m));
        interf_mc(:,m) = conv(interf, rir_interf(:,m));
    end
    
    % Trim to original length
    target_mc = target_mc(1:L, :);
    interf_mc = interf_mc(1:L, :);
    
    %% 5. Adjust SIR to 0 dB
    if verbose, fprintf('Adjusting SIR to 0 dB...\n'); end
    
    % Calculate average power across all microphones
    P_target = mean(target_mc(:).^2);
    P_interf = mean(interf_mc(:).^2);
    
    % Scale interference to achieve 0 dB SIR
    scale_interf = sqrt(P_target / P_interf);
    interf_mc = interf_mc * scale_interf;
    
    %% 6. Create clean mixture
    mixture_clean = target_mc + interf_mc;
    
    %% 7. Add white Gaussian noise for specified SNR
    if verbose, fprintf('Adding noise (SNR = %.1f dB)...\n', SNR_dB); end
    
    mixture = zeros(size(mixture_clean));
    noise_mc = zeros(size(mixture_clean));
    
    % Convert SNR from dB to linear
    SNR_linear = 10^(SNR_dB/10);
    
    for m = 1:n_mics
        % Calculate signal power
        P_signal = mean(mixture_clean(:,m).^2);
        
        % Calculate required noise power
        P_noise = P_signal / SNR_linear;
        
        % Generate white Gaussian noise
        noise = sqrt(P_noise) * randn(size(mixture_clean(:,m)));
        
        % Add noise to mixture
        mixture(:,m) = mixture_clean(:,m) + noise;
        noise_mc(:,m) = noise;
    end
    
    %% 8. Final normalization (prevent clipping)
    peak_val = max(abs(mixture(:)));
    if peak_val > 0
        mixture = 0.99 * mixture / peak_val;
        target_mc = 0.99 * target_mc / peak_val;
        interf_mc = 0.99 * interf_mc / peak_val;
        noise_mc = 0.99 * noise_mc / peak_val;
    end
    
    %% 9. Save audio files if requested
    if save_audio
        if verbose, fprintf('Saving audio files...\n'); end
        audiowrite('mixture.wav', mixture, fs);
        audiowrite('target_reverb.wav', target_mc, fs);
        audiowrite('interf_reverb.wav', interf_mc, fs);
        audiowrite('noise.wav', noise_mc, fs);
    end
    



    %% 10. Plot results if requested
    if plot_results
        plot_simulation_results(mixture, target_mc, interf_mc, noise_mc, ...
                                rir_target, rir_interf, fs, RT60, ...
                                room_dims, mic_pos, src1_pos, src2_pos);
    end
    
    if verbose
        fprintf('\n=== Mixture Creation Complete ===\n');
        % Verify conditions
        P_target_final = mean(target_mc(:).^2);
        P_interf_final = mean(interf_mc(:).^2);
        SIR_actual = 10*log10(P_target_final / P_interf_final);
        
        P_signal_noisy = mean(mixture(:).^2);
        P_noise_actual = mean(noise_mc(:).^2);
        SNR_actual = 10*log10(P_signal_noisy / P_noise_actual);
        
        fprintf('Signal Conditions:\n');
        fprintf('  Achieved SIR: %.2f dB (Target: 0 dB)\n', SIR_actual);
        fprintf('  Achieved SNR: %.2f dB (Target: %.1f dB)\n', SNR_actual, SNR_dB);
        fprintf('  Room RT60: %.3f s (Target: %.2f s)\n', rt60_target, RT60);
    end
end









%% Supporting Functions
function h = acousticRoomResponse_ISM(room_dims, src_pos, mic_pos, R, fs, RT60)
    % Image Source Method for shoebox rooms
    % Simplified version for computational efficiency
    
    c = 340; % Speed of sound (m/s)
    
    % Calculate maximum order based on RT60
    max_time = RT60 * 1.5; % Include extra time
    max_dist = c * max_time;
    min_wall_dist = min(room_dims);
    max_order = ceil(max_dist / min_wall_dist);
    max_order = min(max_order, 10); % Cap for reasonable computation
    
    n_mics = size(mic_pos, 1);
    n_samples = ceil(max_time * fs);
    h = zeros(n_samples, n_mics);
    
    % Generate all image source indices within max_order
    [nx, ny, nz] = ndgrid(-max_order:max_order, -max_order:max_order, -max_order:max_order);
    nx = nx(:); ny = ny(:); nz = nz(:);
    
    % Calculate order for each image
    orders = abs(nx) + abs(ny) + abs(nz);
    valid_idx = orders <= max_order;
    nx = nx(valid_idx); ny = ny(valid_idx); nz = nz(valid_idx);
    orders = orders(valid_idx);
    
    n_images = length(nx);
    
    % Process each image source
    for idx = 1:n_images
        order = orders(idx);
        gain = R^order;
        
        % Generate image positions for all 8 permutations
        for px = [-1, 1]
            for py = [-1, 1]
                for pz = [-1, 1]
                    img_x = 2*nx(idx)*room_dims(1) + px*src_pos(1);
                    img_y = 2*ny(idx)*room_dims(2) + py*src_pos(2);
                    img_z = 2*nz(idx)*room_dims(3) + pz*src_pos(3);
                    
                    for mic = 1:n_mics
                        % Distance from image to microphone
                        dx = img_x - mic_pos(mic, 1);
                        dy = img_y - mic_pos(mic, 2);
                        dz = img_z - mic_pos(mic, 3);
                        d = sqrt(dx^2 + dy^2 + dz^2);
                        
                        if d > 0 && d <= max_dist
                            % Spherical spreading loss
                            amp = gain / (4 * pi * d);
                            
                            % Delay in samples
                            delay = round((d / c) * fs);
                            
                            if delay > 0 && delay <= n_samples
                                h(delay, mic) = h(delay, mic) + amp;
                            end
                        end
                    end
                end
            end
        end
    end
    
    % Apply high-pass filter to remove DC and normalize
    [b, a] = butter(2, 50/(fs/2), 'high');
    for mic = 1:n_mics
        h(:, mic) = filter(b, a, h(:, mic));
        if max(abs(h(:, mic))) > 0
            h(:, mic) = h(:, mic) / max(abs(h(:, mic))) * 0.7;
        end
    end
end

function rt60 = estimate_rt60_schroeder(ir, fs)
    % Estimate RT60 using Schroeder integration (T20 method)
    
    energy = flipud(cumsum(flipud(ir.^2)));
    if max(energy) == 0
        rt60 = NaN;
        return;
    end
    
    energy = energy / max(energy);
    energy_db = 10 * log10(energy + eps);
    
    % Find -5 dB and -25 dB points (T20 method)
    idx_5db = find(energy_db <= -5, 1, 'first');
    idx_25db = find(energy_db <= -25, 1, 'first');
    
    if isempty(idx_5db) || isempty(idx_25db) || idx_25db <= idx_5db
        rt60 = NaN;
        return;
    end
    
    t_5db = (idx_5db-1)/fs;
    t_25db = (idx_25db-1)/fs;
    
    % RT60 = 3 * T20 (extrapolate from 20 dB decay to 60 dB)
    rt60 = 3 * (t_25db - t_5db);
end

function plot_simulation_results(mixture, target_mc, interf_mc, noise_mc, ...
                                 rir_target, rir_interf, fs, RT60, ...
                                 room_dims, mic_pos, src1_pos, src2_pos)
    % Create comprehensive visualization of simulation results
    
    figure('Position', [100, 100, 1600, 900]);
    
    % Time vector
    t_mix = (0:size(mixture,1)-1)/fs;
    t_rir = (0:size(rir_target,1)-1)/fs;
    
    % 1. Mixture signals (time domain)
    subplot(3,4,1:2);
    plot(t_mix, mixture(:,1), 'b', 'LineWidth', 1);
    hold on;
    plot(t_mix, mixture(:,2), 'r--', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Mixture Signals (Time Domain)');
    legend('Mic 1', 'Mic 2', 'Location', 'best');
    grid on;
    xlim([0, min(2, t_mix(end))]);
    
    % 2. Impulse responses
    subplot(3,4,3);
    plot(t_rir, rir_target(:,1), 'b', 'LineWidth', 1);
    hold on;
    plot(t_rir, rir_target(:,2), 'r--', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Target Source RIR');
    grid on;
    xlim([0, 0.5]);
    
    subplot(3,4,4);
    plot(t_rir, rir_interf(:,1), 'b', 'LineWidth', 1);
    hold on;
    plot(t_rir, rir_interf(:,2), 'r--', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Interference Source RIR');
    grid on;
    xlim([0, 0.5]);
    
    % 3. Energy decay curves
    subplot(3,4,5);
    energy = flipud(cumsum(flipud(rir_target(:,1).^2)));
    energy = energy / max(energy);
    energy_db = 10*log10(energy + eps);
    plot(t_rir, energy_db, 'b', 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel('Energy (dB)');
    title('Energy Decay (Target RIR)');
    grid on;
    hold on;
    plot([0, t_rir(end)], [-5, -5], 'k--');
    plot([0, t_rir(end)], [-25, -25], 'k--');
    ylim([-60, 0]);
    xlim([0, RT60*1.2]);
    
    % 4. Room layout (top view)
    subplot(3,4,6);
    plot(mic_pos(:,1), mic_pos(:,2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    hold on;
    plot(src1_pos(1), src1_pos(2), 'bs', 'MarkerSize', 12, 'MarkerFaceColor', 'b');
    plot(src2_pos(1), src2_pos(2), 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
    
    % Draw room boundaries
    rectangle('Position', [0, 0, room_dims(1), room_dims(2)], ...
              'EdgeColor', 'k', 'LineWidth', 2);
    
    xlabel('X (m)'); ylabel('Y (m)');
    title('Room Layout (Top View)');
    legend('Mics', 'Target', 'Interf', 'Location', 'best');
    grid on;
    axis equal;
    xlim([0, room_dims(1)]); ylim([0, room_dims(2)]);
    
    % 5. Spectrograms
    subplot(3,4,7:8);
    window = 256; noverlap = 220; nfft = 256;
    spectrogram(mixture(:,1), window, noverlap, nfft, fs, 'yaxis');
    title('Mic 1 Spectrogram');
    colorbar;
    
    % 6. Signal statistics
    subplot(3,4,9);
    % Calculate powers
    P_target = mean(target_mc(:).^2);
    P_interf = mean(interf_mc(:).^2);
    P_noise = mean(noise_mc(:).^2);
    P_mix = mean(mixture(:).^2);
    
    bar_data = [P_target, P_interf, P_noise, P_mix];
    bar(log10(bar_data + eps));
    ylabel('Log Power (dB)');
    title('Signal Power Distribution');
    set(gca, 'XTickLabel', {'Target', 'Interf', 'Noise', 'Mixture'});
    grid on;
    
    % 7. Cross-correlation between microphones
    subplot(3,4,10);
    [xc, lags] = xcorr(mixture(:,1), mixture(:,2), 100, 'normalized');
    plot(lags/fs*1000, xc, 'b', 'LineWidth', 1.5); % Convert to ms
    xlabel('Time Lag (ms)'); ylabel('Correlation');
    title('Cross-Correlation (Mic1 vs Mic2)');
    grid on;
    xlim([-20, 20]);
    
    % 8. Histogram of mixture
    subplot(3,4,11);
    histogram(mixture(:,1), 50, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    histogram(mixture(:,2), 50, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    xlabel('Amplitude'); ylabel('Count');
    title('Amplitude Distribution');
    legend('Mic 1', 'Mic 2');
    grid on;
    
    % 9. Summary text
    subplot(3,4,12);
    axis off;
    
    % Calculate metrics
    rt60_target = estimate_rt60_schroeder(rir_target(:,1), fs);
    SIR_actual = 10*log10(P_target / P_interf);
    SNR_actual = 10*log10(P_mix / P_noise);
    
    text_str = {
        sprintf('Simulation Summary:');
        sprintf('Room: %.1f x %.1f x %.1f m', room_dims);
        sprintf('RT60: %.3f s (Target: %.2f s)', rt60_target, RT60);
        sprintf('SIR: %.2f dB (Target: 0 dB)', SIR_actual);
        sprintf('SNR: %.2f dB', SNR_actual);
        sprintf('Fs: %d Hz', fs);
        sprintf('c: %.0f m/s', 340);
        sprintf('Mic Spacing: %.2f m', norm(mic_pos(1,:)-mic_pos(2,:)));
        sprintf('Target θ: %.0f°', atan2d(src1_pos(1)-mean(mic_pos(:,1)), ...
                                         src1_pos(2)-mean(mic_pos(:,2))));
        sprintf('Interf θ: %.0f°', atan2d(src2_pos(1)-mean(mic_pos(:,1)), ...
                                         src2_pos(2)-mean(mic_pos(:,2))))
    };
    
    text(0.1, 0.9, text_str, 'VerticalAlignment', 'top', ...
         'FontSize', 10, 'FontName', 'FixedWidth');
    
    sgtitle(sprintf('Reverberant Mixture Simulation (RT60 ≈ %.2f s)', RT60));
end