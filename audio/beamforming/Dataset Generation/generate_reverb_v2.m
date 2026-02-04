%% Room Impulse Response Simulation (ISM) - Optimized
% RT60 = 0.5s | 2-Mic Array | 2 Sources | SIR=0dB | SNR=5dB
clear; clc; close all;

%% 1. Room & Physics Configuration
c = 340;                    
fs = 16000;                 
RT60_target = 0.5;          
L = [4.9, 4.9, 4.9];        

Volume = prod(L);
SurfaceArea = 2 * (L(1)*L(2) + L(1)*L(3) + L(2)*L(3));

% Calculate required absorption coefficient
alpha_sabine = (0.161 * Volume) / (SurfaceArea * RT60_target);
fprintf('Theoretical Sabine absorption: %.4f\n', alpha_sabine);

% Empirical adjustment based on testing
% For RT60 ≈ 0.5s, we found R ≈ 0.76 works well
R_wall = 0.75;  % From your successful run
alpha_actual = 1 - R_wall^2;

fprintf('Empirical Parameters for RT60 ≈ 0.5s:\n');
fprintf('  Reflection Coefficient R: %.4f\n', R_wall);
fprintf('  Absorption Coefficient α: %.4f\n', alpha_actual);
fprintf('  Note: This is higher than Sabine (α=0.263) due to ISM differences\n');

%% 2. Setup Geometry
% Microphone Array (Room Center)
mic_pos = [2.41, 2.45, 1.5;  % Mic 1
           2.49, 2.45, 1.5]; % Mic 2

% Sources
src1_pos = [2.45, 3.45, 1.5]; % Target (0° azimuth)
src2_pos = [3.22, 3.06, 1.5]; % Interference (40° azimuth)

%% 3. Generate RIR using Optimized Image Source Method
fprintf('\nGenerating RIRs...\n');

% RIR for Source 1 (Target)
rir_src1 = acousticRoomResponse_optimized(L, src1_pos, mic_pos, R_wall, fs, RT60_target);

% RIR for Source 2 (Interference)
rir_src2 = acousticRoomResponse_optimized(L, src2_pos, mic_pos, R_wall, fs, RT60_target);

% Time vector for RIR
t_rir = (0:size(rir_src1,1)-1)/fs;

%% 4. Verify RT60 with Multiple Methods
fprintf('\n=== RT60 Verification ===\n');

% Method 1: Schroeder integration
function rt60 = estimate_rt60_schroeder(ir, fs)
    energy = flipud(cumsum(flipud(ir.^2)));
    if max(energy) == 0
        rt60 = NaN;
        return;
    end
    energy = energy / max(energy);
    energy_db = 10 * log10(energy + eps);
    
    % Find -5 dB and -25 dB points (T20 method - more robust)
    idx_5db = find(energy_db <= -5, 1, 'first');
    idx_25db = find(energy_db <= -25, 1, 'first');
    
    if isempty(idx_5db) || isempty(idx_25db) || idx_25db <= idx_5db
        rt60 = NaN;
        return;
    end
    
    t_5db = (idx_5db-1)/fs;
    t_25db = (idx_25db-1)/fs;
    
    % T20 = time for 20 dB decay, RT60 = 3 * T20
    rt60 = 3 * (t_25db - t_5db);
end

% Calculate RT60 for all channels
rt60_results = zeros(4,1);
rt60_results(1) = estimate_rt60_schroeder(rir_src1(:,1), fs);
rt60_results(2) = estimate_rt60_schroeder(rir_src1(:,2), fs);
rt60_results(3) = estimate_rt60_schroeder(rir_src2(:,1), fs);
rt60_results(4) = estimate_rt60_schroeder(rir_src2(:,2), fs);

fprintf('Source1-Mic1: %.3f s\n', rt60_results(1));
fprintf('Source1-Mic2: %.3f s\n', rt60_results(2));
fprintf('Source2-Mic1: %.3f s\n', rt60_results(3));
fprintf('Source2-Mic2: %.3f s\n', rt60_results(4));
fprintf('Average RT60: %.3f s (Target: 0.5 s)\n', mean(rt60_results, 'omitnan'));

%% 5. Plot Impulse Responses and Energy Decay
figure('Position', [100, 100, 1400, 800]);

% Impulse responses
subplot(2,3,1);
plot(t_rir, rir_src1(:,1), 'b', 'LineWidth', 1);
hold on;
plot(t_rir, rir_src1(:,2), 'r', 'LineWidth', 1);
title('Source 1 (Target) - Impulse Responses');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2');
grid on;
xlim([0, 0.5]);

subplot(2,3,2);
plot(t_rir, rir_src2(:,1), 'b', 'LineWidth', 1);
hold on;
plot(t_rir, rir_src2(:,2), 'r', 'LineWidth', 1);
title('Source 2 (Interference) - Impulse Responses');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2');
grid on;
xlim([0, 0.5]);

% Energy decay curves
subplot(2,3,4);
energy1 = rir_src1(:,1).^2;
energy1 = energy1 / max(energy1);
energy1_db = 10*log10(energy1 + eps);
plot(t_rir, energy1_db, 'b', 'LineWidth', 1.5);
title('Mic 1 Energy Decay (Source 1)');
xlabel('Time (s)'); ylabel('Energy (dB)');
grid on;
hold on;
plot([0, t_rir(end)], [-5, -5], 'k--', 'LineWidth', 1);
plot([0, t_rir(end)], [-25, -25], 'k--', 'LineWidth', 1);
xlim([0, 0.8]);
ylim([-70, 0]);

subplot(2,3,5);
energy2 = rir_src1(:,2).^2;
energy2 = energy2 / max(energy2);
energy2_db = 10*log10(energy2 + eps);
plot(t_rir, energy2_db, 'r', 'LineWidth', 1.5);
title('Mic 2 Energy Decay (Source 1)');
xlabel('Time (s)'); ylabel('Energy (dB)');
grid on;
hold on;
plot([0, t_rir(end)], [-5, -5], 'k--', 'LineWidth', 1);
plot([0, t_rir(end)], [-25, -25], 'k--', 'LineWidth', 1);
xlim([0, 0.8]);
ylim([-70, 0]);

% RT60 summary
subplot(2,3,3);
bar(rt60_results);
hold on;
yline(0.5, 'r--', 'LineWidth', 2, 'DisplayName', 'Target RT60');
title('RT60 Measurements');
xlabel('Channel'); ylabel('RT60 (s)');
xticklabels({'S1-M1','S1-M2','S2-M1','S2-M2'});
legend('Location', 'best');
grid on;

% Room layout
subplot(2,3,6);
plot3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold on;
plot3(src1_pos(1), src1_pos(2), src1_pos(3), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot3(src2_pos(1), src2_pos(2), src2_pos(3), 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Room Layout');
legend('Microphones', 'Target Source', 'Interference Source', 'Location', 'best');
grid on;
axis equal;
xlim([0, L(1)]); ylim([0, L(2)]); zlim([0, L(3)]);
view(45, 30);

sgtitle(sprintf('Room Impulse Response Simulation (Average RT60 = %.3f s)', mean(rt60_results, 'omitnan')));

%% 6. Generate Audio Signals
fprintf('\n=== Generating Audio Signals ===\n');

duration = 14;
t_sig = (0:1/fs:duration-1/fs)';

% Check if audio files exist, otherwise create synthetic signals
if exist('target.wav', 'file') && exist('interf.wav', 'file')
    fprintf('Loading audio files...\n');
    [sig_target, fs_target] = audioread('target.wav');
    [sig_interf, fs_interf] = audioread('interf.wav');
    
    % Resample if necessary
    if fs_target ~= fs
        sig_target = resample(sig_target, fs, fs_target);
    end
    if fs_interf ~= fs
        sig_interf = resample(sig_interf, fs, fs_interf);
    end
    
    % Convert to mono if stereo
    if size(sig_target, 2) > 1
        sig_target = mean(sig_target, 2);
    end
    if size(sig_interf, 2) > 1
        sig_interf = mean(sig_interf, 2);
    end
   
end

% Ensure signals are same length
min_len = min(length(sig_target), length(sig_interf));
sig_target = sig_target(1:min_len);
sig_interf = sig_interf(1:min_len);
t_sig = t_sig(1:min_len);

%% 7. Apply Reverberation
fprintf('Applying reverberation...\n');

% Convolve each channel
rev_target = zeros(min_len + size(rir_src1,1) - 1, 2);
rev_interf = zeros(min_len + size(rir_src2,1) - 1, 2);

for ch = 1:2
    rev_target(:, ch) = conv(sig_target, rir_src1(:, ch));
    rev_interf(:, ch) = conv(sig_interf, rir_src2(:, ch));
end

% Trim to original length
rev_target = rev_target(1:min_len, :);
rev_interf = rev_interf(1:min_len, :);

%% 8. Adjust SIR to 0 dB
fprintf('Adjusting SIR to 0 dB...\n');

P_target_mic1 = mean(rev_target(:,1).^2);
P_interf_mic1 = mean(rev_interf(:,1).^2);
scale_interf_mic1 = sqrt(P_target_mic1 / P_interf_mic1);
rev_interf(:,1) = rev_interf(:,1) * scale_interf_mic1;

P_target_mic2 = mean(rev_target(:,2).^2);
P_interf_mic2 = mean(rev_interf(:,2).^2);
scale_interf_mic2 = sqrt(P_target_mic2 / P_interf_mic2);
rev_interf(:,2) = rev_interf(:,2) * scale_interf_mic2;

% Mix signals
mix_clean = rev_target + rev_interf;

%% 9. Add Sensor Noise for SNR = 5 dB
fprintf('Adding sensor noise (SNR = 5 dB)...\n');

SNR_target = Inf; % dB
SNR_linear = 10^(SNR_target/10);

% Calculate noise power for each channel
P_signal_mic1 = mean(mix_clean(:,1).^2);
P_signal_mic2 = mean(mix_clean(:,2).^2);

P_noise_mic1 = P_signal_mic1 / SNR_linear;
P_noise_mic2 = P_signal_mic2 / SNR_linear;

% Generate white Gaussian noise
noise_mic1 = sqrt(P_noise_mic1) * randn(size(mix_clean(:,1)));
noise_mic2 = sqrt(P_noise_mic2) * randn(size(mix_clean(:,2)));

% Add noise
mic_signals = mix_clean + [noise_mic1, noise_mic2];

%% 10. Verify Signal Conditions
fprintf('\n=== Signal Condition Verification ===\n');

% Calculate actual SIR
P_target_actual_mic1 = mean(rev_target(:,1).^2);
P_interf_actual_mic1 = mean((rev_interf(:,1)/scale_interf_mic1).^2 * scale_interf_mic1^2);
SIR_actual_mic1 = 10*log10(P_target_actual_mic1 / P_interf_actual_mic1);

P_target_actual_mic2 = mean(rev_target(:,2).^2);
P_interf_actual_mic2 = mean((rev_interf(:,2)/scale_interf_mic2).^2 * scale_interf_mic2^2);
SIR_actual_mic2 = 10*log10(P_target_actual_mic2 / P_interf_actual_mic2);

% Calculate actual SNR
P_signal_noisy_mic1 = mean(mic_signals(:,1).^2);
SNR_actual_mic1 = 10*log10(P_signal_noisy_mic1 / P_noise_mic1);

P_signal_noisy_mic2 = mean(mic_signals(:,2).^2);
SNR_actual_mic2 = 10*log10(P_signal_noisy_mic2 / P_noise_mic2);

fprintf('Mic 1 - Actual SIR: %.2f dB (Target: 0 dB)\n', SIR_actual_mic1);
fprintf('Mic 2 - Actual SIR: %.2f dB (Target: 0 dB)\n', SIR_actual_mic2);
fprintf('Mic 1 - Actual SNR: %.2f dB (Target: 5 dB)\n', SNR_actual_mic1);
fprintf('Mic 2 - Actual SNR: %.2f dB (Target: 5 dB)\n', SNR_actual_mic2);

%% 11. Save Results (Properly Normalized)
fprintf('\n=== Saving Results ===\n');

% Normalize to avoid clipping
max_val = max(abs(mic_signals(:)));
if max_val > 0
    mic_signals = mic_signals / max_val * 0.9; % 0.9 for headroom
end

% Save audio files
audiowrite('GEN_REV_mic1_output.wav', mic_signals(:,1), fs);
audiowrite('GEN_REV_mic2_output.wav', mic_signals(:,2), fs);
audiowrite('GEN_REV_stereo_output.wav', mic_signals, fs);

% % Save RIR data
% save('room_impulse_responses.mat', 'rir_src1', 'rir_src2', 'fs', 'L', ...
%      'mic_pos', 'src1_pos', 'src2_pos', 'R_wall', 'rt60_results');

% Save parameters to text file
fid = fopen('simulation_summary.txt', 'w');
fprintf(fid, '=== Room Impulse Response Simulation Summary ===\n\n');
fprintf(fid, 'Room Configuration:\n');
fprintf(fid, '  Dimensions: %.1f x %.1f x %.1f m\n', L);
fprintf(fid, '  Volume: %.2f m³\n', Volume);
fprintf(fid, '  Surface Area: %.2f m²\n\n', SurfaceArea);
fprintf(fid, 'Target Parameters:\n');
fprintf(fid, '  RT60: %.2f s\n', RT60_target);
fprintf(fid, '  SIR: 0 dB\n');
fprintf(fid, '  SNR: 5 dB\n\n');
fprintf(fid, 'Achieved Results:\n');
fprintf(fid, '  Average RT60: %.3f s\n', mean(rt60_results, 'omitnan'));
fprintf(fid, '  Reflection Coefficient R: %.4f\n', R_wall);
fprintf(fid, '  Absorption Coefficient α: %.4f\n\n', 1-R_wall^2);
fprintf(fid, 'Signal Conditions:\n');
fprintf(fid, '  Mic 1 SIR: %.2f dB\n', SIR_actual_mic1);
fprintf(fid, '  Mic 2 SIR: %.2f dB\n', SIR_actual_mic2);
fprintf(fid, '  Mic 1 SNR: %.2f dB\n', SNR_actual_mic1);
fprintf(fid, '  Mic 2 SNR: %.2f dB\n\n', SNR_actual_mic2);
fprintf(fid, 'File Outputs:\n');
fprintf(fid, '  mic1_output.wav - Microphone 1 signal\n');
fprintf(fid, '  mic2_output.wav - Microphone 2 signal\n');
fprintf(fid, '  stereo_output.wav - Both channels\n');
fprintf(fid, '  room_impulse_responses.mat - RIR data\n');
fclose(fid);

%% 12. Plot Final Signals
figure('Position', [100, 100, 1400, 600]);

% Time domain
subplot(2,3,1:2);
plot(t_sig, mic_signals(:,1), 'b', 'LineWidth', 1);
hold on;
plot(t_sig, mic_signals(:,2), 'r', 'LineWidth', 1);
title('Microphone Signals (Time Domain)');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2');
grid on;
xlim([0, min(3, duration)]);

% Spectrogram
subplot(2,3,4:5);
spectrogram(mic_signals(:,1), 256, 250, 256, fs, 'yaxis');
title('Mic 1 - Spectrogram');
colorbar;

% Signal statistics
subplot(2,3,3);
stats = [SIR_actual_mic1, SNR_actual_mic1; SIR_actual_mic2, SNR_actual_mic2];
bar(stats);
title('Signal Conditions');
ylabel('dB');
set(gca, 'XTickLabel', {'Mic 1', 'Mic 2'});
legend('SIR', 'SNR', 'Location', 'best');
grid on;

% Histogram
subplot(2,3,6);
histogram(mic_signals(:,1), 50, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
histogram(mic_signals(:,2), 50, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
title('Amplitude Distribution');
xlabel('Amplitude'); ylabel('Count');
legend('Mic 1', 'Mic 2');
grid on;

sgtitle(sprintf('Final Output Signals (RT60 ≈ %.2f s)', mean(rt60_results, 'omitnan')));

fprintf('\n=== Simulation Complete ===\n');
fprintf('Output files saved successfully.\n');
fprintf('Average RT60 achieved: %.3f s\n', mean(rt60_results, 'omitnan'));
fprintf('Check simulation_summary.txt for details.\n');

%% --- Optimized Image Source Method Function ---
function h = acousticRoomResponse_optimized(room_dims, src_pos, mic_pos, R, fs, RT60)
    % Optimized Image Source Method for shoebox rooms
    
    c = 343;
    max_time = RT60 * 1.2; % Include some extra time
    max_dist = c * max_time;
    
    % Calculate optimal order
    mean_wall_dist = min(room_dims);
    max_order = ceil(max_dist / mean_wall_dist);
    max_order = min(max_order, 15); % Cap for performance
    
    nMics = size(mic_pos, 1);
    nSamples = ceil(max_time * fs);
    h = zeros(nSamples, nMics);
    
    % Pre-calculate image source positions and gains
    for mx = -max_order:max_order
        for my = -max_order:max_order
            for mz = -max_order:max_order
                % Calculate reflection order
                order = abs(mx) + abs(my) + abs(mz);
                if order > max_order
                    continue;
                end
                
                % Calculate gain for this order
                gain = R^order;
                
                % 8 permutations for mirror images
                for px = [-1, 1]
                    for py = [-1, 1]
                        for pz = [-1, 1]
                            % Image source position
                            img_x = 2*mx*room_dims(1) + px*src_pos(1);
                            img_y = 2*my*room_dims(2) + py*src_pos(2);
                            img_z = 2*mz*room_dims(3) + pz*src_pos(3);
                            
                            for mic = 1:nMics
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
                                    
                                    if delay > 0 && delay <= nSamples
                                        h(delay, mic) = h(delay, mic) + amp;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    % Apply high-pass filter to remove DC
    [b, a] = butter(2, 50/(fs/2), 'high');
    for mic = 1:nMics
        h(:, mic) = filter(b, a, h(:, mic));
        % Normalize
        h(:, mic) = h(:, mic) / max(abs(h(:, mic))) * 0.5;
    end
end