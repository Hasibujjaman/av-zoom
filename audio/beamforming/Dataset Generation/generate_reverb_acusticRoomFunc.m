%% Room Impulse Response Synthesis for Microphone Array
% Room Setup
clear; clc; close all;

SNR_dB = 5;  % 5 dB SNR as specified
% Room dimensions (shoebox)
roomDimensions = [4.9, 4.9, 4.9]; % [Lx, Ly, Lz] in meters

% Microphone positions (2-microphone linear array)
mic1 = [2.41, 2.45, 1.5];  % Mic 1
mic2 = [2.49, 2.45, 1.5];  % Mic 2
rx = [mic1; mic2];         % Receiver array

% Source positions
source1 = [2.45, 3.45, 1.5];  % Target (0° azimuth)
source2 = [3.22, 3.06, 1.5];  % Interference (40° azimuth)

%% Step 1: Calculate absorption coefficient for RT60 ≈ 0.5 s
% Using Sabine's formula: RT60 = 0.161 * V / (A * S)
% Where: V = volume, S = total surface area, A = average absorption coefficient

V = prod(roomDimensions);  % Room volume
S = 2*(roomDimensions(1)*roomDimensions(2) + ...
       roomDimensions(1)*roomDimensions(3) + ...
       roomDimensions(2)*roomDimensions(3)); % Total surface area

RT60_target = 0.5;  % Target reverberation time
alpha_sabine = 0.161 * V / (RT60_target * S); % Required absorption coefficient

% Convert to absorption coefficient (simplified approach)
% Note: For more accuracy, we might need frequency-dependent coefficients
% We'll use the same absorption for all surfaces initially
% Using Sabine formula: alpha ≈ 0.263 for RT60 = 0.5s in this room
absorption_coefficient = alpha_sabine;  % Use calculated value (~0.263)

% Compute reflection coefficient from absorption coefficient
% reflection_coeff = sqrt(1 - alpha)
reflection_coefficient = sqrt(1 - absorption_coefficient);
fprintf('Calculated absorption coefficient: %.4f\n', absorption_coefficient);
fprintf('Calculated reflection coefficient: %.4f\n', reflection_coefficient);

%% Step 2: Generate Room Impulse Responses using acousticRoomResponse
% Common parameters
fs = 16000;  % Sampling rate
algorithm = "image-source";  % Using image source method
% Higher order needed for RT60 = 0.5s (more reflections = longer decay)
% Rule of thumb: order should be high enough to capture reflections for ~RT60 duration
image_source_order = 7;  % Increased for longer RT60
ms = 0.05;  % Lower scattering for more specular reflections

% Generate RIR for Source 1 (Target)
fprintf('Generating RIR for Source 1 (Target)...\n');
ir_source1 = acousticRoomResponse(roomDimensions, source1, rx, ...
    'SampleRate', fs, ...
    'Algorithm', algorithm, ...
    'ImageSourceOrder', image_source_order, ...
    'MaterialAbsorption', absorption_coefficient, ...
    'MaterialScattering', ms); % Default scattering

% Generate RIR for Source 2 (Interference)
fprintf('Generating RIR for Source 2 (Interference)...\n');
ir_source2 = acousticRoomResponse(roomDimensions, source2, rx, ...
    'SampleRate', fs, ...
    'Algorithm', algorithm, ...
    'ImageSourceOrder', image_source_order, ...
    'MaterialAbsorption', absorption_coefficient, ...
    'MaterialScattering', ms);
% 
% %% Step 3: Plot Impulse Responses
% t = (0:size(ir_source1,2)-1)/fs;
% 
% figure('Position', [100, 100, 1200, 800]);
% 
% % Source 1 RIRs
% subplot(2,2,1);
% plot(t, ir_source1(1,:));
% title('Source 1 - Mic 1 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,2,2);
% plot(t, ir_source1(2,:));
% title('Source 1 - Mic 2 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% % Source 2 RIRs
% subplot(2,2,3);
% plot(t, ir_source2(1,:));
% title('Source 2 - Mic 1 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,2,4);
% plot(t, ir_source2(2,:));
% title('Source 2 - Mic 2 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% sgtitle('Room Impulse Responses (RT60 ≈ 0.5 s)');

%% Step 4: Load and Process Audio Signals
% Load source signals (example - replace with your actual files)
% For demonstration, we'll create synthetic signals
duration = 14;  % 3 seconds of audio
t_audio = (0:fs*duration-1)/fs;

%load real audio files:
[target_signal, fs_target] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav');
[interference_signal, fs_interf] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/female_piano_14s.wav');
% Ensure resampling to 16 kHz if needed

%% Step 5: Apply RIRs to signals (Convolution)
% Initialize received signals
mic1_received = zeros(1, length(target_signal) + size(ir_source1,2) - 1);
mic2_received = zeros(1, length(target_signal) + size(ir_source1,2) - 1);

% Apply RIRs for each source
for i = 1:2
    % Convolve target signal with RIRs
    target_mic1 = conv(target_signal, ir_source1(1,:));
    target_mic2 = conv(target_signal, ir_source1(2,:));
    
    % Convolve interference signal with RIRs
    interf_mic1 = conv(interference_signal, ir_source2(1,:));
    interf_mic2 = conv(interference_signal, ir_source2(2,:));
    
    % Trim to same length
    min_len = min([length(target_mic1), length(interf_mic1), length(mic1_received)]);
    target_mic1 = target_mic1(1:min_len);
    target_mic2 = target_mic2(1:min_len);
    interf_mic1 = interf_mic1(1:min_len);
    interf_mic2 = interf_mic2(1:min_len);
    mic1_received = mic1_received(1:min_len);
    mic2_received = mic2_received(1:min_len);
    
    % Adjust SIR (0 dB - equal power)
    % Calculate current power ratios
    P_target_mic1 = mean(target_mic1.^2);
    P_interf_mic1 = mean(interf_mic1.^2);
    P_target_mic2 = mean(target_mic2.^2);
    P_interf_mic2 = mean(interf_mic2.^2);
    
    % Scale interference to achieve 0 dB SIR
    scale_mic1 = sqrt(P_target_mic1 / P_interf_mic1);
    scale_mic2 = sqrt(P_target_mic2 / P_interf_mic2);
    
    interf_mic1 = interf_mic1 * scale_mic1;
    interf_mic2 = interf_mic2 * scale_mic2;
    
    % Mix signals
    mic1_received = target_mic1 + interf_mic1;
    mic2_received = target_mic2 + interf_mic2;
end

%% Step 6: Add Sensor Noise (SNR = 5 dB)
% Calculate signal power
P_signal_mic1 = mean(mic1_received.^2);
P_signal_mic2 = mean(mic2_received.^2);

% Calculate required noise power for SNR = 5 dB
SNR_linear = 10^(SNR_dB/10);  % Convert dB to linear
P_noise_mic1 = P_signal_mic1 / SNR_linear;
P_noise_mic2 = P_signal_mic2 / SNR_linear;

% Generate white Gaussian noise
noise_mic1 = sqrt(P_noise_mic1) * randn(size(mic1_received));
noise_mic2 = sqrt(P_noise_mic2) * randn(size(mic2_received));

% Add noise to signals
mic1_noisy = mic1_received + noise_mic1;
mic2_noisy = mic2_received + noise_mic2;

%% Step 7: Verify Actual RT60 (Schroeder Integration)
function rt60 = estimate_rt60(ir, fs)
    % Schroeder backward integration
    energy = cumsum(ir(end:-1:1).^2);
    energy = energy(end:-1:1);
    energy = energy / max(energy);
    
    % Convert to dB
    energy_db = 10*log10(energy + eps);
    
    % Find -5 dB and -35 dB points
    idx_5db = find(energy_db <= -5, 1, 'first');
    idx_35db = find(energy_db <= -35, 1, 'first');
    
    if isempty(idx_5db) || isempty(idx_35db)
        rt60 = NaN;
        return;
    end
    
    t_5db = (idx_5db-1)/fs;
    t_35db = (idx_35db-1)/fs;
    
    % RT60 = 2 * (t_35 - t_5) since -35 to -5 is 30 dB drop
    rt60 = 2 * (t_35db - t_5db);
end

% Estimate RT60 for each channel
rt60_estimates = zeros(4,1);
rt60_estimates(1) = estimate_rt60(ir_source1(1,:), fs);
rt60_estimates(2) = estimate_rt60(ir_source1(2,:), fs);
rt60_estimates(3) = estimate_rt60(ir_source2(1,:), fs);
rt60_estimates(4) = estimate_rt60(ir_source2(2,:), fs);

fprintf('\nRT60 Estimates:\n');
fprintf('Source1-Mic1: %.3f s\n', rt60_estimates(1));
fprintf('Source1-Mic2: %.3f s\n', rt60_estimates(2));
fprintf('Source2-Mic1: %.3f s\n', rt60_estimates(3));
fprintf('Source2-Mic2: %.3f s\n', rt60_estimates(4));
fprintf('Average: %.3f s (Target: 0.5 s)\n', mean(rt60_estimates));

% Normalize signals to prevent clipping (scale to [-1, 1] range)
max_amp = max([max(abs(mic1_received(:))), max(abs(mic2_received(:)))]);
if max_amp > 0
    mic1_normalized = mic1_received / max_amp * 0.95;  % Leave 5% headroom
    mic2_normalized = mic2_received / max_amp * 0.95;
else
    mic1_normalized = mic1_received;
    mic2_normalized = mic2_received;
end

stereo_output(:,1) = mic1_normalized(:);
stereo_output(:,2) = mic2_normalized(:);

% Save received audio (normalized to prevent clipping)
audiowrite('mic1_received.wav', mic1_normalized(:), fs);
audiowrite('mic2_received.wav', mic2_normalized(:), fs);
audiowrite('stereo_output.wav', stereo_output, fs);

fprintf('\nMax amplitude before normalization: %.4f\n', max_amp);

fprintf('\nSimulation complete!\n');
fprintf('Files saved:\n');
fprintf('  - room_impulse_responses.mat (RIR data)\n');
fprintf('  - mic1_received.wav (Mic 1 audio)\n');
fprintf('  - mic2_received.wav (Mic 2 audio)\n');
fprintf('  - stereo_output.wav\n');