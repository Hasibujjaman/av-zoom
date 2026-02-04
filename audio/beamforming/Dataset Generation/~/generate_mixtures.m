%% Generate Competition Mixtures with Different Interference Combinations
clear; clc; close all;


%% Parameters (as per competition requirements)
fs = 16000;                 % Sampling rate (Hz)
c = 340;                    % Speed of sound (m/s)
d = 0.08;                   % Microphone spacing (m)
RT60_target = 0.5;          % Reverberation time (s)
SIR_target = 0;             % Signal-to-Interference Ratio (dB)
SNR_target = Inf;             % Signal-to-Noise Ratio (dB)

% Room dimensions
room_dims = [4.9, 4.9, 4.9];  % [Lx, Ly, Lz] in meters

% Fixed positions (as per competition)
mic_pos = [2.41, 2.45, 1.5;  % Mic 1
           2.49, 2.45, 1.5]; % Mic 2
array_center = [2.45, 2.45, 1.5];

% Target source position (0° azimuth, directly in front)
theta_target = 0;  % degrees
target_pos = [2.45, 3.45, 1.5];  % (x, y, z) in meters

% Interference source position (40° azimuth, right-front)
theta_interf = 40;  % degrees
interf_pos = [3.22, 3.06, 1.5];  % (x, y, z) in meters

%% Load folder information
male_folder = '../Male_clean';
female_folder = '../Female';
music_folder = '../Music';
noise_folder = '../Noise';

% Get file lists
male_files = dir(fullfile(male_folder, '*.flac'));
female_files = dir(fullfile(female_folder, '*.flac'));
music_files = dir(fullfile(music_folder, '*.flac'));
noise_files = dir(fullfile(noise_folder, '*.flac'));

fprintf('=== File Counts ===\n');
fprintf('Male files: %d\n', length(male_files));
fprintf('Female files: %d\n', length(female_files));
fprintf('Music files: %d\n', length(music_files));
fprintf('Noise files: %d\n', length(noise_files));

%% Helper function to load and prepare audio
function sig = load_audio_file(filepath, fs, duration)
    % Load audio file and ensure it's 3 seconds at fs Hz
    [sig, file_fs] = audioread(filepath);
    
    % Resample if necessary
    if file_fs ~= fs
        sig = resample(sig, fs, file_fs);
    end
    
    % Convert to mono if stereo
    if size(sig, 2) > 1
        sig = mean(sig, 2);
    end
    
    % Ensure exactly duration seconds
    target_samples = fs * duration;
    if length(sig) > target_samples
        sig = sig(1:target_samples);
    elseif length(sig) < target_samples
        sig = [sig; zeros(target_samples - length(sig), 1)];
    end
end

%% Helper function to verify mixture conditions
function verify_conditions(mixture, target_mc, interf_mc, noise_mc, fs, SIR_target, SNR_target)
    fprintf('\n=== Condition Verification ===\n');
    
    % Calculate SIR (using first microphone)
    P_target = mean(target_mc(:,1).^2);
    P_interf = mean(interf_mc(:,1).^2);  
    SIR_actual = 10*log10(P_target / P_interf);
    fprintf('SIR: Target = %.1f dB, Actual = %.2f dB\n', SIR_target, SIR_actual);
    
    % Calculate SNR (using first microphone)
    P_signal = mean(mixture(:,1).^2);
    P_noise = mean(noise_mc(:,1).^2);
    SNR_actual = 10*log10(P_signal / P_noise);
    fprintf('SNR: Target = %.1f dB, Actual = %.2f dB\n', SNR_target, SNR_actual);
    
    % Check clipping
    if max(abs(mixture(:))) > 0.99
        fprintf('Warning: Signal may be clipping (max amplitude = %.3f)\n', max(abs(mixture(:))));
    else
        fprintf('No clipping detected (max amplitude = %.3f)\n', max(abs(mixture(:))));
    end
    
    % Calculate RT60 of mixture (approximate)
    function rt60 = estimate_rt60_mix(sig, fs)
        % Simple energy decay estimation
        energy = flipud(cumsum(flipud(sig.^2)));
        energy = energy / max(energy);
        energy_db = 10*log10(energy + eps);
        
        % Find -5 dB and -25 dB points
        idx_5db = find(energy_db <= -5, 1, 'first');
        idx_25db = find(energy_db <= -25, 1, 'first');
        
        if isempty(idx_5db) || isempty(idx_25db)
            rt60 = NaN;
            return;
        end
        
        rt60 = 3 * ((idx_25db-1) - (idx_5db-1)) / fs;
    end
    
    rt60_mix = estimate_rt60_mix(mixture(:,1), fs);
    fprintf('Estimated RT60 from mixture: %.3f s (Target: 0.5 s)\n', rt60_mix);
end










%% Create interference combinations
% For each male file, create 5 combinations
for male_idx = 1:min(2, length(male_files)) %
    fprintf('\n\n=== Processing Male File %d/%d ===\n', male_idx, min(2, length(male_files)));
    
    % Load male speech (target)
    male_file = male_files(male_idx);
    male_path = fullfile(male_folder, male_file.name);
    target_sig = load_audio_file(male_path, fs, 3);
    fprintf('Target: %s\n', male_file.name);
    
    % Extract base name (without extension)
    [~, male_base_name, ~] = fileparts(male_file.name);
    
    % Randomly select interference files
    female_idx = randi([1, length(female_files)]);
    music_idx = randi([1, length(music_files)]);
    noise_idx = randi([1, length(noise_files)]);
    
    % Load interference components
    female_sig = load_audio_file(fullfile(female_folder, female_files(female_idx).name), fs, 3);
    music_sig = load_audio_file(fullfile(music_folder, music_files(music_idx).name), fs, 3);
    noise_sig = load_audio_file(fullfile(noise_folder, noise_files(noise_idx).name), fs, 3);
    
    fprintf('Interferences selected:\n');
    fprintf('  Female: %s\n', female_files(female_idx).name);
    fprintf('  Music: %s\n', music_files(music_idx).name);
    fprintf('  Noise: %s\n', noise_files(noise_idx).name);
    
    %% Combination A: Female speech only
    fprintf('\n--- Creating Combination A (Female only) ---\n');
    interf_sig_A = female_sig;
    
    [mixture_A, target_mc_A, interf_mc_A, noise_mc_A] = create_reverberant_mixture(...
        target_sig, interf_sig_A, theta_target, theta_interf, ...
        fs, SNR_target, c, d, ...
        'RT60', RT60_target, ...
        'RoomDims', room_dims, ...
        'MicPositions', mic_pos, ...
        'PlotResults', false, ...
        'SaveAudio', false, ...
        'Verbose', false);
    
    % Verify conditions
    verify_conditions(mixture_A, target_mc_A, interf_mc_A, noise_mc_A, fs, SIR_target, SNR_target);
    
    % Save mixture
    output_name_A = sprintf('%s_A_female_only.wav', male_base_name);
    audiowrite(output_name_A, mixture_A, fs);
    fprintf('Saved: %s\n', output_name_A);
    
    %% Combination B: Female + Music
    fprintf('\n--- Creating Combination B (Female + Music) ---\n');
    % Mix female and music (equal power)
    P_female = mean(female_sig.^2);
    P_music = mean(music_sig.^2);
    music_sig_scaled = music_sig * sqrt(P_female / P_music);
    interf_sig_B = female_sig + music_sig_scaled;
    
    [mixture_B, target_mc_B, interf_mc_B, noise_mc_B] = create_reverberant_mixture(...
        target_sig, interf_sig_B, theta_target, theta_interf, ...
        fs, SNR_target, c, d, ...
        'RT60', RT60_target, ...
        'RoomDims', room_dims, ...
        'MicPositions', mic_pos, ...
        'PlotResults', false, ...
        'SaveAudio', false, ...
        'Verbose', true);
    
    verify_conditions(mixture_B, target_mc_B, interf_mc_B, noise_mc_B, fs, SIR_target, SNR_target);
    
    output_name_B = sprintf('%s_B_female_music.wav', male_base_name);
    audiowrite(output_name_B, mixture_B, fs);
    fprintf('Saved: %s\n', output_name_B);
    
    %% Combination C: Female + Noise
    fprintf('\n--- Creating Combination C (Female + Noise) ---\n');
    % Mix female and noise (equal power)
    P_noise = mean(noise_sig.^2);
    noise_sig_scaled = noise_sig * sqrt(P_female / P_noise);
    interf_sig_C = female_sig + noise_sig_scaled;
    
    [mixture_C, target_mc_C, interf_mc_C, noise_mc_C] = create_reverberant_mixture(...
        target_sig, interf_sig_C, theta_target, theta_interf, ...
        fs, SNR_target, c, d, ...
        'RT60', RT60_target, ...
        'RoomDims', room_dims, ...
        'MicPositions', mic_pos, ...
        'PlotResults', false, ...
        'SaveAudio', false, ...
        'Verbose', true);
    
    verify_conditions(mixture_C, target_mc_C, interf_mc_C, noise_mc_C, fs, SIR_target, SNR_target);
    
    output_name_C = sprintf('%s_C_female_noise.wav', male_base_name);
    audiowrite(output_name_C, mixture_C, fs);
    fprintf('Saved: %s\n', output_name_C);
    
    %% Combination D: Music + Noise
    fprintf('\n--- Creating Combination D (Music + Noise) ---\n');
    % Mix music and noise (equal power)
    noise_sig_scaled2 = noise_sig * sqrt(P_music / P_noise);
    interf_sig_D = music_sig + noise_sig_scaled2;
    
    [mixture_D, target_mc_D, interf_mc_D, noise_mc_D] = create_reverberant_mixture(...
        target_sig, interf_sig_D, theta_target, theta_interf, ...
        fs, SNR_target, c, d, ...
        'RT60', RT60_target, ...
        'RoomDims', room_dims, ...
        'MicPositions', mic_pos, ...
        'PlotResults', false, ...
        'SaveAudio', false, ...
        'Verbose', true);
    
    verify_conditions(mixture_D, target_mc_D, interf_mc_D, noise_mc_D, fs, SIR_target, SNR_target);
    
    output_name_D = sprintf('%s_D_music_noise.wav', male_base_name);
    audiowrite(output_name_D, mixture_D, fs);
    fprintf('Saved: %s\n', output_name_D);
    
    %% Combination E: Female + Music + Noise
    fprintf('\n--- Creating Combination E (Female + Music + Noise) ---\n');
    % Mix all three with equal power
    interf_sig_E = female_sig + music_sig_scaled + noise_sig_scaled;
    
    [mixture_E, target_mc_E, interf_mc_E, noise_mc_E] = create_reverberant_mixture(...
        target_sig, interf_sig_E, theta_target, theta_interf, ...
        fs, SNR_target, c, d, ...
        'RT60', RT60_target, ...
        'RoomDims', room_dims, ...
        'MicPositions', mic_pos, ...
        'PlotResults', false, ...
        'SaveAudio', false, ...
        'Verbose', true);
    
    verify_conditions(mixture_E, target_mc_E, interf_mc_E, noise_mc_E, fs, SIR_target, SNR_target);
    
    output_name_E = sprintf('%s_E_female_music_noise.wav', male_base_name);
    audiowrite(output_name_E, mixture_E, fs);
    fprintf('Saved: %s\n', output_name_E);
    
    %% Create summary for this male file
    fprintf('\n=== Summary for %s ===\n', male_base_name);
    fprintf('Generated 5 mixtures:\n');
    fprintf('1. %s\n', output_name_A);
    fprintf('2. %s\n', output_name_B);
    fprintf('3. %s\n', output_name_C);
    fprintf('4. %s\n', output_name_D);
    fprintf('5. %s\n', output_name_E);
    
    % Save metadata
    metadata = struct();
    metadata.male_file = male_file.name;
    metadata.female_file = female_files(female_idx).name;
    metadata.music_file = music_files(music_idx).name;
    metadata.noise_file = noise_files(noise_idx).name;
    metadata.parameters.fs = fs;
    metadata.parameters.c = c;
    metadata.parameters.d = d;
    metadata.parameters.RT60 = RT60_target;
    metadata.parameters.SIR = SIR_target;
    metadata.parameters.SNR = SNR_target;
    metadata.parameters.room_dims = room_dims;
    metadata.parameters.mic_positions = mic_pos;
    metadata.parameters.target_position = target_pos;
    metadata.parameters.interf_position = interf_pos;
    
    save(sprintf('%s_metadata.mat', male_base_name), 'metadata');
    
    %% Plot comparison of all mixtures
    % figure('Position', [100, 100, 1400, 800]);
    % 
    % % Time domain plots
    % t = (0:size(mixture_A,1)-1)/fs;
    % 
    % subplot(2,3,1);
    % plot(t, mixture_A(:,1), 'b');
    % title(sprintf('A: Female only\n%s', output_name_A), 'Interpreter', 'none');
    % xlabel('Time (s)'); ylabel('Amplitude');
    % grid on; xlim([0, 3]);
    % 
    % subplot(2,3,2);
    % plot(t, mixture_B(:,1), 'b');
    % title(sprintf('B: Female + Music\n%s', output_name_B), 'Interpreter', 'none');
    % xlabel('Time (s)'); ylabel('Amplitude');
    % grid on; xlim([0, 3]);
    % 
    % subplot(2,3,3);
    % plot(t, mixture_C(:,1), 'b');
    % title(sprintf('C: Female + Noise\n%s', output_name_C), 'Interpreter', 'none');
    % xlabel('Time (s)'); ylabel('Amplitude');
    % grid on; xlim([0, 3]);
    % 
    % subplot(2,3,4);
    % plot(t, mixture_D(:,1), 'b');
    % title(sprintf('D: Music + Noise\n%s', output_name_D), 'Interpreter', 'none');
    % xlabel('Time (s)'); ylabel('Amplitude');
    % grid on; xlim([0, 3]);
    % 
    % subplot(2,3,5);
    % plot(t, mixture_E(:,1), 'b');
    % title(sprintf('E: All three\n%s', output_name_E), 'Interpreter', 'none');
    % xlabel('Time (s)'); ylabel('Amplitude');
    % grid on; xlim([0, 3]);
    % 
    % % Spectrogram of mixture A
    % subplot(2,3,6);
    % spectrogram(mixture_A(:,1), 256, 250, 256, fs, 'yaxis');
    % title('Spectrogram (Mixture A)');
    % colorbar;
    % 
    % sgtitle(sprintf('Mixture Combinations for Target: %s', male_base_name), 'Interpreter', 'none');
    % 
    % Save figure
    % saveas(gcf, sprintf('%s_mixtures_comparison.png', male_base_name));
end












%% Create verification report
fprintf('\n=== Generating Verification Report ===\n');

% Create a CSV report
report_file = 'mixture_verification_report.csv';
fid = fopen(report_file, 'w');
fprintf(fid, 'MixtureName,MaleFile,FemaleFile,MusicFile,NoiseFile,RT60,SIR,SNR,MaxAmplitude\n');

% You would populate this by reading back the saved files and verifying
% For now, we'll note that verification was done during creation
fclose(fid);
fprintf('Report saved to: %s\n', report_file);

%% Create README file with mixture specifications
readme_file = 'README_MIXTURES.txt';
fid = fopen(readme_file, 'w');
fprintf(fid, '=== Mixture Generation Specifications ===\n\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '\n--- Competition Parameters ---\n');
fprintf(fid, 'Sampling Rate: %d Hz\n', fs);
fprintf(fid, 'Speed of Sound: %.0f m/s\n', c);
fprintf(fid, 'Microphone Spacing: %.2f m\n', d);
fprintf(fid, 'RT60 Target: %.1f s\n', RT60_target);
fprintf(fid, 'SIR Target: %.0f dB\n', SIR_target);
fprintf(fid, 'SNR Target: %.0f dB\n', SNR_target);
fprintf(fid, '\n--- Room Geometry ---\n');
fprintf(fid, 'Room Dimensions: %.1f x %.1f x %.1f m\n', room_dims);
fprintf(fid, 'Mic 1 Position: (%.2f, %.2f, %.2f) m\n', mic_pos(1,:));
fprintf(fid, 'Mic 2 Position: (%.2f, %.2f, %.2f) m\n', mic_pos(2,:));
fprintf(fid, 'Target Position (0°): (%.2f, %.2f, %.2f) m\n', target_pos);
fprintf(fid, 'Interf Position (40°): (%.2f, %.2f, %.2f) m\n', interf_pos);
fprintf(fid, '\n--- Mixture Combinations ---\n');
fprintf(fid, 'A: Female speech only\n');
fprintf(fid, 'B: Female speech + Music\n');
fprintf(fid, 'C: Female speech + Noise\n');
fprintf(fid, 'D: Music + Noise\n');
fprintf(fid, 'E: Female speech + Music + Noise\n');
fprintf(fid, '\n--- File Naming Convention ---\n');
fprintf(fid, '[MaleBaseName]_[Combination]_[InterferenceTypes].wav\n');
fprintf(fid, 'Example: 1_part1_A_female_only.wav\n');
fprintf(fid, '         male_base = "1_part1"\n');
fprintf(fid, '         combination = "A"\n');
fprintf(fid, '         interference = "female_only"\n');
fclose(fid);

fprintf('\n=== Generation Complete ===\n');
fprintf('All mixtures saved with verification.\n');
fprintf('Check %s for specifications.\n', readme_file);