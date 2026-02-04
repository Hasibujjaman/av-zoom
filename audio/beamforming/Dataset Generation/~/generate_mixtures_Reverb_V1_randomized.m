%% Generate Competition Mixtures - RANDOMIZED VERSION
clear; clc; close all;

% Process first N male files
n_to_process = 2;

%% Parameters (as per competition requirements)
fs = 16000;                 % Sampling rate (Hz)
c = 340;                    % Speed of sound (m/s)
d = 0.08;                   % Microphone spacing (m)
RT60_target = 0.5;          % Reverberation time (s)
SIR_target = 0;             % Signal-to-Interference Ratio (dB)
SNR_target = 5;             % Signal-to-Noise Ratio (dB)

% Set random seed for reproducibility (change this for different runs)
% rng('shuffle');  % Uncomment for different results each time
rng(1);  % Fixed seed for reproducibility

% Display parameters
fprintf('=== Competition Mixture Generation ===\n');
fprintf('Target Parameters:\n');
fprintf('  RT60: %.1f s\n', RT60_target);
fprintf('  SIR: %d dB\n', SIR_target);
fprintf('  SNR: %d dB\n', SNR_target);
fprintf('  Fs: %d Hz\n', fs);
fprintf('  c: %.0f m/s\n', c);
fprintf('  d: %.2f m\n\n', d);

%% Simple and Accurate Mixture Creation Function
function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_competition_mixture(target_sig, interf_sig, fs, SNR_dB, c, d, RT60_target)
    % Competition mixture creation with proper RT60 and SNR control
    
    % Ensure signals are same length
    L = min(length(target_sig), length(interf_sig));
    target_sig = target_sig(1:L);
    interf_sig = interf_sig(1:L);
    
    % 1. Create DOA delays for 2-mic array
    % Target at 0° (directly in front)
    tau_target = d * sind(0) / c;
    delay_target_mic2 = round(tau_target * fs);
    
    % Interference at 40° (right-front)
    tau_interf = d * sind(40) / c;
    delay_interf_mic2 = round(tau_interf * fs);
    
    % 2. Create microphone signals with delays
    target_mc = zeros(L, 2);
    interf_mc = zeros(L, 2);
    
    % Mic 1
    target_mc(:, 1) = target_sig;
    interf_mc(:, 1) = interf_sig;
    
    % Mic 2 with delays
    if delay_target_mic2 > 0
        target_mc(1+delay_target_mic2:end, 2) = target_sig(1:end-delay_target_mic2);
    else
        target_mc(:, 2) = target_sig;
    end
    
    if delay_interf_mic2 > 0
        interf_mc(1+delay_interf_mic2:end, 2) = interf_sig(1:end-delay_interf_mic2);
    else
        interf_mc(:, 2) = interf_sig;
    end
    
    % 3. Add controlled reverberation with EXACT RT60
    % Much shorter and controlled reverberation
    rt60_samples = round(RT60_target * fs);
    
    % Create simple reverb tail (exponential decay)
    % Use a decay that reaches -60dB at RT60_target
    decay_time = RT60_target;  % seconds for -60dB decay
    decay_samples = round(decay_time * fs);
    
    % Create exponential decay
    t_decay = (0:decay_samples-1)' / fs;
    % decay to -60dB (0.001 in linear) at RT60_target
    decay_factor = exp(-log(1000) * t_decay / RT60_target);
    
    % Scale to control reverb level (much smaller)
    reverb_gain = 0.15;  % Reduced from 0.5
    decay_factor = decay_factor * reverb_gain;
    
    % Apply reverb via convolution
    for m = 1:2
        % Target reverb
        target_reverb = conv(target_mc(:, m), decay_factor);
        target_mc(:, m) = target_mc(:, m) + target_reverb(1:L);
        
        % Interference reverb
        interf_reverb = conv(interf_mc(:, m), decay_factor);
        interf_mc(:, m) = interf_mc(:, m) + interf_reverb(1:L);
    end
    
    % 4. Adjust SIR to 0 dB
    P_target = mean(target_mc(:).^2);
    P_interf = mean(interf_mc(:).^2);
    
    if P_interf > 0
        scale_interf = sqrt(P_target / P_interf);
        interf_mc = interf_mc * scale_interf;
    end
    
    % 5. Clean mixture
    mixture_clean = target_mc + interf_mc;
    
    % 6. Add noise for EXACT SNR
    mixture = zeros(size(mixture_clean));
    noise_mc = zeros(size(mixture_clean));
    
    SNR_linear = 10^(SNR_dB/10);
    
    for m = 1:2
        % Calculate clean signal power
        P_signal_clean = mean(mixture_clean(:, m).^2);
        
        % Calculate required noise power
        P_noise_required = P_signal_clean / SNR_linear;
        
        % Generate noise with exact power
        noise = randn(size(mixture_clean(:, m)));
        noise = noise - mean(noise);  % Remove DC
        current_noise_power = mean(noise.^2);
        
        if current_noise_power > 0
            noise = noise * sqrt(P_noise_required / current_noise_power);
        end
        
        noise_mc(:, m) = noise;
        mixture(:, m) = mixture_clean(:, m) + noise;
    end
    
    % 7. Normalize carefully to avoid clipping
    peak_val = max(abs(mixture(:)));
    if peak_val > 0
        mixture = mixture / peak_val * 0.95;  % 0.95 for headroom
        target_mc = target_mc / peak_val * 0.95;
        interf_mc = interf_mc / peak_val * 0.95;
        noise_mc = noise_mc / peak_val * 0.95;
    end
end

%% Helper function to load and prepare audio
function sig = load_audio_file(filepath, fs, duration)
    % Load audio file and ensure it's 3 seconds at fs Hz
    
    % Check file extension
    [~, ~, ext] = fileparts(filepath);
    
    try
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
        
        % Remove DC offset
        sig = sig - mean(sig);
        
    catch ME
        fprintf('Error loading %s: %s\n', filepath, ME.message);
        sig = zeros(fs * duration, 1);
    end
end

%% Accurate condition verification
function [SIR_actual, SNR_actual, rt60_est, max_amp] = verify_mixture_conditions(mixture, target_mc, interf_mc, noise_mc, fs, RT60_target)
    
    % Calculate SIR
    P_target = mean(target_mc(:).^2);
    P_interf = mean(interf_mc(:).^2);
    if P_interf > 0
        SIR_actual = 10*log10(P_target / P_interf);
    else
        SIR_actual = Inf;
    end
    
    % Calculate SNR properly
    mixture_clean_est = target_mc + interf_mc;
    P_signal_clean = mean(mixture_clean_est(:).^2);
    P_noise = mean(noise_mc(:).^2);
    if P_noise > 0
        SNR_actual = 10*log10(P_signal_clean / P_noise);
    else
        SNR_actual = Inf;
    end
    
    % Estimate RT60 using energy decay of reverb tail
    % Take last 0.5 seconds where speech is usually quiet
    if size(mixture, 1) > 0.5*fs
        tail_start = round(0.5 * fs);  % Start at 0.5s
        tail_length = round(0.5 * fs);  % 0.5s tail
        
        if tail_start + tail_length <= size(mixture, 1)
            reverb_tail = mixture(tail_start:tail_start+tail_length-1, 1);
            
            % Energy decay curve
            energy = flipud(cumsum(flipud(reverb_tail.^2)));
            energy = energy / max(energy);
            energy_db = 10*log10(energy + eps);
            
            % Find -5 dB and -25 dB points (T20 method)
            idx_5db = find(energy_db <= -5, 1, 'first');
            idx_25db = find(energy_db <= -25, 1, 'first');
            
            if ~isempty(idx_5db) && ~isempty(idx_25db) && idx_25db > idx_5db
                t_5db = (idx_5db-1)/fs;
                t_25db = (idx_25db-1)/fs;
                rt60_est = 3 * (t_25db - t_5db);
            else
                rt60_est = NaN;
            end
        else
            rt60_est = NaN;
        end
    else
        rt60_est = NaN;
    end
    
    % Check max amplitude
    max_amp = max(abs(mixture(:)));
end

%% Display verification results
function display_verification_results(SIR_actual, SNR_actual, rt60_est, max_amp, RT60_target)
    fprintf('SIR: %.2f dB (target: 0 dB) ', SIR_actual);
    if abs(SIR_actual) < 0.1
        fprintf('✓\n');
    else
        fprintf('✗ (error: %.2f dB)\n', abs(SIR_actual));
    end
    
    fprintf('SNR: %.2f dB (target: 5 dB) ', SNR_actual);
    if abs(SNR_actual - 5) < 0.1
        fprintf('✓\n');
    else
        fprintf('✗ (error: %.2f dB)\n', abs(SNR_actual - 5));
    end
    
    fprintf('RT60: %.3f s (target: %.1f s) ', rt60_est, RT60_target);
    if ~isnan(rt60_est) && abs(rt60_est - RT60_target) < 0.1
        fprintf('✓\n');
    else
        fprintf('✗ (error: %.2f s)\n', abs(rt60_est - RT60_target));
    end
    
    fprintf('Max amplitude: %.3f ', max_amp);
    if max_amp <= 0.95
        fprintf('✓ (safe)\n');
    elseif max_amp <= 0.99
        fprintf('⚠ (near clipping)\n');
    else
        fprintf('✗ (clipping)\n');
    end
end

%% Helper function to randomly select unique indices
function selected_idx = select_random_indices(total_files, num_to_select, avoid_indices)
    % Select random indices from 1:total_files, avoiding duplicates
    if nargin < 3
        avoid_indices = [];
    end
    
    if num_to_select > total_files
        error('Cannot select %d files from %d available', num_to_select, total_files);
    end
    
    % Create list of all indices
    all_indices = 1:total_files;
    
    % Remove indices to avoid
    all_indices = setdiff(all_indices, avoid_indices);
    
    % Randomly shuffle
    shuffled_indices = all_indices(randperm(length(all_indices)));
    
    % Select first num_to_select indices
    selected_idx = shuffled_indices(1:num_to_select);
end















%% Main processing
% Load folder information
male_folder = '../Male_clean';
female_folder = '../Female';
music_folder = '../Music';
noise_folder = '../Noise';

% Get file lists
male_files = dir(fullfile(male_folder, '*flac'));
if isempty(male_files)
    male_files = dir(fullfile(male_folder, '*.flac'));
end
female_files = dir(fullfile(female_folder, '*flac'));
if isempty(female_files)
    female_files = dir(fullfile(female_folder, '*.flac'));
end
music_files = dir(fullfile(music_folder, '*flac'));
if isempty(music_files)
    music_files = dir(fullfile(music_folder, '*.flac'));
end
noise_files = dir(fullfile(noise_folder, '*flac'));
if isempty(noise_files)
    noise_files = dir(fullfile(noise_folder, '*.flac'));
end

fprintf('Found files:\n');
fprintf('  Male: %d\n', length(male_files));
fprintf('  Female: %d\n', length(female_files));
fprintf('  Music: %d\n', length(music_files));
fprintf('  Noise: %d\n\n', length(noise_files));


success_count = 0;

% Keep track of used files to avoid duplicates within the same run
used_female_indices = [];
used_music_indices = [];
used_noise_indices = [];

% For reproducibility, save the random state
random_state = rng;
save('random_state.mat', 'random_state');
fprintf('Random seed saved to random_state.mat for reproducibility\n\n');





for male_idx = 1:min(n_to_process, length(male_files))
    fprintf('=== Processing Male File %d/%d ===\n', male_idx, min(n_to_process, length(male_files)));
    
    % Load male speech (target)
    male_file = male_files(male_idx);
    male_path = fullfile(male_folder, male_file.name);
    target_sig = load_audio_file(male_path, fs, 3);
    
    % Check if signal is valid
    if all(target_sig == 0)
        fprintf('Skipping %s (empty or error)\n', male_file.name);
        continue;
    end
    
    fprintf('Target: %s (%.1f dB)\n', male_file.name, 10*log10(mean(target_sig.^2)));
    
    % Extract base name
    [~, male_base_name, ~] = fileparts(male_file.name);
    
    % RANDOMLY select interference files - NO duplicates within this run
    % Select female file
    female_idx = select_random_indices(length(female_files), 1, used_female_indices);
    used_female_indices = [used_female_indices, female_idx];
    
    % Select music file
    music_idx = select_random_indices(length(music_files), 1, used_music_indices);
    used_music_indices = [used_music_indices, music_idx];
    
    % Select noise file
    noise_idx = select_random_indices(length(noise_files), 1, used_noise_indices);
    used_noise_indices = [used_noise_indices, noise_idx];
    
    % Load interference components
    female_sig = load_audio_file(fullfile(female_folder, female_files(female_idx).name), fs, 3);
    music_sig = load_audio_file(fullfile(music_folder, music_files(music_idx).name), fs, 3);
    noise_sig = load_audio_file(fullfile(noise_folder, noise_files(noise_idx).name), fs, 3);
    
    fprintf('Randomly selected interferences:\n');
    fprintf('  Female: %s (idx: %d)\n', female_files(female_idx).name, female_idx);
    fprintf('  Music: %s (idx: %d)\n', music_files(music_idx).name, music_idx);
    fprintf('  Noise: %s (idx: %d)\n\n', noise_files(noise_idx).name, noise_idx);
    
    % Calculate power for mixing
    P_female = mean(female_sig.^2);
    P_music = mean(music_sig.^2);
    P_noise_base = mean(noise_sig.^2);
    
    %% Combination A: Female speech only
    fprintf('--- Combination A (Female only) ---\n');
    interf_sig_A = female_sig;
    
    [mixture_A, target_mc_A, interf_mc_A, noise_mc_A] = create_competition_mixture(...
        target_sig, interf_sig_A, fs, SNR_target, c, d, RT60_target);
    
    [SIR_A, SNR_A, RT60_A, max_amp_A] = verify_mixture_conditions(...
        mixture_A, target_mc_A, interf_mc_A, noise_mc_A, fs, RT60_target);
    
    display_verification_results(SIR_A, SNR_A, RT60_A, max_amp_A, RT60_target);
    
    output_name_A = sprintf('%s_A_female_only.wav', male_base_name);
    audiowrite(output_name_A, mixture_A, fs);
    fprintf('Saved: %s\n\n', output_name_A);
    
    %% Combination B: Female + Music
    fprintf('--- Combination B (Female + Music) ---\n');
    if P_music > 0
        music_scaled = music_sig * sqrt(P_female / P_music);
    else
        music_scaled = music_sig;
    end
    interf_sig_B = female_sig + music_scaled;
    
    [mixture_B, target_mc_B, interf_mc_B, noise_mc_B] = create_competition_mixture(...
        target_sig, interf_sig_B, fs, SNR_target, c, d, RT60_target);
    
    [SIR_B, SNR_B, RT60_B, max_amp_B] = verify_mixture_conditions(...
        mixture_B, target_mc_B, interf_mc_B, noise_mc_B, fs, RT60_target);
    
    display_verification_results(SIR_B, SNR_B, RT60_B, max_amp_B, RT60_target);
    
    output_name_B = sprintf('%s_B_female_music.wav', male_base_name);
    audiowrite(output_name_B, mixture_B, fs);
    fprintf('Saved: %s\n\n', output_name_B);
    
    %% Combination C: Female + Noise
    fprintf('--- Combination C (Female + Noise) ---\n');
    if P_noise_base > 0
        noise_scaled = noise_sig * sqrt(P_female / P_noise_base);
    else
        noise_scaled = noise_sig;
    end
    interf_sig_C = female_sig + noise_scaled;
    
    [mixture_C, target_mc_C, interf_mc_C, noise_mc_C] = create_competition_mixture(...
        target_sig, interf_sig_C, fs, SNR_target, c, d, RT60_target);
    
    [SIR_C, SNR_C, RT60_C, max_amp_C] = verify_mixture_conditions(...
        mixture_C, target_mc_C, interf_mc_C, noise_mc_C, fs, RT60_target);
    
    display_verification_results(SIR_C, SNR_C, RT60_C, max_amp_C, RT60_target);
    
    output_name_C = sprintf('%s_C_female_noise.wav', male_base_name);
    audiowrite(output_name_C, mixture_C, fs);
    fprintf('Saved: %s\n\n', output_name_C);
    
    %% Combination D: Music + Noise
    fprintf('--- Combination D (Music + Noise) ---\n');
    if P_noise_base > 0
        noise_scaled2 = noise_sig * sqrt(P_music / P_noise_base);
    else
        noise_scaled2 = noise_sig;
    end
    interf_sig_D = music_sig + noise_scaled2;
    
    [mixture_D, target_mc_D, interf_mc_D, noise_mc_D] = create_competition_mixture(...
        target_sig, interf_sig_D, fs, SNR_target, c, d, RT60_target);
    
    [SIR_D, SNR_D, RT60_D, max_amp_D] = verify_mixture_conditions(...
        mixture_D, target_mc_D, interf_mc_D, noise_mc_D, fs, RT60_target);
    
    display_verification_results(SIR_D, SNR_D, RT60_D, max_amp_D, RT60_target);
    
    output_name_D = sprintf('%s_D_music_noise.wav', male_base_name);
    audiowrite(output_name_D, mixture_D, fs);
    fprintf('Saved: %s\n\n', output_name_D);
    
    %% Combination E: Female + Music + Noise
    fprintf('--- Combination E (Female + Music + Noise) ---\n');
    interf_sig_E = female_sig + music_scaled + noise_scaled;
    
    [mixture_E, target_mc_E, interf_mc_E, noise_mc_E] = create_competition_mixture(...
        target_sig, interf_sig_E, fs, SNR_target, c, d, RT60_target);
    
    [SIR_E, SNR_E, RT60_E, max_amp_E] = verify_mixture_conditions(...
        mixture_E, target_mc_E, interf_mc_E, noise_mc_E, fs, RT60_target);
    
    display_verification_results(SIR_E, SNR_E, RT60_E, max_amp_E, RT60_target);
    
    output_name_E = sprintf('%s_E_female_music_noise.wav', male_base_name);
    audiowrite(output_name_E, mixture_E, fs);
    fprintf('Saved: %s\n\n', output_name_E);
    
    %% Save metadata with random indices
    % metadata = struct();
    % metadata.male_file = male_file.name;
    % metadata.female_file = female_files(female_idx).name;
    % metadata.female_index = female_idx;
    % metadata.music_file = music_files(music_idx).name;
    % metadata.music_index = music_idx;
    % metadata.noise_file = noise_files(noise_idx).name;
    % metadata.noise_index = noise_idx;
    % metadata.parameters.fs = fs;
    % metadata.parameters.c = c;
    % metadata.parameters.d = d;
    % metadata.parameters.RT60_target = RT60_target;
    % metadata.parameters.SIR_target = SIR_target;
    % metadata.parameters.SNR_target = SNR_target;
    % metadata.generated_date = datestr(now);
    % metadata.random_seed = random_state.Seed;
    % 
    % save(sprintf('%s_metadata.mat', male_base_name), 'metadata');
    % fprintf('Metadata saved: %s_metadata.mat\n', male_base_name);
    
    success_count = success_count + 1;
    
    fprintf('=== Summary for %s ===\n', male_base_name);
    fprintf('Generated 5 mixtures:\n');
    fprintf('1. %s\n', output_name_A);
    fprintf('2. %s\n', output_name_B);
    fprintf('3. %s\n', output_name_C);
    fprintf('4. %s\n', output_name_D);
    fprintf('5. %s\n\n', output_name_E);
end















% %% Final summary
% fprintf('\n=== GENERATION COMPLETE ===\n');
% fprintf('Successfully processed %d male files.\n', success_count);
% fprintf('Generated %d mixtures in total.\n', success_count * 5);
% fprintf('\nRandom seed used: %d\n', random_state.Seed);
% fprintf('To reproduce exactly, run: rng(%d)\n', random_state.Seed);
% fprintf('\nFile naming convention:\n');
% fprintf('[MaleBaseName]_[A-E]_[interference_type].wav\n');
% fprintf('Example: 1000_part1_A_female_only.wav\n');
% fprintf('\nEach file contains 2 channels (2 microphones).\n');
% fprintf('All files are 3 seconds long at 16 kHz.\n');
% 
% % Create summary file with random seed info
% summary_file = 'generation_summary.txt';
% fid = fopen(summary_file, 'w');
% fprintf(fid, 'Competition Mixture Generation Summary\n');
% fprintf(fid, 'Generated: %s\n\n', datestr(now));
% fprintf(fid, 'Parameters:\n');
% fprintf(fid, '  Sampling rate: %d Hz\n', fs);
% fprintf(fid, '  RT60 target: %.1f s\n', RT60_target);
% fprintf(fid, '  SIR target: %d dB\n', SIR_target);
% fprintf(fid, '  SNR target: %d dB\n', SNR_target);
% fprintf(fid, '  Mic spacing: %.2f m\n', d);
% fprintf(fid, '  Speed of sound: %.0f m/s\n', c);
% fprintf(fid, '  Random seed: %d\n\n', random_state.Seed);
% fprintf(fid, 'Generated %d sets of 5 mixtures each.\n', success_count);
% fprintf(fid, '\nCombinations:\n');
% fprintf(fid, '  A: Female speech only\n');
% fprintf(fid, '  B: Female speech + Music\n');
% fprintf(fid, '  C: Female speech + Noise\n');
% fprintf(fid, '  D: Music + Noise\n');
% fprintf(fid, '  E: Female speech + Music + Noise\n');
% fclose(fid);
