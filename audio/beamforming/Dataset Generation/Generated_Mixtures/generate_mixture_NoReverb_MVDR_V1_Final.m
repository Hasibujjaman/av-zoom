%% Generate Competition Mixtures - FINAL WORKING deepseek
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

clear; clc; close all;

%% Randomization Seed
% rng(42);
rng('shuffle');

%% Process N male files
n_to_process = 42805;
%% apply mvdr ?
apply_mvdr_to_mixture = true;

%% Parameters (as per competition requirements)
fs = 16000;                 % Sampling rate (Hz)
c = 340;                    % Speed of sound (m/s)
d = 0.08;                   % Microphone spacing (m)

SIR_target = 0;             % Signal-to-Interference Ratio (dB)
SNR_target = 5;             % Signal-to-Noise Ratio (dB)

%% -------- MVDR Params --------
theta_target = 0;
theta_noise  = 40;
theta_target_test = 0; 
mic_pos = [-d/2; d/2];
% STFT params
N = 256;
hop = 128;
nfft = 512;
window = sqrt(hann(N,'periodic'));
%ISTFT params
L = 3*fs;



% Display parameters
fprintf('=== Competition Mixture Generation ===\n');
fprintf('Target Parameters:\n');
fprintf('  SIR: %d dB\n', SIR_target);
fprintf('  SNR: %d dB\n', SNR_target);
fprintf('  Fs: %d Hz\n', fs);
fprintf('  c: %.0f m/s\n', c);
fprintf('  d: %.2f m\n\n', d);



%% Mixture Creation Function
function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_mixture_no_reverb(target, interf, theta_target, ...
                   theta_interf, fs, SNR_dB, c, d)

% =================================================
% Geometry: 2-mic linear array
% =================================================
micPos = [-d/2; d/2];     % meters (x-axis)

% =================================================
% Length match
% =================================================
L = min(length(target), length(interf));
target = target(1:L);
interf = interf(1:L);

% =================================================
% Fractional delay (plane wave)
% =================================================
apply_delay = @(x, tau) interp1( ...
    (0:length(x)-1)', x, ...
    (0:length(x)-1)' - tau*fs, ...
    'linear', 0);

make_multichannel = @(sig, theta) ...
    [ apply_delay(sig, micPos(1)*sin(deg2rad(theta))/c), ...
      apply_delay(sig, micPos(2)*sin(deg2rad(theta))/c) ];

% =================================================
% Multichannel signals
% =================================================
target_mc = make_multichannel(target, theta_target);
interf_mc = make_multichannel(interf, theta_interf);

% =================================================
% SIR CONTROL (0 dB) — USE BOTH CHANNELS
% =================================================
Pt = mean(target_mc(:).^2);
Pi = mean(interf_mc(:).^2);

interf_mc = interf_mc * sqrt(Pt / Pi);   % SIR = 0 dB

% =================================================
% Clean mixture
% =================================================
mixture_clean = target_mc + interf_mc;

% =================================================
% Add white Gaussian noise (SNR = SNR_dB)
% =================================================
mixture  = zeros(size(mixture_clean));
noise_mc = zeros(size(mixture_clean));

for m = 1:size(mixture_clean,2)
    mixture(:,m) = awgn(mixture_clean(:,m), ...
                        SNR_dB, 'measured');
    noise_mc(:,m) = mixture(:,m) - mixture_clean(:,m);
end

% =================================================
% FINAL NORMALIZATION (CRITICAL)
% Prevent loud hiss & audiowrite clipping
% =================================================
peak = max(abs(mixture(:)));
if peak > 0
    mixture  = 0.99 * mixture  / peak;
    target_mc = 0.99 * target_mc / peak;
    interf_mc = 0.99 * interf_mc / peak;
    noise_mc  = 0.99 * noise_mc  / peak;
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
function [SIR_actual, SNR_actual, max_amp] = verify_mixture_conditions(mixture, target_mc, interf_mc, noise_mc, fs)
    
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
    
    % Check max amplitude
    max_amp = max(abs(mixture(:)));
end

%% Display verification results
function display_verification_results(SIR_actual, SNR_actual, max_amp)
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
  
    fprintf('Max amplitude: %.3f ', max_amp);
    if max_amp <= 0.95
        fprintf('✓ (safe)\n');
    elseif max_amp <= 0.99
        fprintf('⚠ (near clipping)\n');
    else
        fprintf('✗ (clipping)\n');
    end
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
music_files = dir(fullfile(music_folder, '*flac'));
noise_files = dir(fullfile(noise_folder, '*flac'));

fprintf('Found files:\n');
fprintf('  Male: %d\n', length(male_files));
fprintf('  Female: %d\n', length(female_files));
fprintf('  Music: %d\n', length(music_files));
fprintf('  Noise: %d\n\n', length(noise_files));



n_processed_male_samples = 1;

for i = 1:min(n_to_process, length(male_files))
    % select random male file index
    male_idx = randi(length(male_files)); 
    fprintf('\n\n(iteration %d)\n=== Processing Male File %d/%d ===\n',n_processed_male_samples, male_idx, min(n_to_process, length(male_files)));
    
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
    
    % Select different interference files
    % female_idx = mod(male_idx, length(female_files)) + 1;
    % music_idx = mod(male_idx + 100, length(music_files)) + 1;
    % noise_idx = mod(male_idx + 200, length(noise_files)) + 1;
    female_idx = randi(length(female_files));
    music_idx  = randi(length(music_files));
    noise_idx  = randi(length(noise_files));
    
    % Load interference components
    female_sig = load_audio_file(fullfile(female_folder, female_files(female_idx).name), fs, 3);
    %%
    % female_sig = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation/Female/2_part22.flac');
    %%
    music_sig = load_audio_file(fullfile(music_folder, music_files(music_idx).name), fs, 3);
    noise_sig = load_audio_file(fullfile(noise_folder, noise_files(noise_idx).name), fs, 3);
    
    fprintf('Interferences:\n');
    fprintf('  Female: %s\n', female_files(female_idx).name);
    fprintf('  Music: %s\n', music_files(music_idx).name);
    fprintf('  Noise: %s\n\n', noise_files(noise_idx).name);
    
    % Calculate power for mixing
    P_female = mean(female_sig.^2);
    P_music = mean(music_sig.^2);
    P_noise_base = mean(noise_sig.^2);
    
    %% Combination A: Female speech only
    fprintf('--- Combination A (Female only) ---\n');
    interf_sig_A = female_sig;
    
    [mixture_A, target_mc_A, interf_mc_A, noise_mc_A] = create_mixture_no_reverb(...
        target_sig, interf_sig_A, theta_target_test, theta_noise, fs, SNR_target, c, d);
    
    [SIR_A, SNR_A, max_amp_A] = verify_mixture_conditions(...
        mixture_A, target_mc_A, interf_mc_A, noise_mc_A, fs);
    
    display_verification_results(SIR_A, SNR_A, max_amp_A);
    
    output_name_A = sprintf('%s_A_female_only.wav', male_base_name);

       
    % Apply mvdr
    if apply_mvdr_to_mixture
        x = mixture_A;
        X = stft_multichannel(x, window, hop, nfft);
        [numFreqs, numFrames, numMics] = size(X);

        freqs = (0:numFreqs-1)' * fs / nfft;

        % -------- Covariance --------
        Rxx = init_covariance(numFreqs, numMics);
        alpha = 0.98;

        for n = 1:numFrames
            X_frame = squeeze(X(:,n,:));   % [freq x mic]
            Rxx = update_covariance(Rxx, X_frame, alpha);
        end

        % -------- Steering vector --------
        dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

        % -------- MVDR weights --------
        delta = 1e-3;
        W = compute_mvdr_weights(Rxx, dvec, delta);

        % -------- Apply MVDR --------
        Y = apply_mvdr(X, W);

        % -------- ISTFT --------
        y_mvdr = istft_single_channel(Y, window, hop, nfft, L);

        % Crop to original signal length
        y_mvdr_yours = real(y_mvdr);

        mixture_A = y_mvdr_yours;

    end

    % save output
    audiowrite(output_name_A, mixture_A, fs);
    % audiowrite(output_name_A, y_mvdr_yours, fs);
    fprintf('Saved: %s\n\n', output_name_A);
    
    %% Combination B: Female + Music
    fprintf('--- Combination B (Female + Music) ---\n');
    if P_music > 0
        music_scaled = music_sig * sqrt(P_female / P_music);
    else
        music_scaled = music_sig;
    end
    interf_sig_B = female_sig + music_scaled;
    
    [mixture_B, target_mc_B, interf_mc_B, noise_mc_B] = create_mixture_no_reverb(...
        target_sig, interf_sig_B, theta_target_test, theta_noise, fs, SNR_target, c, d);
    
    [SIR_B, SNR_B, max_amp_B] = verify_mixture_conditions(...
        mixture_B, target_mc_B, interf_mc_B, noise_mc_B, fs);
    
    display_verification_results(SIR_B, SNR_B, max_amp_B);
    
    output_name_B = sprintf('%s_B_female_music.wav', male_base_name);

    if apply_mvdr_to_mixture
    % Apply mvdr
    x = mixture_B;
    X = stft_multichannel(x, window, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    freqs = (0:numFreqs-1)' * fs / nfft;

    % -------- Covariance --------
    Rxx = init_covariance(numFreqs, numMics);
    alpha = 0.98;

    for n = 1:numFrames
        X_frame = squeeze(X(:,n,:));   % [freq x mic]
        Rxx = update_covariance(Rxx, X_frame, alpha);
    end

    % -------- Steering vector --------
    dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

    % -------- MVDR weights --------
    delta = 1e-3;
    W = compute_mvdr_weights(Rxx, dvec, delta);

    % -------- Apply MVDR --------
    Y = apply_mvdr(X, W);

    % -------- ISTFT --------
    y_mvdr = istft_single_channel(Y, window, hop, nfft, L);

    % Crop to original signal length
    y_mvdr_yours = real(y_mvdr);

    mixture_B = y_mvdr_yours;
    
    end

    % save output
    audiowrite(output_name_B, mixture_B, fs);
    % audiowrite(output_name_B, y_mvdr_yours, fs);
    fprintf('Saved: %s\n\n', output_name_B);
    
    %% Combination C: Female + Noise
    fprintf('--- Combination C (Female + Noise) ---\n');
    if P_noise_base > 0
        noise_scaled = noise_sig * sqrt(P_female / P_noise_base);
    else
        noise_scaled = noise_sig;
    end
    interf_sig_C = female_sig + noise_scaled;
    
    [mixture_C, target_mc_C, interf_mc_C, noise_mc_C] = create_mixture_no_reverb(...
        target_sig, interf_sig_C, theta_target_test, theta_noise, fs, SNR_target, c, d);
    
    [SIR_C, SNR_C, max_amp_C] = verify_mixture_conditions(...
        mixture_C, target_mc_C, interf_mc_C, noise_mc_C, fs);
    
    display_verification_results(SIR_C, SNR_C, max_amp_C);
    
    output_name_C = sprintf('%s_C_female_noise.wav', male_base_name);

    if apply_mvdr_to_mixture
    % Apply mvdr
    x = mixture_C;
    X = stft_multichannel(x, window, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    freqs = (0:numFreqs-1)' * fs / nfft;

    % -------- Covariance --------
    Rxx = init_covariance(numFreqs, numMics);
    alpha = 0.98;

    for n = 1:numFrames
        X_frame = squeeze(X(:,n,:));   % [freq x mic]
        Rxx = update_covariance(Rxx, X_frame, alpha);
    end

    % -------- Steering vector --------
    dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

    % -------- MVDR weights --------
    delta = 1e-3;
    W = compute_mvdr_weights(Rxx, dvec, delta);

    % -------- Apply MVDR --------
    Y = apply_mvdr(X, W);

    % -------- ISTFT --------
    y_mvdr = istft_single_channel(Y, window, hop, nfft, L);

    % Crop to original signal length
    y_mvdr_yours = real(y_mvdr);

    mixture_C = y_mvdr_yours;

    end
       

    % save output
    audiowrite(output_name_C, mixture_C, fs);
    % audiowrite(output_name_C, y_mvdr_yours, fs);
    fprintf('Saved: %s\n\n', output_name_C);
    
    %% Combination D: Music + Noise
    fprintf('--- Combination D (Music + Noise) ---\n');
    if P_noise_base > 0
        noise_scaled2 = noise_sig * sqrt(P_music / P_noise_base);
    else
        noise_scaled2 = noise_sig;
    end
    interf_sig_D = music_sig + noise_scaled2;
    
    [mixture_D, target_mc_D, interf_mc_D, noise_mc_D] = create_mixture_no_reverb(...
        target_sig, interf_sig_D, theta_target_test, theta_noise, fs, SNR_target, c, d);
    
    [SIR_D, SNR_D, max_amp_D] = verify_mixture_conditions(...
        mixture_D, target_mc_D, interf_mc_D, noise_mc_D, fs);
    
    display_verification_results(SIR_D, SNR_D, max_amp_D);
    
    output_name_D = sprintf('%s_D_music_noise.wav', male_base_name);

    
    if apply_mvdr_to_mixture
    % Apply mvdr
    x = mixture_D;
    X = stft_multichannel(x, window, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    freqs = (0:numFreqs-1)' * fs / nfft;

    % -------- Covariance --------
    Rxx = init_covariance(numFreqs, numMics);
    alpha = 0.98;

    for n = 1:numFrames
        X_frame = squeeze(X(:,n,:));   % [freq x mic]
        Rxx = update_covariance(Rxx, X_frame, alpha);
    end

    % -------- Steering vector --------
    dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

    % -------- MVDR weights --------
    delta = 1e-3;
    W = compute_mvdr_weights(Rxx, dvec, delta);

    % -------- Apply MVDR --------
    Y = apply_mvdr(X, W);

    % -------- ISTFT --------
    y_mvdr = istft_single_channel(Y, window, hop, nfft, L);

    % Crop to original signal length
    y_mvdr_yours = real(y_mvdr);

    mixture_D = y_mvdr_yours;

    end

    % save output
    % audiowrite(output_name_D, y_mvdr_yours, fs);
    audiowrite(output_name_D, mixture_D, fs);
    fprintf('Saved: %s\n\n', output_name_D);
    
    %% Combination E: Female + Music + Noise
    fprintf('--- Combination E (Female + Music + Noise) ---\n');
    interf_sig_E = female_sig + music_scaled + noise_scaled;
    
    [mixture_E, target_mc_E, interf_mc_E, noise_mc_E] = create_mixture_no_reverb(...
        target_sig, interf_sig_B, theta_target_test, theta_noise, fs, SNR_target, c, d);
    
    [SIR_E, SNR_E, max_amp_E] = verify_mixture_conditions(...
        mixture_E, target_mc_E, interf_mc_E, noise_mc_E, fs);
    
    display_verification_results(SIR_E, SNR_E, max_amp_E);

    output_name_E = sprintf('%s_E_female_music_noise.wav', male_base_name);

    if apply_mvdr_to_mixture
    % Apply mvdr
    x = mixture_E;
    X = stft_multichannel(x, window, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    freqs = (0:numFreqs-1)' * fs / nfft;

    % -------- Covariance --------
    Rxx = init_covariance(numFreqs, numMics);
    alpha = 0.98;

    for n = 1:numFrames
        X_frame = squeeze(X(:,n,:));   % [freq x mic]
        Rxx = update_covariance(Rxx, X_frame, alpha);
    end

    % -------- Steering vector --------
    dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);

    % -------- MVDR weights --------
    delta = 1e-3;
    W = compute_mvdr_weights(Rxx, dvec, delta);

    % -------- Apply MVDR --------
    Y = apply_mvdr(X, W);

    % -------- ISTFT --------
    y_mvdr = istft_single_channel(Y, window, hop, nfft, L);

    % Crop to original signal length
    y_mvdr_yours = real(y_mvdr);

    mixture_E = y_mvdr_yours;

    end

    % save output
    % audiowrite(output_name_E, y_mvdr_yours, fs);
    audiowrite(output_name_E, mixture_E, fs);
    fprintf('Saved: %s\n\n', output_name_E);
    
    %% Save metadata
    % metadata = struct();
    % metadata.male_file = male_file.name;
    % metadata.female_file = female_files(female_idx).name;
    % metadata.music_file = music_files(music_idx).name;
    % metadata.noise_file = noise_files(noise_idx).name;
    % metadata.parameters.fs = fs;
    % metadata.parameters.c = c;
    % metadata.parameters.d = d;
    % metadata.parameters.RT60_target = RT60_target;
    % metadata.parameters.SIR_target = SIR_target;
    % metadata.parameters.SNR_target = SNR_target;
    % metadata.generated_date = datestr(now);
    % 
    % save(sprintf('%s_metadata.mat', male_base_name), 'metadata');
    % fprintf('Metadata saved: %s_metadata.mat\n', male_base_name);
    % 
    % 
    % fprintf('=== Summary for %s ===\n', male_base_name);
    % fprintf('Generated 5 mixtures:\n');
    % fprintf('1. %s\n', output_name_A);
    % fprintf('2. %s\n', output_name_B);
    % fprintf('3. %s\n', output_name_C);
    % fprintf('4. %s\n', output_name_D);
    % fprintf('5. %s\n\n', output_name_E);

    n_processed_male_samples = n_processed_male_samples + 1;
end

%% Final summary
fprintf('\n=== GENERATION COMPLETE ===\n');
fprintf('Generated %d mixtures in total.\n',  n_to_process* 5);
fprintf('\nFile naming convention:\n');
fprintf('[MaleBaseName]_[A-E]_[interference_type].wav\n');
fprintf('Example: 1000_part1_A_female_only.wav\n');
fprintf('\nEach file contains 2 channels (2 microphones).\n');
fprintf('All files are 3 seconds long at 16 kHz.\n');

% % Create summary file
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
% fprintf(fid, '  Speed of sound: %.0f m/s\n\n', c);
% fprintf(fid, 'Generated %d sets of 5 mixtures each.\n', n_to_process);
% fclose(fid);
% 
% fprintf('Summary saved to: %s\n', summary_file);