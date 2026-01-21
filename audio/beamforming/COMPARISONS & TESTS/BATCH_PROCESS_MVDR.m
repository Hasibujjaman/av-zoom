%% ############ BATCH MVDR PROCESSING SCRIPT ############
% This script processes multiple clean/noise combinations through MVDR beamforming
% Input: Pre_MVDR_Dataset (Male, Female, Music, Noise folders)
% Output: Post_MVDR_Dataset (organized by noise type)

clear; clc;

%% ================= SETUP PATHS =================
% Add necessary paths
addpath('../');  % beamforming folder
addpath('../Taki');
addpath('../Metrics');

% Define input and output directories
input_base = 'D:\Dataset\av-zoom\Pre_MVDR_Dataset';
output_base = 'D:\Dataset\av-zoom\Post_MVDR_Dataset';

% Create output directory
if ~exist(output_base, 'dir')
    mkdir(output_base);
end

%% ================= BEAMFORMING CONFIG =================
fs = 16000;
theta_target = 0;       % Target speaker direction (degrees)
theta_noise  = 40;      % Noise direction (degrees)
theta_target_test = 0;  % Test direction for MVDR steering (Your MVDR)

% Mixture Conditions
SIR_dB       = 0;       % Signal-to-Interference Ratio: 0 dB (equal power target & interferer)
SNR_dB       = 5;       % Signal-to-Noise Ratio: 5 dB (additive white Gaussian sensor noise)

c = 340;                % Speed of sound (m/s)
d = 0.08;               % Microphone spacing (8 cm)
mic_pos = [-d/2; d/2];

% STFT parameters
N = 256;
hop = 128;
nfft = 512;
window = sqrt(hann(N,'periodic'));

% Covariance update parameter
alpha = 0.98;

% Diagonal loading factor for MVDR
delta = 1e-3;

%% ================= GET FILE LISTS =================
fprintf('Scanning input directories...\n');

male_files = dir(fullfile(input_base, 'Male', '*.flac'));
female_files = dir(fullfile(input_base, 'Female', '*.flac'));
music_files = dir(fullfile(input_base, 'Music', '*.flac'));
noise_files = dir(fullfile(input_base, 'Noise', '*.flac'));

fprintf('Found:\n');
fprintf('  Male: %d files\n', length(male_files));
fprintf('  Female: %d files\n', length(female_files));
fprintf('  Music: %d files\n', length(music_files));
fprintf('  Noise: %d files\n', length(noise_files));

%% ================= DEFINE PROCESSING SCENARIOS =================
scenarios = {
    % {noise_type, noise_folders, output_subfolder}
    {'Female', {'Female'}, 'Female'}
    {'Noise', {'Noise'}, 'Noise'}
    {'Music', {'Music'}, 'Music'}
    {'Female+Noise', {'Female', 'Noise'}, 'Female_Noise'}
    {'Female+Noise+Music', {'Female', 'Noise', 'Music'}, 'Female_Noise_Music'}
};

%% ================= PROCESS EACH SCENARIO =================
for scenario_idx = 5:length(scenarios)
    scenario_name = scenarios{scenario_idx}{1};
    noise_folders = scenarios{scenario_idx}{2};
    output_folder = scenarios{scenario_idx}{3};
    
    fprintf('\n========================================\n');
    fprintf('Processing Scenario: %s\n', scenario_name);
    fprintf('========================================\n');
    
    % Create output directory for this scenario
    output_dir = fullfile(output_base, output_folder);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Get noise file lists for this scenario
    all_noise_files = {};
    for nf_idx = 1:length(noise_folders)
        nf_name = noise_folders{nf_idx};
        
        % Use pre-scanned file lists
        if strcmp(nf_name, 'Female')
            nf_list = female_files;
        elseif strcmp(nf_name, 'Music')
            nf_list = music_files;
        elseif strcmp(nf_name, 'Noise')
            nf_list = noise_files;
        else
            error('Unknown noise folder: %s', nf_name);
        end
        
        % Build full paths
        for i = 1:length(nf_list)
            all_noise_files{end+1} = fullfile(input_base, nf_name, nf_list(i).name);
        end
    end
    
    fprintf('Total noise files for this scenario: %d\n', length(all_noise_files));
    
    if isempty(all_noise_files)
        error('No noise files found for scenario: %s', scenario_name);
    end
    
    % Process each male (clean) file
    for male_idx = 1:length(male_files)
        clean_path = fullfile(input_base, 'Male', male_files(male_idx).name);
        [~, clean_name, ~] = fileparts(male_files(male_idx).name);
        
        fprintf('\n[%d/%d] Processing clean file: %s\n', male_idx, length(male_files), clean_name);
        
        % Load clean signal
        [s_clean, fs_clean] = audioread(clean_path);
        if fs_clean ~= fs
            error('Sample rate mismatch: expected %d, got %d', fs, fs_clean);
        end
        
        % Process with each noise file
        noise_idx = mod(male_idx - 1, length(all_noise_files)) + 1;
        noise_path = all_noise_files{noise_idx};
        [~, noise_name, ~] = fileparts(noise_path);
        
        fprintf('  Using noise: %s\n', noise_name);
        
        % Load noise signal
        [v_noise, fs_noise] = audioread(noise_path);
        if fs_noise ~= fs
            error('Sample rate mismatch: expected %d, got %d', fs, fs_noise);
        end
        
        % Match lengths
        L = min(length(s_clean), length(v_noise));
        s_clean = s_clean(1:L);
        v_noise = v_noise(1:L);
        
        % Normalize for SIR = 0 dB (equal power)
        % Scale interferer to match target power
        power_clean = mean(s_clean.^2);
        power_noise = mean(v_noise.^2);
        v_noise = v_noise * sqrt(power_clean / power_noise);
        
        % For multi-noise scenarios, combine noise sources
        if length(noise_folders) > 1
            % Mix additional noise sources
            for extra_idx = 2:length(noise_folders)
                extra_noise_idx = mod(male_idx + extra_idx - 2, length(all_noise_files)) + 1;
                extra_noise_path = all_noise_files{extra_noise_idx};
                [v_extra, ~] = audioread(extra_noise_path);
                v_extra = v_extra(1:L);
        % SIR = 0 dB (normalized above), SNR = 5 dB (white Gaussian noise added by create_mixture)
                % Equal power mixing
                v_noise = (v_noise + v_extra) / sqrt(2);
            end
        end
        
        % Create multichannel mixture
        [x, target_mc, interf_mc, noise_mc] = create_mixture( s_clean, v_noise, ...
            theta_target, theta_noise, fs, SNR_dB, c, d);
        
        x = x(1:L,:);
        x_mono = x(:,1);
        
        %% ================= YOUR MVDR (UNCHANGED) =================
        % -------- STFT params --------
        X = stft_multichannel(x, window, hop, nfft);
        [numFreqs, numFrames, numMics] = size(X);
        
        freqs = (0:numFreqs-1) * fs / nfft;
        
        % -------- Covariance --------
        Rxx = init_covariance(numFreqs, numMics);
        
        for n = 1:numFrames
            X_frame = squeeze(X(:,n,:));   % [freq x mic]
            Rxx = update_covariance(Rxx, X_frame, alpha);
        end
        
        % -------- Steering vector --------
        dvec = compute_steering_vector(theta_target_test, freqs, mic_pos, c);
        
        % -------- MVDR weights --------
        W = compute_mvdr_weights(Rxx, dvec, delta);
        
        % -------- Apply MVDR --------
        Y = apply_mvdr(X, W);
        
        % -------- ISTFT --------
        y = istft_single_channel(Y, window, hop, nfft, L);
        
        % Crop to original signal length
        y_mvdr_yours = real(y(1:L));
        
        %% ================= COMPUTE METRICS =================
        % Align lengths
        Lmin = min([length(s_clean), length(x_mono), length(y_mvdr_yours)]);
        s  = s_clean(1:Lmin);
        x0 = x_mono(1:Lmin);
        y1 = y_mvdr_yours(1:Lmin);
        
        % SI-SDR
        sisdr_mic = si_sdr(x0, s);
        sisdr_yours = si_sdr(y1, s);
        
        % OSINR
        res_mic = x0 - s;
        res_yours = y1 - s;
        osinr_mic = 10*log10( sum(s.^2) / (sum(res_mic.^2) + 1e-12) );
        osinr_yours = 10*log10( sum(s.^2) / (sum(res_yours.^2) + 1e-12) );
        
        fprintf('  SI-SDR: Mic=%.2f dB, Yours=%.2f dB (Δ=%.2f dB)\n', ...
            sisdr_mic, sisdr_yours, sisdr_yours - sisdr_mic);
        fprintf('  OSINR:  Mic=%.2f dB, Yours=%.2f dB (Δ=%.2f dB)\n', ...
            osinr_mic, osinr_yours, osinr_yours - osinr_mic);
        
        %% ================= SAVE OUTPUTS =================
        % Create output filename
        output_filename = sprintf('%s_MVDR.wav', clean_name);
        output_path = fullfile(output_dir, output_filename);
        
        % Save YOUR MVDR output
        audiowrite(output_path, y_mvdr_yours, fs);
        
        fprintf('  Saved: %s\n', output_filename);
    end
end

fprintf('\n========================================\n');
fprintf('All processing complete!\n');
fprintf('========================================\n');
fprintf('Output directory: %s\n', output_base);
