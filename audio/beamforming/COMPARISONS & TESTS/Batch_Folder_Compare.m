%% ############ BATCH FOLDER COMPARISON SCRIPT ############
% Compares clean reference, MVDR outputs, and model enhanced outputs
% across all files in specified folders and computes average metrics.
clear; clc;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation');

%% ================= FOLDER PATHS (MODIFY THESE) =================
clean_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_NonRerverb/clean';      % Folder with clean reference files
mvdr_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_NonRerverb/noisy'; % Folder with MVDR output files
% clean_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_Reverb/Compensated/clean/Test';      % Folder with clean reference files
% mvdr_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_Reverb/Compensated/MVDR_filtered/Test'; % Folder with MVDR output files
model_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_NonRerverb/rnnoise_enhanced'; % Folder with model output files

%% ================= CONFIG =================
fs = 16000;  % Expected sample rate

% Supported audio extensions
audio_extensions = {'*.wav', '*.flac', '*.mp3', '*.m4a'};

%% ================= GET FILE LIST =================
% Get all audio files from clean folder
clean_files = [];
for ext = audio_extensions
    clean_files = [clean_files; dir(fullfile(clean_folder, ext{1}))];
end

numFiles = length(clean_files);
if numFiles == 0
    error('No audio files found in clean folder: %s', clean_folder);
end

fprintf('Found %d audio files in clean folder.\n', numFiles);

%% ================= INITIALIZE METRIC ARRAYS =================
% MVDR metrics
sisdr_mvdr_all = zeros(numFiles, 1);
stoi_mvdr_all = zeros(numFiles, 1);
visqol_mos_mvdr_all = zeros(numFiles, 1);
visqol_nsim_mvdr_all = zeros(numFiles, 1);

% Model metrics
sisdr_model_all = zeros(numFiles, 1);
stoi_model_all = zeros(numFiles, 1);
visqol_mos_model_all = zeros(numFiles, 1);
visqol_nsim_model_all = zeros(numFiles, 1);

% Track successfully processed files
valid_count = 0;
failed_files = {};
processed_filenames = cell(numFiles, 1);  % Store processed file names

%% ================= PROCESS EACH FILE =================
for i = 1:numFiles
    clean_filename = clean_files(i).name;
    [~, basename, ext] = fileparts(clean_filename);
    
    % Construct file paths
    clean_path = fullfile(clean_folder, clean_filename);
    mvdr_path = fullfile(mvdr_folder, clean_filename);
    model_path = fullfile(model_folder, [basename '_enhanced' ext]);
    
    % Check if all files exist
    if ~isfile(mvdr_path)
        fprintf('WARNING: MVDR file not found: %s\n', mvdr_path);
        failed_files{end+1} = clean_filename;
        continue;
    end
    if ~isfile(model_path)
        fprintf('WARNING: Model file not found: %s\n', model_path);
        failed_files{end+1} = clean_filename;
        continue;
    end
    
    fprintf('\n[%d/%d] Processing: %s\n', i, numFiles, clean_filename);
    
    try
        % -------- Load audio files --------
        [s_clean, fs1] = audioread(clean_path);
        [y_mvdr, fs2] = audioread(mvdr_path);
        [y_model, fs3] = audioread(model_path);
        
        % Ensure mono signals
        if size(s_clean, 2) > 1
            s_clean = s_clean(:, 1);
        end
        if size(y_mvdr, 2) > 1
            y_mvdr = y_mvdr(:, 1);
        end
        if size(y_model, 2) > 1
            y_model = y_model(:, 1);
        end
        
        % Check sample rates
        if fs1 ~= fs || fs2 ~= fs || fs3 ~= fs
            fprintf('WARNING: Sample rate mismatch for %s (clean=%d, mvdr=%d, model=%d)\n', ...
                clean_filename, fs1, fs2, fs3);
            failed_files{end+1} = clean_filename;
            continue;
        end
        
        % -------- Align lengths --------
        Lmin = min([length(s_clean), length(y_mvdr), length(y_model)]);
        s = s_clean(1:Lmin);
        
        % -------- Align MVDR to clean --------
        [y_mvdr_aligned, ~] = alignsignals(y_mvdr, s);
        y_mvdr_aligned = y_mvdr_aligned(1:Lmin);
        
        % -------- Align Model to MVDR (since model is post-processed MVDR) --------
        [y_model_aligned, ~] = alignsignals(y_model, y_mvdr_aligned);
        y_model_aligned = y_model_aligned(1:Lmin);
        
        % -------- Calculate Metrics --------
        
        % SI-SDR
        sisdr_mvdr = si_sdr(y_mvdr_aligned, s);
        sisdr_model = si_sdr(y_model_aligned, s);
        
        % STOI
        stoi_mvdr = stoi(s, y_mvdr_aligned, fs);
        stoi_model = stoi(s, y_model_aligned, fs);
        
        % ViSQOL
        [visqol_mvdr, ~, ~] = visqol(y_mvdr_aligned, s, fs, mode='speech', OutputMetric="MOS and NSIM");
        [visqol_model, ~, ~] = visqol(y_model_aligned, s, fs, mode='speech', OutputMetric="MOS and NSIM");
        
        % -------- Store results --------
        valid_count = valid_count + 1;
        processed_filenames{valid_count} = clean_filename;  % Track filename
        
        sisdr_mvdr_all(valid_count) = sisdr_mvdr;
        stoi_mvdr_all(valid_count) = stoi_mvdr;
        visqol_mos_mvdr_all(valid_count) = visqol_mvdr(1);
        visqol_nsim_mvdr_all(valid_count) = visqol_mvdr(2);
        
        sisdr_model_all(valid_count) = sisdr_model;
        stoi_model_all(valid_count) = stoi_model;
        visqol_mos_model_all(valid_count) = visqol_model(1);
        visqol_nsim_model_all(valid_count) = visqol_model(2);
        
        fprintf('  SI-SDR: MVDR=%.2f dB, Model=%.2f dB\n', sisdr_mvdr, sisdr_model);
        fprintf('  STOI:   MVDR=%.3f, Model=%.3f\n', stoi_mvdr, stoi_model);
        fprintf('  ViSQOL MOS: MVDR=%.2f, Model=%.2f\n', visqol_mvdr(1), visqol_model(1));
        
    catch ME
        fprintf('ERROR processing %s: %s\n', clean_filename, ME.message);
        failed_files{end+1} = clean_filename;
        continue;
    end
end

%% ================= TRIM ARRAYS TO VALID COUNT =================
sisdr_mvdr_all = sisdr_mvdr_all(1:valid_count);
stoi_mvdr_all = stoi_mvdr_all(1:valid_count);
visqol_mos_mvdr_all = visqol_mos_mvdr_all(1:valid_count);
visqol_nsim_mvdr_all = visqol_nsim_mvdr_all(1:valid_count);

sisdr_model_all = sisdr_model_all(1:valid_count);
stoi_model_all = stoi_model_all(1:valid_count);
visqol_mos_model_all = visqol_mos_model_all(1:valid_count);
visqol_nsim_model_all = visqol_nsim_model_all(1:valid_count);
processed_filenames = processed_filenames(1:valid_count);

%% ================= COMPUTE AVERAGES =================
fprintf('\n\n');
fprintf('===============================================\n');
fprintf('         BATCH COMPARISON RESULTS             \n');
fprintf('===============================================\n');
fprintf('Successfully processed: %d / %d files\n', valid_count, numFiles);

if ~isempty(failed_files)
    fprintf('\nFailed files:\n');
    for j = 1:length(failed_files)
        fprintf('  - %s\n', failed_files{j});
    end
end

if valid_count > 0
    fprintf('\n--------------- AVERAGE METRICS ---------------\n');
    
    fprintf('\nSI-SDR (dB):\n');
    fprintf('  MVDR:   %.2f ± %.2f\n', mean(sisdr_mvdr_all, 'omitnan'), std(sisdr_mvdr_all, 'omitnan'));
    fprintf('  Model:  %.2f ± %.2f\n', mean(sisdr_model_all, 'omitnan'), std(sisdr_model_all, 'omitnan'));
    fprintf('  Improvement: %.2f dB\n', mean(sisdr_model_all - sisdr_mvdr_all, 'omitnan'));
    
    fprintf('\nSTOI:\n');
    fprintf('  MVDR:   %.3f ± %.3f\n', mean(stoi_mvdr_all, 'omitnan'), std(stoi_mvdr_all, 'omitnan'));
    fprintf('  Model:  %.3f ± %.3f\n', mean(stoi_model_all, 'omitnan'), std(stoi_model_all, 'omitnan'));
    fprintf('  Improvement: %.3f\n', mean(stoi_model_all - stoi_mvdr_all, 'omitnan'));
    
    fprintf('\nViSQOL MOS [1-5]:\n');
    fprintf('  MVDR:   %.2f ± %.2f\n', mean(visqol_mos_mvdr_all, 'omitnan'), std(visqol_mos_mvdr_all, 'omitnan'));
    fprintf('  Model:  %.2f ± %.2f\n', mean(visqol_mos_model_all, 'omitnan'), std(visqol_mos_model_all, 'omitnan'));
    fprintf('  Improvement: %.2f\n', mean(visqol_mos_model_all - visqol_mos_mvdr_all, 'omitnan'));
    
    fprintf('\nViSQOL NSIM [-1 to 1]:\n');
    fprintf('  MVDR:   %.3f ± %.3f\n', mean(visqol_nsim_mvdr_all, 'omitnan'), std(visqol_nsim_mvdr_all, 'omitnan'));
    fprintf('  Model:  %.3f ± %.3f\n', mean(visqol_nsim_model_all, 'omitnan'), std(visqol_nsim_model_all, 'omitnan'));
    fprintf('  Improvement: %.3f\n', mean(visqol_nsim_model_all - visqol_nsim_mvdr_all, 'omitnan'));
    
    %% ================= SAVE PER-FILE RESULTS TO CSV =================
    results_filename = fullfile(model_folder, 'batch_comparison_results.csv');
    
    % Create table with results
    results_table = table(processed_filenames, ...
        sisdr_mvdr_all, sisdr_model_all, sisdr_model_all - sisdr_mvdr_all, ...
        stoi_mvdr_all, stoi_model_all, stoi_model_all - stoi_mvdr_all, ...
        visqol_mos_mvdr_all, visqol_mos_model_all, visqol_mos_model_all - visqol_mos_mvdr_all, ...
        visqol_nsim_mvdr_all, visqol_nsim_model_all, visqol_nsim_model_all - visqol_nsim_mvdr_all, ...
        'VariableNames', {'Filename', ...
            'SISDR_MVDR', 'SISDR_Model', 'SISDR_Improvement', ...
            'STOI_MVDR', 'STOI_Model', 'STOI_Improvement', ...
            'ViSQOL_MOS_MVDR', 'ViSQOL_MOS_Model', 'ViSQOL_MOS_Improvement', ...
            'ViSQOL_NSIM_MVDR', 'ViSQOL_NSIM_Model', 'ViSQOL_NSIM_Improvement'});
    
    writetable(results_table, results_filename);
    fprintf('\nPer-file results saved to: %s\n', results_filename);
else
    fprintf('\nNo files were successfully processed.\n');
end

fprintf('\n===============================================\n');
fprintf('                 SUMMARY                       \n');
fprintf('===============================================\n');
if valid_count > 0
    fprintf('Average SI-SDR improvement: %.2f dB\n', mean(sisdr_model_all - sisdr_mvdr_all, 'omitnan'));
    fprintf('Average STOI improvement:   %.3f\n', mean(stoi_model_all - stoi_mvdr_all, 'omitnan'));
    fprintf('Average ViSQOL MOS improvement: %.2f\n', mean(visqol_mos_model_all - visqol_mos_mvdr_all, 'omitnan'));
    fprintf('Average ViSQOL NSIM improvement: %.3f\n', mean(visqol_nsim_model_all - visqol_nsim_mvdr_all, 'omitnan'));
end
