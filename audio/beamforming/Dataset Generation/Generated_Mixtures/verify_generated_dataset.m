%% ============================================================
%  verify_generated_dataset.m
%  Verify SIR, SNR, and quality metrics for generated V3 dataset
%
%  For each sample computes:
%    1. SIR / SNR  (from metadata — used during generation)
%    2. SI-SDR     of Mic(ch1), MVDR, MVDR+PostFilter  vs  clean
%    3. STOI       of Mic(ch1), MVDR, MVDR+PostFilter  vs  clean
%    4. ViSQOL     of Mic(ch1), MVDR, MVDR+PostFilter  vs  clean
%                  (optional, on a subset — slow)
%
%  Date: 9 Feb 2026
% =============================================================

clear; clc; close all;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');

%% ======================== CONFIG ========================
fs = 16000;

% Dataset root
dataset_root = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/prepared_dataset_v3';

% Folders
clean_dir   = fullfile(dataset_root, 'clean');
mix_dir     = fullfile(dataset_root, 'Mixtures');
mvdr_dir    = fullfile(dataset_root, 'MVDR_outputs');
pf_dir      = fullfile(dataset_root, 'MVDR_filtered');

% How many samples to evaluate (0 = all)
n_eval = 0;

% ViSQOL is slow (~1s per call × 3 methods × N samples)
% Set to 0 to skip, or a number for subset
n_visqol = 50;

%% ======================== LOAD METADATA ========================
meta_path = fullfile(dataset_root, 'metadata.csv');
if isfile(meta_path)
    meta = readtable(meta_path);
    has_meta = true;
    fprintf('Loaded metadata: %d entries\n', height(meta));
else
    has_meta = false;
    fprintf('No metadata.csv found — SIR/SNR distributions will be skipped.\n');
end

%% ======================== LIST FILES ========================
clean_files = dir(fullfile(clean_dir, '*.wav'));
n_files = length(clean_files);
fprintf('Found %d clean files\n\n', n_files);

if n_eval > 0
    n_files = min(n_files, n_eval);
    fprintf('Evaluating subset: %d samples\n', n_files);
end

%% ======================== PREALLOCATE ========================
sisdr_mic  = nan(n_files, 1);
sisdr_mvdr = nan(n_files, 1);
sisdr_pf   = nan(n_files, 1);

stoi_mic   = nan(n_files, 1);
stoi_mvdr  = nan(n_files, 1);
stoi_pf    = nan(n_files, 1);

if n_visqol > 0
    n_vq = min(n_files, n_visqol);
    visqol_mic  = nan(n_vq, 2);   % [MOS, NSIM]
    visqol_mvdr = nan(n_vq, 2);
    visqol_pf   = nan(n_vq, 2);
end

interf_types = cell(n_files, 1);

%% ======================== MAIN LOOP ========================
t_start = tic;
n_errors = 0;

fprintf('=== Evaluating %d samples ===\n\n', n_files);

for ii = 1:n_files
    if mod(ii, 100) == 0 || ii == 1
        elapsed = toc(t_start);
        rate = ii / elapsed;
        eta  = (n_files - ii) / rate;
        fprintf('[%d/%d] (%.1f%%)  %.1f samples/s  ETA: %.0f s\n', ...
            ii, n_files, 100*ii/n_files, rate, eta);
    end

    fname = clean_files(ii).name;

    try
        % ---- Load all 4 versions ----
        s     = audioread(fullfile(clean_dir, fname));       % mono clean
        x_mix = audioread(fullfile(mix_dir,   fname));       % stereo mixture
        y_mvdr_sig = audioread(fullfile(mvdr_dir, fname));   % mono MVDR
        y_pf_sig   = audioread(fullfile(pf_dir,   fname));   % mono MVDR+PF

        % Mic input = ch1 of stereo mixture
        x_mic = x_mix(:, 1);

        % Length-match (should already match, but safety)
        Lmin = min([length(s), length(x_mic), length(y_mvdr_sig), length(y_pf_sig)]);
        s          = s(1:Lmin);
        x_mic      = x_mic(1:Lmin);
        y_mvdr_sig = y_mvdr_sig(1:Lmin);
        y_pf_sig   = y_pf_sig(1:Lmin);

        % ---- Extract interference type from filename ----
        % Naming: maleBase_A_interfBase.wav  (type is after first _)
        parts = split(fname, '_');
        % Find the type token (A, B, or C) — it's a single character token
        itype = '';
        for pp = 1:length(parts)
            tok = parts{pp};
            if length(tok) == 1 && any(tok == 'ABC')
                itype = tok;
                break;
            end
        end
        interf_types{ii} = itype;

        % ---- SI-SDR ----
        sisdr_mic(ii)  = si_sdr(x_mic, s);
        sisdr_mvdr(ii) = si_sdr(y_mvdr_sig, s);
        sisdr_pf(ii)   = si_sdr(y_pf_sig, s);

        % ---- STOI ----
        stoi_mic(ii)   = stoi(s, x_mic, fs);
        stoi_mvdr(ii)  = stoi(s, y_mvdr_sig, fs);
        stoi_pf(ii)    = stoi(s, y_pf_sig, fs);

        % ---- ViSQOL (subset only) ----
        if n_visqol > 0 && ii <= n_vq
            [vq, ~, ~] = visqol(x_mic, s, fs, mode='speech', OutputMetric="MOS and NSIM");
            visqol_mic(ii, :) = vq(:)';
            [vq, ~, ~] = visqol(y_mvdr_sig, s, fs, mode='speech', OutputMetric="MOS and NSIM");
            visqol_mvdr(ii, :) = vq(:)';
            [vq, ~, ~] = visqol(y_pf_sig, s, fs, mode='speech', OutputMetric="MOS and NSIM");
            visqol_pf(ii, :) = vq(:)';
        end

    catch ME
        fprintf('  ERROR [%s]: %s\n', fname, ME.message);
        n_errors = n_errors + 1;
    end
end

elapsed_total = toc(t_start);
fprintf('\n=== Evaluation done in %.1f min ===\n', elapsed_total/60);
fprintf('Errors: %d / %d\n\n', n_errors, n_files);

%% ======================== METADATA DISTRIBUTIONS ========================
if has_meta
    fprintf('========================================\n');
    fprintf('  METADATA DISTRIBUTIONS (generation params)\n');
    fprintf('========================================\n');

    fprintf('\nSIR (dB):\n');
    fprintf('  Mean: %.2f   Std: %.2f   Min: %.2f   Max: %.2f\n', ...
        mean(meta.sir_db), std(meta.sir_db), min(meta.sir_db), max(meta.sir_db));

    fprintf('\nSNR (dB):\n');
    fprintf('  Mean: %.2f   Std: %.2f   Min: %.2f   Max: %.2f\n', ...
        mean(meta.snr_db), std(meta.snr_db), min(meta.snr_db), max(meta.snr_db));

    fprintf('\nθ_interf (deg):\n');
    fprintf('  Mean: %.1f   Std: %.1f   Min: %.1f   Max: %.1f\n', ...
        mean(meta.theta_interf), std(meta.theta_interf), ...
        min(meta.theta_interf), max(meta.theta_interf));

    fprintf('\nInterference types:\n');
    for t = {'A','B','C'}
        n = sum(strcmp(meta.interf_type, t{1}));
        fprintf('  %s: %d (%.1f%%)\n', t{1}, n, 100*n/height(meta));
    end
    fprintf('\n');
end

%% ======================== QUALITY METRICS ========================
% Remove NaN rows (from errors)
valid = ~isnan(sisdr_mic);

fprintf('========================================\n');
fprintf('  QUALITY METRICS  (%d valid samples)\n', sum(valid));
fprintf('========================================\n');

% ---------- SI-SDR ----------
fprintf('\n--- SI-SDR (dB) ---\n');
fprintf('%-12s  %8s  %8s  %8s  %8s\n', '', 'Mean', 'Std', 'Min', 'Max');
fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'Mic (ch1)', ...
    mean(sisdr_mic(valid)), std(sisdr_mic(valid)), min(sisdr_mic(valid)), max(sisdr_mic(valid)));
fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'MVDR', ...
    mean(sisdr_mvdr(valid)), std(sisdr_mvdr(valid)), min(sisdr_mvdr(valid)), max(sisdr_mvdr(valid)));
fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'MVDR+PF', ...
    mean(sisdr_pf(valid)), std(sisdr_pf(valid)), min(sisdr_pf(valid)), max(sisdr_pf(valid)));

fprintf('\n  MVDR gain:     %+.2f dB  (over mic)\n', mean(sisdr_mvdr(valid) - sisdr_mic(valid)));
fprintf('  PostFilt gain: %+.2f dB  (over mic)\n', mean(sisdr_pf(valid) - sisdr_mic(valid)));
fprintf('  PF over MVDR:  %+.2f dB\n', mean(sisdr_pf(valid) - sisdr_mvdr(valid)));

% ---------- Per-type SI-SDR ----------
fprintf('\n--- SI-SDR by interference type ---\n');
fprintf('%-6s  %8s  %8s  %8s  %8s\n', 'Type', 'Mic', 'MVDR', 'MVDR+PF', 'PF gain');
for t = {'A','B','C'}
    mask = valid & strcmp(interf_types, t{1});
    if any(mask)
        fprintf('%-6s  %8.2f  %8.2f  %8.2f  %+8.2f\n', t{1}, ...
            mean(sisdr_mic(mask)), mean(sisdr_mvdr(mask)), ...
            mean(sisdr_pf(mask)), mean(sisdr_pf(mask) - sisdr_mic(mask)));
    end
end

% ---------- STOI ----------
fprintf('\n--- STOI ---\n');
fprintf('%-12s  %8s  %8s  %8s  %8s\n', '', 'Mean', 'Std', 'Min', 'Max');
fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'Mic (ch1)', ...
    mean(stoi_mic(valid)), std(stoi_mic(valid)), min(stoi_mic(valid)), max(stoi_mic(valid)));
fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'MVDR', ...
    mean(stoi_mvdr(valid)), std(stoi_mvdr(valid)), min(stoi_mvdr(valid)), max(stoi_mvdr(valid)));
fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'MVDR+PF', ...
    mean(stoi_pf(valid)), std(stoi_pf(valid)), min(stoi_pf(valid)), max(stoi_pf(valid)));

% ---------- ViSQOL ----------
if n_visqol > 0
    valid_vq = ~isnan(visqol_mic(:,1));
    n_vq_valid = sum(valid_vq);

    fprintf('\n--- ViSQOL (%d samples) ---\n', n_vq_valid);
    fprintf('%-12s  %8s  %8s\n', '', 'MOS', 'NSIM');
    fprintf('%-12s  %8.2f  %8.3f\n', 'Mic (ch1)', ...
        mean(visqol_mic(valid_vq,1)), mean(visqol_mic(valid_vq,2)));
    fprintf('%-12s  %8.2f  %8.3f\n', 'MVDR', ...
        mean(visqol_mvdr(valid_vq,1)), mean(visqol_mvdr(valid_vq,2)));
    fprintf('%-12s  %8.2f  %8.3f\n', 'MVDR+PF', ...
        mean(visqol_pf(valid_vq,1)), mean(visqol_pf(valid_vq,2)));
end

%% ======================== HISTOGRAMS ========================
figure('Name', 'Dataset Verification', 'Position', [100 100 1400 900]);

% -- SI-SDR histograms --
subplot(2,3,1);
histogram(sisdr_mic(valid), 50, 'FaceAlpha', 0.5); hold on;
histogram(sisdr_mvdr(valid), 50, 'FaceAlpha', 0.5);
histogram(sisdr_pf(valid), 50, 'FaceAlpha', 0.5);
legend('Mic','MVDR','MVDR+PF'); xlabel('SI-SDR (dB)'); ylabel('Count');
title('SI-SDR Distribution'); grid on;

% -- STOI histograms --
subplot(2,3,2);
histogram(stoi_mic(valid), 50, 'FaceAlpha', 0.5); hold on;
histogram(stoi_mvdr(valid), 50, 'FaceAlpha', 0.5);
histogram(stoi_pf(valid), 50, 'FaceAlpha', 0.5);
legend('Mic','MVDR','MVDR+PF'); xlabel('STOI'); ylabel('Count');
title('STOI Distribution'); grid on;

% -- SI-SDR improvement --
subplot(2,3,3);
histogram(sisdr_pf(valid) - sisdr_mic(valid), 50);
xlabel('SI-SDR gain (dB)'); ylabel('Count');
title('MVDR+PF Gain over Mic'); grid on;
xline(mean(sisdr_pf(valid) - sisdr_mic(valid)), 'r--', 'LineWidth', 2);

if has_meta
    % Evaluate only for evaluated files
    eval_meta_idx = min(height(meta), n_files);

    % -- SIR distribution --
    subplot(2,3,4);
    histogram(meta.sir_db(1:eval_meta_idx), 50);
    xlabel('SIR (dB)'); ylabel('Count');
    title('SIR Distribution'); grid on;
    xline(0, 'r--', 'Competition', 'LineWidth', 2);

    % -- SNR distribution --
    subplot(2,3,5);
    histogram(meta.snr_db(1:eval_meta_idx), 50);
    xlabel('SNR (dB)'); ylabel('Count');
    title('SNR Distribution'); grid on;
    xline(5, 'r--', 'Competition', 'LineWidth', 2);

    % -- θ_interf distribution --
    subplot(2,3,6);
    histogram(meta.theta_interf(1:eval_meta_idx), 50);
    xlabel('\theta_{interf} (deg)'); ylabel('Count');
    title('\theta_{interf} Distribution'); grid on;
    xline(40, 'r--', 'Competition', 'LineWidth', 2);
end

sgtitle('Dataset V3 Verification', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================== SAVE RESULTS ========================
results = struct();
results.n_samples = sum(valid);
results.sisdr = struct('mic', sisdr_mic(valid), 'mvdr', sisdr_mvdr(valid), 'pf', sisdr_pf(valid));
results.stoi  = struct('mic', stoi_mic(valid),  'mvdr', stoi_mvdr(valid),  'pf', stoi_pf(valid));
results.interf_types = interf_types(valid);

save(fullfile(dataset_root, 'verification_results.mat'), 'results');
fprintf('\nResults saved to verification_results.mat\n');

fprintf('\n========================================\n');
fprintf('  SUMMARY\n');
fprintf('========================================\n');
fprintf('Samples evaluated: %d\n', sum(valid));
fprintf('Mean SI-SDR:  Mic=%.2f  MVDR=%.2f  MVDR+PF=%.2f dB\n', ...
    mean(sisdr_mic(valid)), mean(sisdr_mvdr(valid)), mean(sisdr_pf(valid)));
fprintf('Mean STOI:    Mic=%.3f  MVDR=%.3f  MVDR+PF=%.3f\n', ...
    mean(stoi_mic(valid)), mean(stoi_mvdr(valid)), mean(stoi_pf(valid)));
if n_visqol > 0 && any(valid_vq)
    fprintf('Mean ViSQOL:  Mic=%.2f  MVDR=%.2f  MVDR+PF=%.2f (MOS)\n', ...
        mean(visqol_mic(valid_vq,1)), mean(visqol_mvdr(valid_vq,1)), mean(visqol_pf(valid_vq,1)));
end
fprintf('\n');
