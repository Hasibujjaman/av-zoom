% ============================================================================
%  compare_beamformers.m
%  SP Cup 2026 — Phase 2
%
%  Compare our MVDR + Zelinski PF against three MATLAB Phased Array Toolbox
%  beamformers on the same dataset:
%    1. Time-Delay Beamformer        (phased.TimeDelayBeamformer)
%    2. Frost Beamformer             (phased.FrostBeamformer)
%    3. Subband MVDR Beamformer      (phased.SubbandMVDRBeamformer)
%
%  Metrics: SI-SDR (dB), STOI, ViSQOL, Latency (ms/file)
%
%  USAGE:
%    1. Set `clean_dir` and `mix_dir` below.
%       - clean_dir : folder of mono clean .wav files  (16 kHz)
%       - mix_dir   : folder of 2-ch mixture .wav files (16 kHz)
%       File stems must match (e.g., 001.wav in both dirs).
%    2. Run the script.  Results are printed and saved to .mat/.csv.
%
%  REQUIREMENTS:
%    - Phased Array System Toolbox  (for MATLAB beamformers)
%    - Audio Toolbox               (for ViSQOL via `visqol`)
%    - Signal Processing Toolbox   (for STOI via `stoi`)
%    - Our custom functions on the path:
%        compute_steering_vector, compute_mvdr_weights,
%        mvdr_postfilter, stft_multichannel, istft_single_channel, si_sdr
% ============================================================================
clear; clc; close all;

%% ============================  USER CONFIG  ================================
clean_dir = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_NonRerverb/mvdr_test/clean_small';   % <-- SET: folder with mono clean .wav references
mix_dir   = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Test_NonRerverb/mvdr_test/mixture_small';   % <-- SET: folder with 2-ch mixture .wav files

N_files   = inf;  % number of files to evaluate (inf = all)
fs        = 16000;

% Array geometry (must match dataset generation)
d_mic     = 0.08;           % inter-mic spacing (m)
c         = 340;            % speed of sound (m/s)
theta_target_deg = 90;      % our convention: 90° = broadside

% STFT params (shared with our MVDR)
nfft      = 512;
hop       = 128;
win       = sqrt(hann(nfft, 'periodic'));

% MVDR params
delta     = 1e-3;           % diagonal loading

% Frost params
frost_filter_len = 10;      % FIR filter taps per channel

% Subband MVDR params
n_subbands = 64;            % number of subbands

% Output
out_dir   = fullfile(fileparts(mfilename('fullpath')), 'comparison_results');
% ============================================================================

%% ========================  VALIDATE INPUTS  ================================
assert(~isempty(clean_dir) && isfolder(clean_dir), ...
    'Set clean_dir to a valid folder of clean .wav files.');
assert(~isempty(mix_dir) && isfolder(mix_dir), ...
    'Set mix_dir to a valid folder of 2-ch mixture .wav files.');

% Add beamforming functions to path
script_dir = fileparts(mfilename('fullpath'));
bf_root = fullfile(script_dir, '..', '..');   % audio/beamforming/
bf_root = char(java.io.File(bf_root).getCanonicalPath());  % resolve '..' safely
addpath(bf_root);
metrics_dir = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics';
if isfolder(metrics_dir)
    addpath(metrics_dir);
else
    warning('Metrics folder not found at: %s', metrics_dir);
end

% Check toolbox availability
has_phased = ~isempty(ver('phased'));
has_audio  = ~isempty(ver('audio'));
assert(has_phased, 'Phased Array System Toolbox is required.');

has_visqol_fn = exist('visqol', 'file') == 2 || exist('visqol', 'file') == 5;
has_stoi_fn   = exist('stoi', 'file') == 2   || exist('stoi', 'file') == 5;
if ~has_stoi_fn
    warning('stoi() not found — STOI will be skipped.');
end
if ~has_visqol_fn
    warning('visqol() not found — ViSQOL will be skipped.');
end

%% ========================  DISCOVER FILES  =================================
clean_files = dir(fullfile(clean_dir, '*.wav'));
mix_files   = dir(fullfile(mix_dir,   '*.wav'));

% Match by stem
clean_stems = cellfun(@(f) extractBefore(f, '.wav'), ...
    {clean_files.name}, 'Uni', false);
mix_stems   = cellfun(@(f) extractBefore(f, '.wav'), ...
    {mix_files.name}, 'Uni', false);
[common_stems, ic, im] = intersect(clean_stems, mix_stems);

assert(~isempty(common_stems), 'No matching file stems between clean and mix dirs.');

N = min(length(common_stems), N_files);
fprintf('Found %d matching files.  Evaluating %d.\n', length(common_stems), N);

%% ========================  BUILD MATLAB ARRAY OBJECT  ======================
mic_element = phased.OmnidirectionalMicrophoneElement( ...
    'FrequencyRange', [20, fs/2]);
ula = phased.ULA( ...
    'Element',        mic_element, ...
    'NumElements',    2, ...
    'ElementSpacing', d_mic);

% Phased toolbox convention: 0° azimuth = broadside
target_az_phased  = 0;   % broadside
target_el_phased  = 0;
target_dir_phased = [target_az_phased; target_el_phased];

%% ========================  CREATE BEAMFORMER OBJECTS  =======================

% 1. Time-Delay Beamformer
bf_td = phased.TimeDelayBeamformer( ...
    'SensorArray',      ula, ...
    'SampleRate',        fs, ...
    'PropagationSpeed',  c, ...
    'Direction',         target_dir_phased);

% 2. Frost Beamformer
bf_frost = phased.FrostBeamformer( ...
    'SensorArray',       ula, ...
    'SampleRate',        fs, ...
    'PropagationSpeed',  c, ...
    'Direction',         target_dir_phased, ...
    'FilterLength',      frost_filter_len);

% 3. Subband MVDR Beamformer
bf_sub = phased.SubbandMVDRBeamformer( ...
    'SensorArray',          ula, ...
    'SampleRate',           fs, ...
    'PropagationSpeed',     c, ...
    'OperatingFrequency',   fs, ...
    'Direction',            target_dir_phased, ...
    'NumSubbands',          n_subbands, ...
    'DiagonalLoadingFactor', delta * nfft);

%% ========================  PREALLOCATE RESULTS  ============================
method_names = {'Mixture (ref-ch)', 'Our MVDR+PF', 'Our MVDR (no PF)', ...
                'TimeDelay BF', 'Frost BF', 'Subband MVDR'};
n_methods = length(method_names);

si_sdr_all    = nan(N, n_methods);
stoi_all      = nan(N, n_methods);
visqol_all    = nan(N, n_methods);
latency_ms    = nan(N, n_methods);   % per-file wall-clock ms

%% ========================  MAIN LOOP  ======================================
fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('  BEAMFORMER COMPARISON  (%d files)\n', N);
fprintf('%s\n\n', repmat('=', 1, 72));

for i = 1:N
    % ---- Load ----
    clean_path = fullfile(clean_dir,  clean_files(ic(i)).name);
    mix_path   = fullfile(mix_dir,    mix_files(im(i)).name);

    [s_clean, fs_c] = audioread(clean_path);
    [x_mix,   fs_m] = audioread(mix_path);

    assert(fs_c == fs && fs_m == fs, 'Sample rate mismatch for file %d.', i);
    assert(size(x_mix, 2) == 2, 'Mixture must be 2-channel for file %d.', i);
    if size(s_clean, 2) > 1, s_clean = s_clean(:,1); end

    % Ensure same length
    L = min(length(s_clean), size(x_mix, 1));
    s_clean = s_clean(1:L);
    x_mix   = x_mix(1:L, :);

    % Reference channel (mic 1)
    x_ref = x_mix(:, 1);

    % ---- Progress ----
    if mod(i, 50) == 1 || i == N
        fprintf('  [%4d / %4d]  %s\n', i, N, clean_files(ic(i)).name);
    end

    % ==================================================================
    % Method 0: Mixture reference channel (baseline)
    % ==================================================================
    si_sdr_all(i, 1) = si_sdr(x_ref, s_clean);
    if has_stoi_fn,   stoi_all(i, 1)   = stoi(s_clean, x_ref, fs); end
    if has_visqol_fn, visqol_all(i, 1) = visqol(s_clean, x_ref, fs); end

    % ==================================================================
    % Method 1: Our MVDR + Zelinski Post-Filter
    % ==================================================================
    t0 = tic;
    X_stft = stft_multichannel(x_mix, win, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X_stft);
    freqs = (0:numFreqs-1)' * fs / nfft;
    micPos = [0; d_mic];

    dvec = compute_steering_vector(theta_target_deg, freqs, micPos, c);

    % Batch covariance
    Rxx = zeros(numMics, numMics, numFreqs);
    for n = 1:numFrames
        for k = 1:numFreqs
            xk = squeeze(X_stft(k, n, :));
            Rxx(:,:,k) = Rxx(:,:,k) + xk * xk';
        end
    end
    Rxx = Rxx / numFrames;

    W = compute_mvdr_weights(Rxx, dvec, delta);
    Y_pf = mvdr_postfilter(X_stft, W, dvec);
    y_ours = istft_single_channel(Y_pf, win, hop, nfft, L);
    latency_ms(i, 2) = toc(t0) * 1000;

    y_ours = y_ours(1:L);
    si_sdr_all(i, 2) = si_sdr(y_ours, s_clean);
    if has_stoi_fn,   stoi_all(i, 2)   = stoi(s_clean, y_ours, fs); end
    if has_visqol_fn, visqol_all(i, 2) = visqol(s_clean, y_ours, fs); end

    % ==================================================================
    % Method 2: Our MVDR (no Post-Filter)
    % ==================================================================
    t0 = tic;
    Y_mvdr_only = apply_mvdr(X_stft, W);
    y_mvdr_only = istft_single_channel(Y_mvdr_only, win, hop, nfft, L);
    latency_ms(i, 3) = toc(t0) * 1000;

    y_mvdr_only = y_mvdr_only(1:L);
    si_sdr_all(i, 3) = si_sdr(y_mvdr_only, s_clean);
    if has_stoi_fn,   stoi_all(i, 3)   = stoi(s_clean, y_mvdr_only, fs); end
    if has_visqol_fn, visqol_all(i, 3) = visqol(s_clean, y_mvdr_only, fs); end

    % ==================================================================
    % Method 3: MATLAB Time-Delay Beamformer
    % ==================================================================
    release(bf_td);
    t0 = tic;
    y_td = bf_td(x_mix);
    latency_ms(i, 4) = toc(t0) * 1000;

    y_td = y_td(1:L);
    si_sdr_all(i, 4) = si_sdr(y_td, s_clean);
    if has_stoi_fn,   stoi_all(i, 4)   = stoi(s_clean, y_td, fs); end
    if has_visqol_fn, visqol_all(i, 4) = visqol(s_clean, y_td, fs); end

    % ==================================================================
    % Method 4: MATLAB Frost Beamformer
    % ==================================================================
    release(bf_frost);
    t0 = tic;
    y_frost = bf_frost(x_mix);
    latency_ms(i, 5) = toc(t0) * 1000;

    y_frost = y_frost(1:L);
    si_sdr_all(i, 5) = si_sdr(y_frost, s_clean);
    if has_stoi_fn,   stoi_all(i, 5)   = stoi(s_clean, y_frost, fs); end
    if has_visqol_fn, visqol_all(i, 5) = visqol(s_clean, y_frost, fs); end

    % ==================================================================
    % Method 5: MATLAB Subband MVDR Beamformer
    % ==================================================================
    release(bf_sub);
    t0 = tic;
    y_sub = bf_sub(x_mix);
    latency_ms(i, 6) = toc(t0) * 1000;

    y_sub = y_sub(1:L);
    si_sdr_all(i, 6) = si_sdr(y_sub, s_clean);
    if has_stoi_fn,   stoi_all(i, 6)   = stoi(s_clean, y_sub, fs); end
    if has_visqol_fn, visqol_all(i, 6) = visqol(s_clean, y_sub, fs); end
end

%% ========================  AGGREGATE RESULTS  ==============================
mean_sisdr   = nanmean(si_sdr_all, 1);
std_sisdr    = nanstd(si_sdr_all, 0, 1);
med_sisdr    = nanmedian(si_sdr_all, 1);

mean_stoi_v  = nanmean(stoi_all, 1);
std_stoi_v   = nanstd(stoi_all, 0, 1);

mean_visqol  = nanmean(visqol_all, 1);
std_visqol   = nanstd(visqol_all, 0, 1);

mean_lat     = nanmean(latency_ms, 1);
std_lat      = nanstd(latency_ms, 0, 1);

% SI-SDR improvement over mixture baseline
sisdr_imp = mean_sisdr - mean_sisdr(1);

%% ========================  PRINT TABLE  ====================================
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('  COMPARISON RESULTS  (%d files)\n', N);
fprintf('%s\n\n', repmat('=', 1, 90));

% Header
fprintf('  %-22s | %10s | %10s | %10s | %10s | %12s\n', ...
    'Method', 'SI-SDR(dB)', 'Δ SI-SDR', 'STOI', 'ViSQOL', 'Latency(ms)');
fprintf('  %s\n', repmat('-', 1, 86));

for m = 1:n_methods
    sisdr_str = sprintf('%6.2f±%.2f', mean_sisdr(m), std_sisdr(m));
    imp_str   = sprintf('%+6.2f', sisdr_imp(m));

    if has_stoi_fn
        stoi_str = sprintf('%6.4f±%.3f', mean_stoi_v(m), std_stoi_v(m));
    else
        stoi_str = '    N/A   ';
    end

    if has_visqol_fn
        vq_str = sprintf('%6.3f±%.2f', mean_visqol(m), std_visqol(m));
    else
        vq_str = '    N/A   ';
    end

    if m == 1
        lat_str = '     ---     ';
    else
        lat_str = sprintf('%7.2f±%.1f', mean_lat(m), std_lat(m));
    end

    fprintf('  %-22s | %10s | %10s | %10s | %10s | %12s\n', ...
        method_names{m}, sisdr_str, imp_str, stoi_str, vq_str, lat_str);
end
fprintf('  %s\n', repmat('-', 1, 86));

%% ========================  RANK TABLE  =====================================
fprintf('\n  RANKINGS (by mean SI-SDR, descending):\n');
[~, rank_idx] = sort(mean_sisdr(2:end), 'descend');
rank_idx = rank_idx + 1;  % offset past mixture baseline
for r = 1:length(rank_idx)
    m = rank_idx(r);
    fprintf('    #%d  %-22s   SI-SDR = %+.2f dB  (Δ = %+.2f dB)\n', ...
        r, method_names{m}, mean_sisdr(m), sisdr_imp(m));
end

%% ========================  BOX PLOTS  ======================================
figure('Position', [100, 100, 1100, 700], 'Color', 'w');

% SI-SDR
subplot(2,2,1);
boxplot(si_sdr_all(:, 2:end), method_names(2:end));
ylabel('SI-SDR (dB)');
title('SI-SDR Comparison');
grid on;
xtickangle(20);

% SI-SDR improvement over mixture
subplot(2,2,2);
sisdr_imp_per_file = si_sdr_all(:, 2:end) - si_sdr_all(:, 1);
boxplot(sisdr_imp_per_file, method_names(2:end));
ylabel('\Delta SI-SDR (dB)');
title('SI-SDR Improvement over Mixture');
grid on;
xtickangle(20);

% STOI
subplot(2,2,3);
if has_stoi_fn
    boxplot(stoi_all(:, 2:end), method_names(2:end));
    ylabel('STOI');
    title('STOI Comparison');
    grid on;
    xtickangle(20);
else
    text(0.5, 0.5, 'STOI not available', 'HorizontalAlignment', 'center');
    axis off;
end

% Latency
subplot(2,2,4);
boxplot(latency_ms(:, 2:end), method_names(2:end));
ylabel('Latency (ms/file)');
title('Processing Latency');
grid on;
xtickangle(20);

sgtitle(sprintf('Beamformer Comparison — %d files', N), 'FontWeight', 'bold');

%% ========================  SAVE RESULTS  ===================================
os.makedirs = @(d) mkdir(d);
if ~isfolder(out_dir), mkdir(out_dir); end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% .mat
results = struct();
results.method_names  = method_names;
results.si_sdr_all    = si_sdr_all;
results.stoi_all      = stoi_all;
results.visqol_all    = visqol_all;
results.latency_ms    = latency_ms;
results.mean_sisdr    = mean_sisdr;
results.mean_stoi     = mean_stoi_v;
results.mean_visqol   = mean_visqol;
results.mean_latency  = mean_lat;
results.sisdr_imp     = sisdr_imp;
results.clean_dir     = clean_dir;
results.mix_dir       = mix_dir;
results.N_files       = N;
results.config.nfft   = nfft;
results.config.hop    = hop;
results.config.delta  = delta;
results.config.d_mic  = d_mic;
results.config.frost_filter_len = frost_filter_len;
results.config.n_subbands       = n_subbands;
results.timestamp     = timestamp;

mat_path = fullfile(out_dir, sprintf('beamformer_comparison_%s.mat', timestamp));
save(mat_path, 'results');

% .csv summary
csv_path = fullfile(out_dir, sprintf('beamformer_comparison_%s.csv', timestamp));
fid = fopen(csv_path, 'w');
fprintf(fid, 'Method,Mean_SI-SDR_dB,Std_SI-SDR,Delta_SI-SDR_dB');
if has_stoi_fn,   fprintf(fid, ',Mean_STOI,Std_STOI'); end
if has_visqol_fn, fprintf(fid, ',Mean_ViSQOL,Std_ViSQOL'); end
fprintf(fid, ',Mean_Latency_ms,Std_Latency_ms\n');

for m = 1:n_methods
    fprintf(fid, '%s,%.4f,%.4f,%.4f', ...
        method_names{m}, mean_sisdr(m), std_sisdr(m), sisdr_imp(m));
    if has_stoi_fn,   fprintf(fid, ',%.6f,%.6f', mean_stoi_v(m), std_stoi_v(m)); end
    if has_visqol_fn, fprintf(fid, ',%.4f,%.4f', mean_visqol(m), std_visqol(m)); end
    if m == 1
        fprintf(fid, ',NaN,NaN');
    else
        fprintf(fid, ',%.4f,%.4f', mean_lat(m), std_lat(m));
    end
    fprintf(fid, '\n');
end
fclose(fid);

% Save figure
fig_path = fullfile(out_dir, sprintf('beamformer_comparison_%s.png', timestamp));
saveas(gcf, fig_path);

fprintf('\nResults saved:\n');
fprintf('  MAT : %s\n', mat_path);
fprintf('  CSV : %s\n', csv_path);
fprintf('  FIG : %s\n', fig_path);
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('  Done.\n');
fprintf('%s\n', repmat('=', 1, 90));
