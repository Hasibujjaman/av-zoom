%% ============================================================
%  verify_reverb_dataset.m
%  Verify Task 2 reverberant dataset (prepared_dataset_reverb_v3)
%
%  Supports the dual-save layout:
%    <root>/compensated/   — delay-aligned (DNN training)
%    <root>/uncompensated/ — raw timing (competition / testing)
%
%  For each version present, runs:
%    1. File integrity — all 4 folders have matching filenames & lengths
%    2. Quality metrics (SI-SDR, OSINR, STOI, ViSQOL, PESQ)
%    3. Per-type breakdown (A/B/C)
%    4. Sanity checks & histograms
%
%  Shared checks (run once):
%    - RIR data (RT60, DRR, propagation delay)
%    - Metadata distributions (SIR, SNR, θ_interf, type shares)
%
%  Date: 11 Feb 2026
% =============================================================

clear; clc; close all;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');

%% ======================== CONFIG ========================
fs = 16000;

% Dataset root
dataset_root = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/prepared_dataset_reverb_v3';

% How many samples to evaluate per version (0 = all)
n_eval = 0;

% ViSQOL is slow (~1s per call × 3 methods × N).  0 = skip.
n_visqol = 50;

% PESQ.  0 = skip.
n_pesq = 50;

fprintf('============================================\n');
fprintf('  TASK 2 REVERBERANT DATASET VERIFICATION\n');
fprintf('============================================\n\n');

%% ======================== DETECT VERSIONS ========================
version_names = {};
if isfolder(fullfile(dataset_root, 'compensated'))
    version_names{end+1} = 'compensated';
end
if isfolder(fullfile(dataset_root, 'uncompensated'))
    version_names{end+1} = 'uncompensated';
end

if isempty(version_names)
    error('No compensated/ or uncompensated/ subfolder found under:\n  %s', dataset_root);
end

fprintf('Detected versions: %s\n\n', strjoin(version_names, ', '));

%% ======================== RIR DATA CHECK (shared) ========================
fprintf('=== RIR data verification ===\n');

rir_path = fullfile(dataset_root, 'rir_bank.mat');
has_rir = false;
if isfile(rir_path)
    rir_data = load(rir_path);
    has_rir  = true;
    fprintf('  Loaded rir_bank.mat\n');
    fprintf('  Room:           [%.1f × %.1f × %.1f] m\n', rir_data.room_dim);
    fprintf('  RT60 target:    %.2f s\n', rir_data.RT60_target);
    fprintf('  RT60 achieved:  %.3f s\n', rir_data.rt60_mid);
    fprintf('  Absorption α:   %.4f\n', rir_data.absorption_final);
    fprintf('  ISM order:      %d\n', rir_data.ism_order);
    fprintf('  Prop delay:     %d samples (%.2f ms)\n', ...
        rir_data.prop_delay_smp, rir_data.prop_delay_smp/fs*1000);

    if isfield(rir_data, 'az_interf')
        fprintf('  Interf azimuth: %.1f° (fixed)\n', rir_data.az_interf);
    elseif isfield(rir_data, 'az_bank')
        fprintf('  Interf RIR bank: %d azimuths (%.1f°–%.1f°)\n', ...
            length(rir_data.az_bank), rir_data.az_bank(1), rir_data.az_bank(end));
    end

    % RT60 from stored target RIR
    rt60_tgt1 = estimate_rt60_local(rir_data.rir_target(1,:), fs);
    rt60_tgt2 = estimate_rt60_local(rir_data.rir_target(2,:), fs);
    fprintf('\n  RT60 from stored target RIR:\n');
    fprintf('    Mic1: %.3f s   Mic2: %.3f s\n', rt60_tgt1, rt60_tgt2);

    % RT60 from interference RIR
    if isfield(rir_data, 'rir_interf')
        rt60_i = estimate_rt60_local(rir_data.rir_interf(1,:), fs);
        fprintf('  RT60 from stored interf RIR (mic1): %.3f s\n', rt60_i);
    elseif isfield(rir_data, 'rir_interf_bank')
        n_check = min(5, length(rir_data.rir_interf_bank));
        check_idx = round(linspace(1, length(rir_data.rir_interf_bank), n_check));
        fprintf('  RT60 spot-check (interf RIRs):\n');
        for ci = check_idx
            rt60_i = estimate_rt60_local(rir_data.rir_interf_bank{ci}(1,:), fs);
            fprintf('    az = %.1f° → RT60 = %.3f s\n', rir_data.az_bank(ci), rt60_i);
        end
    end

    % DRR from target RIR
    ir1 = rir_data.rir_target(1,:);
    [~, pk] = max(abs(ir1));
    direct_win = max(1, pk-round(0.001*fs)) : min(length(ir1), pk+round(0.001*fs));
    E_direct = sum(ir1(direct_win).^2);
    E_rest   = sum(ir1.^2) - E_direct;
    DRR = 10*log10(E_direct / (E_rest + eps));
    fprintf('\n  DRR (direct peak ± 1ms): %+.1f dB\n', DRR);
else
    fprintf('  rir_bank.mat not found — skipping RIR checks\n');
end
fprintf('\n');

%% ======================== LOAD METADATA (shared) ========================
meta_path = fullfile(dataset_root, 'metadata.csv');
has_meta = false;
if isfile(meta_path)
    meta = readtable(meta_path);
    has_meta = true;
    fprintf('=== Metadata: %d entries ===\n\n', height(meta));
else
    fprintf('No metadata.csv found — SIR/SNR distributions will be skipped.\n\n');
end

%% ======================== METADATA DISTRIBUTIONS (shared) ========================
if has_meta
    fprintf('========================================\n');
    fprintf('  METADATA DISTRIBUTIONS\n');
    fprintf('========================================\n');

    fprintf('\nSIR (dB):   target = 0 dB\n');
    fprintf('  Mean: %.2f   Std: %.2f   Min: %.2f   Max: %.2f\n', ...
        mean(meta.sir_db), std(meta.sir_db), min(meta.sir_db), max(meta.sir_db));

    fprintf('\nSNR (dB):   target = 5 dB\n');
    fprintf('  Mean: %.2f   Std: %.2f   Min: %.2f   Max: %.2f\n', ...
        mean(meta.snr_db), std(meta.snr_db), min(meta.snr_db), max(meta.snr_db));

    fprintf('\nθ_interf (deg):   target = 40° (fixed)\n');
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

%% ================================================================
%  PER-VERSION EVALUATION LOOP
%  ================================================================
all_results = struct();

for vi = 1:length(version_names)
    ver = version_names{vi};

    fprintf('\n############################################################\n');
    fprintf('  EVALUATING VERSION: %s\n', upper(ver));
    fprintf('############################################################\n\n');

    clean_dir = fullfile(dataset_root, ver, 'clean');
    mix_dir   = fullfile(dataset_root, ver, 'Mixtures');
    mvdr_dir  = fullfile(dataset_root, ver, 'MVDR_outputs');
    pf_dir    = fullfile(dataset_root, ver, 'MVDR_filtered');

    %% ---- File integrity ----
    fprintf('=== File integrity [%s] ===\n', ver);
    clean_files = dir(fullfile(clean_dir, '*.wav'));
    mix_files   = dir(fullfile(mix_dir,   '*.wav'));
    mvdr_files  = dir(fullfile(mvdr_dir,  '*.wav'));
    pf_files    = dir(fullfile(pf_dir,    '*.wav'));

    n_clean = length(clean_files);
    n_mix   = length(mix_files);
    n_mvdr  = length(mvdr_files);
    n_pf    = length(pf_files);

    fprintf('  clean/:          %d files\n', n_clean);
    fprintf('  Mixtures/:       %d files\n', n_mix);
    fprintf('  MVDR_outputs/:   %d files\n', n_mvdr);
    fprintf('  MVDR_filtered/:  %d files\n', n_pf);

    if n_clean == n_mix && n_mix == n_mvdr && n_mvdr == n_pf
        fprintf('  ✓ All folders have matching file counts\n\n');
    else
        warning('File count mismatch across folders in %s!', ver);
    end

    % Check that filenames match across all folders
    clean_names = sort({clean_files.name}');
    mix_names   = sort({mix_files.name}');
    mvdr_names  = sort({mvdr_files.name}');
    pf_names    = sort({pf_files.name}');

    name_match = isequal(clean_names, mix_names) && ...
                 isequal(clean_names, mvdr_names) && ...
                 isequal(clean_names, pf_names);
    if name_match
        fprintf('  ✓ Filenames match across all 4 folders\n\n');
    else
        warning('Filename mismatch across folders in %s!', ver);
        missing_mix  = setdiff(clean_names, mix_names);
        missing_mvdr = setdiff(clean_names, mvdr_names);
        missing_pf   = setdiff(clean_names, pf_names);
        if ~isempty(missing_mix),  fprintf('  Missing from Mixtures: %d\n', length(missing_mix)); end
        if ~isempty(missing_mvdr), fprintf('  Missing from MVDR_outputs: %d\n', length(missing_mvdr)); end
        if ~isempty(missing_pf),   fprintf('  Missing from MVDR_filtered: %d\n', length(missing_pf)); end
        fprintf('\n');
    end

    % Spot-check audio lengths
    fprintf('  Spot-checking audio lengths (first 5 files) ...\n');
    for jj = 1:min(5, n_clean)
        fname = clean_files(jj).name;
        info_c = audioinfo(fullfile(clean_dir, fname));
        info_m = audioinfo(fullfile(mix_dir,   fname));
        info_v = audioinfo(fullfile(mvdr_dir,  fname));
        info_p = audioinfo(fullfile(pf_dir,    fname));
        lens = [info_c.TotalSamples, info_m.TotalSamples, info_v.TotalSamples, info_p.TotalSamples];
        all_match = all(lens == lens(1));
        fprintf('    %s: clean=%d  mix=%d  mvdr=%d  pf=%d  %s\n', ...
            fname, lens(1), lens(2), lens(3), lens(4), ...
            ternary(all_match, '✓', '✗ LENGTH MISMATCH'));
    end
    fprintf('\n');

    %% ---- Setup evaluation ----
    n_files = n_clean;
    if n_eval > 0
        n_files = min(n_files, n_eval);
        fprintf('Evaluating subset: %d samples\n\n', n_files);
    end

    % Preallocate
    sisdr_mic  = nan(n_files, 1);
    sisdr_mvdr = nan(n_files, 1);
    sisdr_pf   = nan(n_files, 1);

    osinr_mic  = nan(n_files, 1);
    osinr_mvdr = nan(n_files, 1);
    osinr_pf   = nan(n_files, 1);

    stoi_mic   = nan(n_files, 1);
    stoi_mvdr  = nan(n_files, 1);
    stoi_pf    = nan(n_files, 1);

    if n_visqol > 0
        n_vq = min(n_files, n_visqol);
        visqol_mic  = nan(n_vq, 1);
        visqol_mvdr = nan(n_vq, 1);
        visqol_pf   = nan(n_vq, 1);
    end

    if n_pesq > 0
        n_pq = min(n_files, n_pesq);
        pesq_mic  = nan(n_pq, 1);
        pesq_mvdr = nan(n_pq, 1);
        pesq_pf   = nan(n_pq, 1);
    end

    interf_types = cell(n_files, 1);

    %% ---- Main evaluation loop ----
    t_start = tic;
    n_errors = 0;
    fprintf('=== Evaluating %d samples [%s] ===\n\n', n_files, ver);

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
            % Load all 4 versions
            s          = audioread(fullfile(clean_dir, fname));
            x_mix      = audioread(fullfile(mix_dir,   fname));
            y_mvdr_sig = audioread(fullfile(mvdr_dir,  fname));
            y_pf_sig   = audioread(fullfile(pf_dir,    fname));

            x_mic = x_mix(:, 1);

            % Length-match
            Lmin = min([length(s), length(x_mic), length(y_mvdr_sig), length(y_pf_sig)]);
            s          = s(1:Lmin);
            x_mic      = x_mic(1:Lmin);
            y_mvdr_sig = y_mvdr_sig(1:Lmin);
            y_pf_sig   = y_pf_sig(1:Lmin);

            % Interference type from filename
            parts = split(fname, '_');
            itype = '';
            for pp = 1:length(parts)
                tok = parts{pp};
                if length(tok) == 1 && any(tok == 'ABC')
                    itype = tok;
                    break;
                end
            end
            interf_types{ii} = itype;

            % SI-SDR
            sisdr_mic(ii)  = si_sdr(x_mic, s);
            sisdr_mvdr(ii) = si_sdr(y_mvdr_sig, s);
            sisdr_pf(ii)   = si_sdr(y_pf_sig, s);

            % OSINR (approx vs dry clean)
            resid_mic  = x_mic - s;
            resid_mvdr = y_mvdr_sig - s;
            resid_pf   = y_pf_sig - s;
            osinr_mic(ii)  = 10*log10(mean(s.^2) / (mean(resid_mic.^2)  + 1e-9));
            osinr_mvdr(ii) = 10*log10(mean(s.^2) / (mean(resid_mvdr.^2) + 1e-9));
            osinr_pf(ii)   = 10*log10(mean(s.^2) / (mean(resid_pf.^2)   + 1e-9));

            % STOI
            stoi_mic(ii)   = stoi(s, x_mic, fs);
            stoi_mvdr(ii)  = stoi(s, y_mvdr_sig, fs);
            stoi_pf(ii)    = stoi(s, y_pf_sig, fs);

            % ViSQOL (subset)
            if n_visqol > 0 && ii <= n_vq
                try
                    [vq, ~, ~] = visqol(x_mic, s, fs, Mode='speech');
                    visqol_mic(ii) = vq;
                    [vq, ~, ~] = visqol(y_mvdr_sig, s, fs, Mode='speech');
                    visqol_mvdr(ii) = vq;
                    [vq, ~, ~] = visqol(y_pf_sig, s, fs, Mode='speech');
                    visqol_pf(ii) = vq;
                catch
                end
            end

            % PESQ (subset)
            if n_pesq > 0 && ii <= n_pq
                try
                    pesq_mic(ii)  = pesq(s, x_mic, fs);
                    pesq_mvdr(ii) = pesq(s, y_mvdr_sig, fs);
                    pesq_pf(ii)   = pesq(s, y_pf_sig, fs);
                catch
                end
            end

        catch ME
            fprintf('  ERROR [%s]: %s\n', fname, ME.message);
            n_errors = n_errors + 1;
        end
    end

    elapsed_total = toc(t_start);
    fprintf('\n=== Evaluation done [%s] in %.1f min ===\n', ver, elapsed_total/60);
    fprintf('Errors: %d / %d\n\n', n_errors, n_files);

    %% ---- Quality metrics report ----
    valid = ~isnan(sisdr_mic);

    fprintf('========================================\n');
    fprintf('  QUALITY METRICS [%s]  (%d valid samples)\n', upper(ver), sum(valid));
    fprintf('========================================\n');

    % SI-SDR
    fprintf('\n--- SI-SDR vs dry clean (dB) ---\n');
    if strcmp(ver, 'compensated')
        fprintf('  (Delay-aligned — optimal for SI-SDR)\n');
    else
        fprintf('  (Raw timing — includes propagation delay mismatch)\n');
    end
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

    % OSINR
    fprintf('\n--- OSINR vs dry clean (dB) ---\n');
    fprintf('%-12s  %8s  %8s  %8s  %8s\n', '', 'Mean', 'Std', 'Min', 'Max');
    fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'Mic (ch1)', ...
        mean(osinr_mic(valid)), std(osinr_mic(valid)), min(osinr_mic(valid)), max(osinr_mic(valid)));
    fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'MVDR', ...
        mean(osinr_mvdr(valid)), std(osinr_mvdr(valid)), min(osinr_mvdr(valid)), max(osinr_mvdr(valid)));
    fprintf('%-12s  %8.2f  %8.2f  %8.2f  %8.2f\n', 'MVDR+PF', ...
        mean(osinr_pf(valid)), std(osinr_pf(valid)), min(osinr_pf(valid)), max(osinr_pf(valid)));

    fprintf('\n  MVDR gain:     %+.2f dB  (over mic)\n', mean(osinr_mvdr(valid) - osinr_mic(valid)));
    fprintf('  PostFilt gain: %+.2f dB  (over mic)\n', mean(osinr_pf(valid) - osinr_mic(valid)));

    % Per-type SI-SDR
    fprintf('\n--- SI-SDR by interference type ---\n');
    fprintf('%-6s  %6s  %8s  %8s  %8s  %9s\n', 'Type', 'Count', 'Mic', 'MVDR', 'MVDR+PF', 'PF gain');
    for t = {'A','B','C'}
        mask = valid & strcmp(interf_types, t{1});
        if any(mask)
            fprintf('%-6s  %6d  %8.2f  %8.2f  %8.2f  %+9.2f\n', t{1}, sum(mask), ...
                mean(sisdr_mic(mask)), mean(sisdr_mvdr(mask)), ...
                mean(sisdr_pf(mask)), mean(sisdr_pf(mask) - sisdr_mic(mask)));
        end
    end

    % STOI
    fprintf('\n--- STOI ---\n');
    fprintf('%-12s  %8s  %8s  %8s  %8s\n', '', 'Mean', 'Std', 'Min', 'Max');
    fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'Mic (ch1)', ...
        mean(stoi_mic(valid)), std(stoi_mic(valid)), min(stoi_mic(valid)), max(stoi_mic(valid)));
    fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'MVDR', ...
        mean(stoi_mvdr(valid)), std(stoi_mvdr(valid)), min(stoi_mvdr(valid)), max(stoi_mvdr(valid)));
    fprintf('%-12s  %8.3f  %8.3f  %8.3f  %8.3f\n', 'MVDR+PF', ...
        mean(stoi_pf(valid)), std(stoi_pf(valid)), min(stoi_pf(valid)), max(stoi_pf(valid)));

    % ViSQOL
    if n_visqol > 0
        valid_vq = ~isnan(visqol_mic);
        n_vq_valid = sum(valid_vq);
        if n_vq_valid > 0
            fprintf('\n--- ViSQOL MOS (%d samples) ---\n', n_vq_valid);
            fprintf('%-12s  %8s  %8s\n', '', 'Mean', 'Std');
            fprintf('%-12s  %8.2f  %8.2f\n', 'Mic (ch1)', ...
                mean(visqol_mic(valid_vq)), std(visqol_mic(valid_vq)));
            fprintf('%-12s  %8.2f  %8.2f\n', 'MVDR', ...
                mean(visqol_mvdr(valid_vq)), std(visqol_mvdr(valid_vq)));
            fprintf('%-12s  %8.2f  %8.2f\n', 'MVDR+PF', ...
                mean(visqol_pf(valid_vq)), std(visqol_pf(valid_vq)));
        end
    end

    % PESQ
    if n_pesq > 0
        valid_pq = ~isnan(pesq_mic);
        n_pq_valid = sum(valid_pq);
        if n_pq_valid > 0
            fprintf('\n--- PESQ MOS (%d samples) ---\n', n_pq_valid);
            fprintf('%-12s  %8s  %8s\n', '', 'Mean', 'Std');
            fprintf('%-12s  %8.2f  %8.2f\n', 'Mic (ch1)', ...
                mean(pesq_mic(valid_pq)), std(pesq_mic(valid_pq)));
            fprintf('%-12s  %8.2f  %8.2f\n', 'MVDR', ...
                mean(pesq_mvdr(valid_pq)), std(pesq_mvdr(valid_pq)));
            fprintf('%-12s  %8.2f  %8.2f\n', 'MVDR+PF', ...
                mean(pesq_pf(valid_pq)), std(pesq_pf(valid_pq)));
        end
    end

    %% ---- Sanity checks ----
    fprintf('\n========================================\n');
    fprintf('  SANITY CHECK [%s]\n', upper(ver));
    fprintf('========================================\n');

    sisdr_pf_mean = mean(sisdr_pf(valid));
    stoi_pf_mean  = mean(stoi_pf(valid));

    if strcmp(ver, 'uncompensated')
        fprintf('  Note: uncompensated SI-SDR will be lower due to ~%d-sample delay\n', ...
            ternary(has_rir, rir_data.prop_delay_smp, 47));
    end

    if strcmp(ver, 'compensated')
        if sisdr_pf_mean > 15
            fprintf('  ⚠ SI-SDR(PF) = %.1f dB seems too high — check delay alignment\n', sisdr_pf_mean);
        elseif sisdr_pf_mean < -10
            fprintf('  ⚠ SI-SDR(PF) = %.1f dB seems too low — check MVDR pipeline\n', sisdr_pf_mean);
        else
            fprintf('  ✓ SI-SDR(PF) = %.1f dB — reasonable for reverberant Task 2\n', sisdr_pf_mean);
        end
    else
        % Uncompensated: SI-SDR typically much lower (negative)
        if sisdr_pf_mean > 5
            fprintf('  ⚠ SI-SDR(PF) = %.1f dB — suspiciously high for uncompensated\n', sisdr_pf_mean);
        else
            fprintf('  ✓ SI-SDR(PF) = %.1f dB — expected for uncompensated (delay mismatch)\n', sisdr_pf_mean);
        end
    end

    if stoi_pf_mean > 0.95 && strcmp(ver, 'compensated')
        fprintf('  ⚠ STOI(PF) = %.3f seems too high for reverberant conditions\n', stoi_pf_mean);
    elseif stoi_pf_mean < 0.2
        fprintf('  ⚠ STOI(PF) = %.3f seems too low — check pipeline\n', stoi_pf_mean);
    else
        fprintf('  ✓ STOI(PF) = %.3f — reasonable\n', stoi_pf_mean);
    end

    mvdr_gain = mean(sisdr_mvdr(valid) - sisdr_mic(valid));
    pf_gain   = mean(sisdr_pf(valid) - sisdr_mic(valid));
    if mvdr_gain < 0
        fprintf('  ⚠ MVDR gain = %+.1f dB — MVDR should improve over mic\n', mvdr_gain);
    else
        fprintf('  ✓ MVDR gain = %+.1f dB over mic\n', mvdr_gain);
    end
    if pf_gain < mvdr_gain
        fprintf('  ⚠ PF gain (%+.1f) < MVDR gain (%+.1f) — post-filter should help\n', pf_gain, mvdr_gain);
    else
        fprintf('  ✓ PF gain = %+.1f dB over mic (PF adds %+.1f dB over MVDR)\n', ...
            pf_gain, pf_gain - mvdr_gain);
    end

    %% ---- Histograms ----
    fig_title = sprintf('Task 2 Reverb Verification — %s', ver);
    figure('Name', fig_title, 'Position', [50+400*(vi-1) 50 1500 1000]);

    % SI-SDR
    subplot(3,3,1);
    histogram(sisdr_mic(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.8 0.2 0.2]); hold on;
    histogram(sisdr_mvdr(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.6 0.8]);
    histogram(sisdr_pf(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.8 0.2]);
    legend('Mic','MVDR','MVDR+PF'); xlabel('SI-SDR (dB)'); ylabel('Count');
    title('SI-SDR vs Dry Clean'); grid on;

    % OSINR
    subplot(3,3,2);
    histogram(osinr_mic(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.8 0.2 0.2]); hold on;
    histogram(osinr_mvdr(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.6 0.8]);
    histogram(osinr_pf(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.8 0.2]);
    legend('Mic','MVDR','MVDR+PF'); xlabel('OSINR (dB)'); ylabel('Count');
    title('OSINR'); grid on;

    % STOI
    subplot(3,3,3);
    histogram(stoi_mic(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.8 0.2 0.2]); hold on;
    histogram(stoi_mvdr(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.6 0.8]);
    histogram(stoi_pf(valid), 50, 'FaceAlpha', 0.5, 'FaceColor', [0.2 0.8 0.2]);
    legend('Mic','MVDR','MVDR+PF'); xlabel('STOI'); ylabel('Count');
    title('STOI vs Dry Clean'); grid on;

    % SI-SDR gain
    subplot(3,3,4);
    histogram(sisdr_pf(valid) - sisdr_mic(valid), 50, 'FaceColor', [0.4 0.7 0.3]);
    xlabel('SI-SDR gain (dB)'); ylabel('Count');
    title('MVDR+PF Gain over Mic'); grid on;
    xline(mean(sisdr_pf(valid) - sisdr_mic(valid)), 'r--', 'LineWidth', 2);

    % Per-type boxplot
    subplot(3,3,5);
    type_labels = interf_types(valid);
    sisdr_vals  = sisdr_pf(valid);
    types_unique = {'A','B','C'};
    type_numeric = nan(size(type_labels));
    for ti = 1:3
        type_numeric(strcmp(type_labels, types_unique{ti})) = ti;
    end
    valid_typed = ~isnan(type_numeric);
    if any(valid_typed)
        boxplot(sisdr_vals(valid_typed), type_numeric(valid_typed), ...
            'Labels', {'A (Female)', 'B (Music)', 'C (Noise)'});
        ylabel('SI-SDR (dB)'); title('MVDR+PF SI-SDR by Type'); grid on;
    end

    % STOI gain
    subplot(3,3,6);
    histogram(stoi_pf(valid) - stoi_mic(valid), 50, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('STOI gain'); ylabel('Count');
    title('MVDR+PF STOI Gain over Mic'); grid on;
    xline(mean(stoi_pf(valid) - stoi_mic(valid)), 'r--', 'LineWidth', 2);

    if has_meta
        eval_n = min(height(meta), n_files);

        subplot(3,3,7);
        histogram(meta.sir_db(1:eval_n), 50, 'FaceColor', [0.6 0.4 0.2]);
        xlabel('SIR (dB)'); ylabel('Count');
        title('SIR Distribution'); grid on;
        xline(0, 'r--', 'Competition', 'LineWidth', 2);

        subplot(3,3,8);
        histogram(meta.snr_db(1:eval_n), 50, 'FaceColor', [0.5 0.3 0.6]);
        xlabel('SNR (dB)'); ylabel('Count');
        title('SNR Distribution'); grid on;
        xline(5, 'r--', 'Competition', 'LineWidth', 2);

        subplot(3,3,9);
        histogram(meta.theta_interf(1:eval_n), 50, 'FaceColor', [0.2 0.6 0.5]);
        xlabel('\theta_{interf} (deg)'); ylabel('Count');
        title('\theta_{interf} Distribution'); grid on;
        xline(40, 'r--', 'Competition', 'LineWidth', 2);
    end

    sgtitle(fig_title, 'FontSize', 14, 'FontWeight', 'bold');

    %% ---- Store results for this version ----
    res = struct();
    res.version   = ver;
    res.n_samples = sum(valid);
    res.n_errors  = n_errors;
    res.sisdr = struct('mic', sisdr_mic(valid), 'mvdr', sisdr_mvdr(valid), 'pf', sisdr_pf(valid));
    res.osinr = struct('mic', osinr_mic(valid), 'mvdr', osinr_mvdr(valid), 'pf', osinr_pf(valid));
    res.stoi  = struct('mic', stoi_mic(valid),  'mvdr', stoi_mvdr(valid),  'pf', stoi_pf(valid));
    res.interf_types = interf_types(valid);

    if n_visqol > 0
        vv = ~isnan(visqol_mic);
        res.visqol = struct('mic', visqol_mic(vv), 'mvdr', visqol_mvdr(vv), 'pf', visqol_pf(vv));
    end
    if n_pesq > 0
        pp = ~isnan(pesq_mic);
        res.pesq = struct('mic', pesq_mic(pp), 'mvdr', pesq_mvdr(pp), 'pf', pesq_pf(pp));
    end

    all_results.(ver) = res;

    %% ---- Per-version summary ----
    fprintf('\n--- SUMMARY [%s] ---\n', upper(ver));
    fprintf('Samples evaluated: %d  (errors: %d)\n', sum(valid), n_errors);
    fprintf('Mean SI-SDR:  Mic=%+.2f  MVDR=%+.2f  PF=%+.2f dB\n', ...
        mean(sisdr_mic(valid)), mean(sisdr_mvdr(valid)), mean(sisdr_pf(valid)));
    fprintf('Mean OSINR:   Mic=%+.2f  MVDR=%+.2f  PF=%+.2f dB\n', ...
        mean(osinr_mic(valid)), mean(osinr_mvdr(valid)), mean(osinr_pf(valid)));
    fprintf('Mean STOI:    Mic=%.3f   MVDR=%.3f   PF=%.3f\n', ...
        mean(stoi_mic(valid)), mean(stoi_mvdr(valid)), mean(stoi_pf(valid)));
    fprintf('Gains:        MVDR=%+.2f  PF=%+.2f dB (SI-SDR over mic)\n', ...
        mean(sisdr_mvdr(valid) - sisdr_mic(valid)), ...
        mean(sisdr_pf(valid) - sisdr_mic(valid)));

end   % version loop

%% ======================== CROSS-VERSION COMPARISON ========================
if length(version_names) == 2
    fprintf('\n\n############################################################\n');
    fprintf('  CROSS-VERSION COMPARISON: compensated vs uncompensated\n');
    fprintf('############################################################\n\n');

    rc = all_results.compensated;
    ru = all_results.uncompensated;

    fprintf('%-20s  %12s  %12s  %12s\n', 'Metric', 'Compensated', 'Uncomp.', 'Delta');
    fprintf('%-20s  %12.2f  %12.2f  %+12.2f\n', 'SI-SDR PF (dB)', ...
        mean(rc.sisdr.pf), mean(ru.sisdr.pf), mean(rc.sisdr.pf) - mean(ru.sisdr.pf));
    fprintf('%-20s  %12.2f  %12.2f  %+12.2f\n', 'OSINR PF (dB)', ...
        mean(rc.osinr.pf), mean(ru.osinr.pf), mean(rc.osinr.pf) - mean(ru.osinr.pf));
    fprintf('%-20s  %12.3f  %12.3f  %+12.3f\n', 'STOI PF', ...
        mean(rc.stoi.pf), mean(ru.stoi.pf), mean(rc.stoi.pf) - mean(ru.stoi.pf));
    fprintf('%-20s  %12.2f  %12.2f  %+12.2f\n', 'SI-SDR gain (dB)', ...
        mean(rc.sisdr.pf - rc.sisdr.mic), mean(ru.sisdr.pf - ru.sisdr.mic), ...
        mean(rc.sisdr.pf - rc.sisdr.mic) - mean(ru.sisdr.pf - ru.sisdr.mic));

    fprintf('\nExpected: compensated SI-SDR ~30 dB higher (delay alignment).\n');
    fprintf('STOI/PESQ should be similar (both handle small delays internally).\n');
end

%% ======================== SAVE ALL RESULTS ========================
if has_rir
    all_results.rt60_target    = rir_data.RT60_target;
    all_results.rt60_achieved  = rir_data.rt60_mid;
    all_results.absorption     = rir_data.absorption_final;
    all_results.prop_delay_smp = rir_data.prop_delay_smp;
end

save(fullfile(dataset_root, 'verification_results.mat'), 'all_results');
fprintf('\nResults saved to verification_results.mat\n');

fprintf('\n========================================\n');
fprintf('  VERIFICATION COMPLETE\n');
fprintf('  Output: %s\n', dataset_root);
fprintf('========================================\n');


%% ################################################################
%  LOCAL FUNCTIONS
%  ################################################################

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function rt60 = estimate_rt60_local(ir, fs)
    ir = ir(:);
    energy = cumsum(ir(end:-1:1).^2);
    energy = energy(end:-1:1);
    energy = energy / max(energy + eps);
    energy_db = 10*log10(energy + eps);

    idx_5db = find(energy_db <= -5, 1, 'first');
    if isempty(idx_5db), rt60 = NaN; return; end

    idx_35db = find(energy_db <= -35, 1, 'first');
    if ~isempty(idx_35db)
        rt60 = 2 * ((idx_35db - idx_5db) / fs);
        return;
    end

    idx_25db = find(energy_db <= -25, 1, 'first');
    if ~isempty(idx_25db)
        rt60 = 3 * ((idx_25db - idx_5db) / fs);
        return;
    end

    idx_15db = find(energy_db <= -15, 1, 'first');
    if ~isempty(idx_15db)
        rt60 = 6 * ((idx_15db - idx_5db) / fs);
        return;
    end

    rt60 = NaN;
end
