%% ================================================================
%  generate_mixture_Reverb_MVDR_V3.m
%  Task 2 Dataset Generator — Reverberant MVDR + Zelinski Post-Filter
%
%  Competition parameters (SP Cup 2026):
%    Room:  4.9 × 4.9 × 4.9 m shoebox
%    RT60:  0.5 s (ISM with binary-search absorption tuning)
%    Array: 2-mic ULA, d = 0.08 m, at room centre, height 1.5 m
%    Target:       broadside (90° cos convention), 1 m distance
%    Interference: 40° off-axis (fixed), ~1 m distance
%    SIR = 0 dB ± N(0,0.5),  SNR: 80% at 5±N(0,0.5) / 20% U[2,8]
%
%  Pipeline per sample:
%    Pre-computed RIR → fftfilt convolve → SIR scale → AWGN
%    → STFT → batch Rxx → MVDR(δ=1e-3) → Zelinski PF(α=0.92,β=0.02)
%    → ISTFT → save both compensated & uncompensated → metadata
%
%  Pre-computation (before loop):
%    1. Binary search for absorption coefficient → RT60 = 0.50 s
%    2. Target RIR: fixed position (2.45, 3.45, 1.50) → 2 mics
%    3. Interference RIR: single azimuth at 40° (fixed)
%
%  Output — two parallel trees (configurable via save_* flags):
%    <root>/compensated/    — delay-aligned pairs (for DNN training)
%      clean/, Mixtures/, MVDR_outputs/, MVDR_filtered/
%    <root>/uncompensated/  — raw timing (for testing / competition eval)
%      clean/, Mixtures/, MVDR_outputs/, MVDR_filtered/
%
%  Training pair alignment (compensated only):
%    The target RIR introduces a propagation delay (~47 samples at 1m).
%    Compensated pairs are sample-aligned (clean trimmed from end,
%    reverberant/processed trimmed from start) so the DNN trains on
%    the actual enhancement task, not a constant time shift.
%
%  Naming: maleAudioName_combinationType_interferenceName.wav
%
%  Date: 11 Feb 2026
% =================================================================

clear; clc; close all;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');

%% ======================== CONFIG ========================
rng(5);

% ---------- How many samples to generate ----------
n_total = 50;   % total samples across all types

% ---------- Audio ----------
fs       = 16000;
duration = 3;       % seconds
L        = fs * duration;

% ---------=- Room (competition spec) ----------
room_dim = [4.9, 4.9, 4.9];   % metres
RT60_target = 0.50;            % seconds

% ---------- Array geometry ----------
c = 340;                           % m/s
d = 0.08;                          % mic spacing
mic_pos_1d = [-d/2; d/2];         % for steering vector (1D)

array_centre = [2.45, 2.45, 1.50];
mic_pos_3d   = [array_centre(1)-d/2, array_centre(2), array_centre(3);
                array_centre(1)+d/2, array_centre(2), array_centre(3)];

% ---------- Source positions ----------
source_target = [2.45, 3.45, 1.50];   % broadside, 1 m
dist_target   = norm(source_target - array_centre);

% Interference positions: computed from azimuth (see RIR bank below)
dist_interf = 1.0;   % metres from array centre

% ---------- DOAs (MATLAB convention: 90°=broadside, cos delay) ----------
theta_target = 90;   % always broadside

% ---------- STFT / MVDR (must match inference pipeline) ----------
N_fft_win = 512;
hop       = 128;
nfft      = 512;
window    = sqrt(hann(N_fft_win, 'periodic'));
delta     = 1e-3;    % diagonal loading

% ---------- ISM ----------
ism_order_init = 15;   % auto-increased below

% ---------- Source folders ----------
male_folder   = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Male';
female_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Female';
music_folder  = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Music';
noise_folder  = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Noise';

% ---------- Output ----------
output_base = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/prepared_dataset_reverb_v3';

% ---------- Save configuration ----------
%  Toggle which versions to write.  At least one must be true.
save_compensated   = true;    % delay-aligned pairs (for DNN training)
save_uncompensated = true;    % raw timing (for competition / testing)

assert(save_compensated || save_uncompensated, ...
    'At least one of save_compensated / save_uncompensated must be true.');

% ---------- Interference direction (fixed) ----------
theta_interf_fixed = 40;   % degrees (competition spec, no variation)

% ---------- Interference type shares ----------
%  A = Female (50%), B = Music (30%), C = Noise-file (20%)
type_shares = struct('A', 0.50, 'B', 0.30, 'C', 0.20);

%% ======================== FILE LISTING ========================
male_files   = [dir(fullfile(male_folder, '*.flac')); dir(fullfile(male_folder, '*.wav'))];
female_files = [dir(fullfile(female_folder, '*.flac')); dir(fullfile(female_folder, '*.wav'))];
music_files  = [dir(fullfile(music_folder, '*.flac')); dir(fullfile(music_folder, '*.wav'))];
noise_files  = [dir(fullfile(noise_folder, '*.flac')); dir(fullfile(noise_folder, '*.wav'))];

fprintf('============================================\n');
fprintf('  TASK 2 REVERBERANT DATASET GENERATOR\n');
fprintf('============================================\n\n');
fprintf('=== Source files ===\n');
fprintf('  Male:   %d\n', length(male_files));
fprintf('  Female: %d\n', length(female_files));
fprintf('  Music:  %d\n', length(music_files));
fprintf('  Noise:  %d\n\n', length(noise_files));

assert(~isempty(male_files),   'No male files found in %s', male_folder);
assert(~isempty(female_files), 'No female files found in %s', female_folder);
assert(~isempty(music_files),  'No music files found in %s', music_folder);
assert(~isempty(noise_files),  'No noise files found in %s', noise_folder);

%% ======================== COMPUTE COUNTS PER TYPE ========================
n_A = round(n_total * type_shares.A);
n_B = round(n_total * type_shares.B);
n_C = n_total - n_A - n_B;

fprintf('=== Samples per type ===\n');
fprintf('  A (Female):     %d  (%.0f%%)\n', n_A, 100*n_A/n_total);
fprintf('  B (Music):      %d  (%.0f%%)\n', n_B, 100*n_B/n_total);
fprintf('  C (Noise-file): %d  (%.0f%%)\n\n', n_C, 100*n_C/n_total);

%% ======================== CREATE OUTPUT DIRS ========================
sub_dirs = {'clean', 'Mixtures', 'MVDR_outputs', 'MVDR_filtered'};
versions = {};
if save_compensated,   versions{end+1} = 'compensated';   end
if save_uncompensated, versions{end+1} = 'uncompensated'; end

for vi = 1:length(versions)
    for di = 1:length(sub_dirs)
        dpath = fullfile(output_base, versions{vi}, sub_dirs{di});
        if ~exist(dpath, 'dir'), mkdir(dpath); end
    end
end

%% ================================================================
%  PRE-COMPUTATION: ISM RIR BANK
%  ================================================================
fprintf('============================================\n');
fprintf('  PRE-COMPUTING RIR BANK\n');
fprintf('============================================\n\n');
t_pre = tic;

% ---- Room acoustics ----
V      = prod(room_dim);
S_area = 2*(room_dim(1)*room_dim(2) + room_dim(1)*room_dim(3) + ...
             room_dim(2)*room_dim(3));
alpha_eyring = 1 - exp(-0.161 * V / (S_area * RT60_target));
fprintf('  Eyring absorption estimate: α = %.4f\n', alpha_eyring);

% ---- ISM order ----
refl_eyring = sqrt(1 - alpha_eyring);
bounces_35dB = ceil(3.5 / (-log10(refl_eyring + eps)));
ism_order = max(bounces_35dB + 5, 30);
if ism_order_init < ism_order
    fprintf('  Auto-increasing ISM order: %d → %d\n', ism_order_init, ism_order);
end
fprintf('  ISM order: %d\n\n', ism_order);

% ---- Binary search for absorption ----
fprintf('  Tuning absorption via binary search (target RT60 = %.2f s) ...\n', RT60_target);
scatter_final = 0.0;
alpha_lo = 0.05;
alpha_hi = 0.80;

ir_lo = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
    fs, c, ism_order, alpha_lo*ones(6,1), scatter_final*ones(6,1));
rt60_lo = estimate_rt60(ir_lo(1,:), fs);

ir_hi = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
    fs, c, ism_order, alpha_hi*ones(6,1), scatter_final*ones(6,1));
rt60_hi = estimate_rt60(ir_hi(1,:), fs);

fprintf('    α = %.3f → RT60 = %.3f s\n', alpha_lo, rt60_lo);
fprintf('    α = %.3f → RT60 = %.3f s\n', alpha_hi, rt60_hi);

if isnan(rt60_lo) || isnan(rt60_hi)
    error('Cannot bracket RT60 — check ISM parameters');
end

for iter = 1:30
    alpha_mid = (alpha_lo + alpha_hi) / 2;
    ir_mid = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
        fs, c, ism_order, alpha_mid*ones(6,1), scatter_final*ones(6,1));
    rt60_mid = estimate_rt60(ir_mid(1,:), fs);

    if isnan(rt60_mid)
        alpha_hi = alpha_mid;
        continue;
    end

    if rt60_mid > RT60_target
        alpha_lo = alpha_mid;
    else
        alpha_hi = alpha_mid;
    end

    if abs(rt60_mid - RT60_target) < 0.005
        break;
    end
end

absorption_final = alpha_mid;
refl_final = sqrt(max(0, 1 - absorption_final));
fprintf('    Binary search: %d iterations\n', iter);
fprintf('  Result:  α = %.4f   refl/bounce = %.4f\n', absorption_final, refl_final);
fprintf('  RT60 = %.3f s   (target %.3f s, err = %.3f s)  ✓\n\n', ...
    rt60_mid, RT60_target, abs(rt60_mid - RT60_target));

% ---- Generate target RIR (fixed position) ----
fprintf('  Generating target RIR (source → 2 mics) ...\n');
rir_target = ismShoeboxRIR(room_dim, source_target, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));

% Propagation delay from target RIR (for training pair alignment)
[~, peak_idx_target] = max(abs(rir_target(1,:)));
prop_delay_smp = peak_idx_target - 1;
fprintf('  Target RIR: %d samples (%.3f s)\n', size(rir_target,2), size(rir_target,2)/fs);
fprintf('  Propagation delay: %d samples (%.2f ms)\n\n', prop_delay_smp, prop_delay_smp/fs*1000);

% RT60 verification
rt60_tgt1 = estimate_rt60(rir_target(1,:), fs);
rt60_tgt2 = estimate_rt60(rir_target(2,:), fs);
fprintf('  RT60 target→mic1: %.3f s\n', rt60_tgt1);
fprintf('  RT60 target→mic2: %.3f s\n\n', rt60_tgt2);

% ---- Generate interference RIR (fixed 40°) ----
az_interf = theta_interf_fixed;   % degrees
az_rad_interf = deg2rad(az_interf);
source_interf = array_centre + dist_interf * [cos(az_rad_interf), sin(az_rad_interf), 0];

fprintf('  Generating interference RIR at %.1f° (single direction) ...\n', az_interf);
rir_interf = ismShoeboxRIR(room_dim, source_interf, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));

rt60_interf = estimate_rt60(rir_interf(1,:), fs);
fprintf('  Interference RIR: %d samples (%.3f s)\n', size(rir_interf,2), size(rir_interf,2)/fs);
fprintf('  RT60 (interf→mic1): %.3f s\n', rt60_interf);

fprintf('\n  Pre-computation time: %.1f s\n', toc(t_pre));
fprintf('  RIR bank: 1 target + 1 interference (%.1f°) = 2 RIR pairs\n\n', az_interf);

%% ======================== BUILD SAMPLE LIST ========================
sample_list = cell(n_total, 1);
idx = 1;
for ii = 1:n_A, sample_list{idx} = 'A'; idx = idx+1; end
for ii = 1:n_B, sample_list{idx} = 'B'; idx = idx+1; end
for ii = 1:n_C, sample_list{idx} = 'C'; idx = idx+1; end

% Shuffle
perm = randperm(n_total);
sample_list = sample_list(perm);

%% ======================== METADATA TABLE ========================
meta_id        = cell(n_total, 1);
meta_type      = cell(n_total, 1);
meta_sir       = zeros(n_total, 1);
meta_snr       = zeros(n_total, 1);
meta_theta_int = zeros(n_total, 1);
meta_male_file = cell(n_total, 1);
meta_int_file  = cell(n_total, 1);

%% ======================== PRE-COMPUTE STEERING ========================
freqs_vec = (0:(nfft/2))' * fs / nfft;
dvec_target = compute_steering_vector(theta_target, freqs_vec, mic_pos_1d, c);

%% ======================== MAIN GENERATION LOOP ========================
t_start = tic;
n_errors = 0;
used_names = containers.Map();

fprintf('============================================\n');
fprintf('  GENERATING %d REVERBERANT SAMPLES\n', n_total);
fprintf('============================================\n\n');

for ii = 1:n_total

    % ---- Progress ----
    if mod(ii, 100) == 0 || ii == 1
        elapsed = toc(t_start);
        rate = ii / elapsed;
        eta  = (n_total - ii) / rate;
        pct  = 100 * ii / n_total;
        fprintf('[%d/%d] (%.1f%%)  %.1f samples/s  ETA: %.0f min  | A:%d B:%d C:%d  errors:%d\n', ...
            ii, n_total, pct, rate, eta/60, ...
            sum(strcmp(meta_type(1:ii), 'A')), ...
            sum(strcmp(meta_type(1:ii), 'B')), ...
            sum(strcmp(meta_type(1:ii), 'C')), ...
            n_errors);
    end

    try
        % ---- Interference type ----
        itype = sample_list{ii};

        % ---- Sample conditions ----
        sir_db       = sample_sir();
        snr_db       = sample_snr();
        theta_interf = theta_interf_fixed;   % fixed 40°

        % ---- Pick unique (male, type, interf) combination ----
        max_retries = 50;
        found_valid = false;
        for retry = 1:max_retries
            male_idx = randi(length(male_files));
            switch itype
                case 'A', int_idx = randi(length(female_files)); int_fname = female_files(int_idx).name;
                case 'B', int_idx = randi(length(music_files));  int_fname = music_files(int_idx).name;
                case 'C', int_idx = randi(length(noise_files));  int_fname = noise_files(int_idx).name;
            end
            [~, male_base, ~] = fileparts(male_files(male_idx).name);
            [~, int_base, ~]  = fileparts(int_fname);
            out_name = sprintf('%s_%s_%s.wav', male_base, itype, int_base);
            if ~used_names.isKey(out_name)
                found_valid = true;
                break;
            end
        end
        if ~found_valid
            error('Could not find unique combo after %d retries', max_retries);
        end

        % ---- Load audio ----
        target_sig = load_audio_file( ...
            fullfile(male_folder, male_files(male_idx).name), fs, duration);

        switch itype
            case 'A'
                interf_sig = load_audio_file( ...
                    fullfile(female_folder, female_files(int_idx).name), fs, duration);
            case 'B'
                interf_sig = load_audio_file( ...
                    fullfile(music_folder, music_files(int_idx).name), fs, duration);
            case 'C'
                interf_sig = load_audio_file( ...
                    fullfile(noise_folder, noise_files(int_idx).name), fs, duration);
        end

        % Skip silent sources
        if rms(target_sig) < 1e-6 || rms(interf_sig) < 1e-6
            n_errors = n_errors + 1;
            continue;
        end

        % Reserve name only after confirming valid audio
        used_names(out_name) = true;

        % ---- Create reverberant 2-channel mixture ----
        [mixture, target_mc, interf_mc, noise_mc] = create_mixture_reverb( ...
            target_sig, interf_sig, rir_target, rir_interf, ...
            L, sir_db, snr_db);

        % ---- STFT ----
        X = stft_multichannel(mixture, window, hop, nfft);
        [numFreqs, numFrames, numMics] = size(X);

        % ---- Batch sample covariance ----
        Rxx = zeros(numMics, numMics, numFreqs);
        for n = 1:numFrames
            X_frame = squeeze(X(:,n,:));
            for k = 1:numFreqs
                xk = X_frame(k,:).';
                Rxx(:,:,k) = Rxx(:,:,k) + (xk * xk');
            end
        end
        Rxx = Rxx / numFrames;

        % ---- MVDR weights ----
        W = compute_mvdr_weights(Rxx, dvec_target, delta);

        % ---- MVDR only ----
        Y_mvdr = apply_mvdr(X, W);

        % ---- MVDR + Zelinski post-filter ----
        Y_pf = mvdr_postfilter(X, W, dvec_target);

        % ---- ISTFT ----
        y_mvdr = real(istft_single_channel(Y_mvdr, window, hop, nfft, L));
        y_mvdr = y_mvdr(1:L);

        y_pf = real(istft_single_channel(Y_pf, window, hop, nfft, L));
        y_pf = y_pf(1:L);

        % ---- Clean reference: dry target (DNN ground truth) ----
        clean_dry = target_sig(1:L);

        % ---- Mixture stereo (full, unaligned — for reference) ----
        mix_stereo = mixture(1:L, :);

        % ---- Trim 20 ms boundary artefacts ----
        trim_samp = round(0.020 * fs);   % 320 samples
        trim_idx  = (trim_samp + 1) : (L - trim_samp);

        y_mvdr    = y_mvdr(trim_idx);
        y_pf      = y_pf(trim_idx);
        clean_dry = clean_dry(trim_idx);
        mix_stereo = mix_stereo(trim_idx, :);

        % =============== UNCOMPENSATED (raw timing) ===============
        if save_uncompensated
            uc_clean  = safe_peak_norm(clean_dry);
            uc_mvdr   = safe_peak_norm(y_mvdr);
            uc_pf     = safe_peak_norm(y_pf);
            uc_mix    = safe_peak_norm_stereo(mix_stereo);

            audiowrite(fullfile(output_base,'uncompensated','clean',         out_name), uc_clean, fs);
            audiowrite(fullfile(output_base,'uncompensated','Mixtures',      out_name), uc_mix,   fs);
            audiowrite(fullfile(output_base,'uncompensated','MVDR_outputs',  out_name), uc_mvdr,  fs);
            audiowrite(fullfile(output_base,'uncompensated','MVDR_filtered', out_name), uc_pf,    fs);
        end

        % =============== COMPENSATED (delay-aligned for training) ===============
        if save_compensated
            if prop_delay_smp > 0
                N_align = length(clean_dry) - prop_delay_smp;
                c_clean = clean_dry(1:N_align);
                c_mvdr  = y_mvdr(prop_delay_smp+1 : prop_delay_smp+N_align);
                c_pf    = y_pf(prop_delay_smp+1 : prop_delay_smp+N_align);
                c_mix   = mix_stereo(prop_delay_smp+1 : prop_delay_smp+N_align, :);
            else
                c_clean = clean_dry;
                c_mvdr  = y_mvdr;
                c_pf    = y_pf;
                c_mix   = mix_stereo;
            end

            c_clean = safe_peak_norm(c_clean);
            c_mvdr  = safe_peak_norm(c_mvdr);
            c_pf    = safe_peak_norm(c_pf);
            c_mix   = safe_peak_norm_stereo(c_mix);

            audiowrite(fullfile(output_base,'compensated','clean',         out_name), c_clean, fs);
            audiowrite(fullfile(output_base,'compensated','Mixtures',      out_name), c_mix,   fs);
            audiowrite(fullfile(output_base,'compensated','MVDR_outputs',  out_name), c_mvdr,  fs);
            audiowrite(fullfile(output_base,'compensated','MVDR_filtered', out_name), c_pf,    fs);
        end

        % ---- Metadata ----
        meta_id{ii}        = out_name;
        meta_type{ii}      = itype;
        meta_sir(ii)       = sir_db;
        meta_snr(ii)       = snr_db;
        meta_theta_int(ii) = theta_interf;
        meta_male_file{ii} = male_files(male_idx).name;
        meta_int_file{ii}  = int_fname;

    catch ME
        fprintf('  ERROR sample %d: %s\n', ii, ME.message);
        n_errors = n_errors + 1;
    end
end

elapsed_total = toc(t_start);
fprintf('\n=== DONE ===\n');
fprintf('Generated %d samples in %.1f min (%.1f samples/s)\n', ...
    n_total - n_errors, elapsed_total/60, (n_total-n_errors)/elapsed_total);
fprintf('Errors/skipped: %d\n\n', n_errors);

%% ======================== SAVE METADATA ========================
T = table(meta_id, meta_type, meta_sir, meta_snr, ...
          meta_theta_int, meta_male_file, meta_int_file, ...
    'VariableNames', {'filename','interf_type','sir_db','snr_db', ...
                      'theta_interf','male_file','interf_file'});
T = T(~cellfun(@isempty, T.filename), :);

writetable(T, fullfile(output_base, 'metadata.csv'));
fprintf('Metadata saved: %s\n', fullfile(output_base, 'metadata.csv'));

% Save dataset info
fid = fopen(fullfile(output_base, 'dataset_info.txt'), 'w');
fprintf(fid, 'Dataset: Reverb_MVDR_V3  (Task 2)\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Total samples: %d\n', n_total - n_errors);
fprintf(fid, '\nRoom:\n');
fprintf(fid, '  Dimensions: [%.1f × %.1f × %.1f] m\n', room_dim);
fprintf(fid, '  RT60 target: %.2f s\n', RT60_target);
fprintf(fid, '  RT60 achieved: %.3f s\n', rt60_mid);
fprintf(fid, '  Absorption α: %.4f\n', absorption_final);
fprintf(fid, '  Scatter: %.2f\n', scatter_final);
fprintf(fid, '  ISM order: %d\n', ism_order);
fprintf(fid, '\nGeometry:\n');
fprintf(fid, '  Array centre: [%.2f, %.2f, %.2f]\n', array_centre);
fprintf(fid, '  Mic spacing: %.2f m\n', d);
fprintf(fid, '  Target position: [%.2f, %.2f, %.2f]  (broadside, 1m)\n', source_target);
fprintf(fid, '  Interf distance: %.2f m\n', dist_interf);
fprintf(fid, '  Interf direction: %.1f° (fixed)\n', az_interf);
fprintf(fid, '  Propagation delay: %d samples (%.2f ms)\n', prop_delay_smp, prop_delay_smp/fs*1000);
fprintf(fid, '\nSave configuration:\n');
fprintf(fid, '  save_compensated:   %d\n', save_compensated);
fprintf(fid, '  save_uncompensated: %d\n', save_uncompensated);
fprintf(fid, '\nOutput folders (under each version):\n');
fprintf(fid, '  clean/         — dry target mono (DNN ground truth)\n');
fprintf(fid, '  Mixtures/      — 2-ch stereo reverberant mixture\n');
fprintf(fid, '  MVDR_outputs/  — mono MVDR-only output\n');
fprintf(fid, '  MVDR_filtered/ — mono MVDR + Zelinski PF output\n');
fprintf(fid, '\n  compensated/   — delay-aligned (training pairs)\n');
fprintf(fid, '  uncompensated/ — raw timing (competition / testing)\n');
fprintf(fid, '\nNaming: maleAudioName_combinationType_interferenceName.wav\n');
fprintf(fid, '\nConditions:\n');
fprintf(fid, '  theta_target = 90° (broadside, cos convention)\n');
fprintf(fid, '  theta_interf = 40° (fixed, competition spec)\n');
fprintf(fid, '  SIR = 0 dB ± N(0,0.5), clamped [-2,2]\n');
fprintf(fid, '  SNR: 80%% at 5±N(0,0.5) dB, 20%% uniform [2,8]\n');
fprintf(fid, '  Sensor noise: AWGN on every sample\n');
fprintf(fid, '  RT60 = %.2f s (fixed, competition spec)\n', RT60_target);
fprintf(fid, '\nInterference types:\n');
fprintf(fid, '  A (Female): 50%%\n');
fprintf(fid, '  B (Music):  30%%\n');
fprintf(fid, '  C (Noise):  20%%\n');
fprintf(fid, '\nPipeline:\n');
fprintf(fid, '  Pre-computed ISM RIR → fftfilt convolve → SIR scale → AWGN\n');
fprintf(fid, '  → STFT(512) → batch Rxx → MVDR(δ=1e-3) → Zelinski PF(α=0.92,β=0.02) → ISTFT\n');
fprintf(fid, '  → save uncompensated (raw) + compensated (delay-align %d smp)\n', prop_delay_smp);
fprintf(fid, 'Clean ref: dry target (ground truth for DNN dereverberation + denoising)\n');
fclose(fid);
fprintf('Dataset info saved.\n');

% Save RIR data and parameters for reproducibility
save(fullfile(output_base, 'rir_bank.mat'), ...
    'rir_target', 'rir_interf', 'az_interf', 'source_interf', ...
    'absorption_final', 'scatter_final', 'refl_final', 'ism_order', ...
    'room_dim', 'mic_pos_3d', 'source_target', 'dist_interf', ...
    'prop_delay_smp', 'RT60_target', 'rt60_mid', 'fs', 'c', 'd', ...
    'save_compensated', 'save_uncompensated', ...
    '-v7.3');
fprintf('RIR data saved: %s\n', fullfile(output_base, 'rir_bank.mat'));

fprintf('\n============================================\n');
fprintf('  Dataset generation complete.\n');
fprintf('  Output: %s\n', output_base);
fprintf('============================================\n');


%% ################################################################
%  LOCAL FUNCTIONS
%  ################################################################

%% ---- Sample SIR: 0 dB ± N(0, 0.5), clamped [-2, 2] ----
function sir = sample_sir()
    sir = 0 + 0.5 * randn();
    sir = max(-2, min(2, sir));
end

%% ---- Sample SNR: 80% at 5 ± N(0,0.5) dB, 20% uniform [2,8] ----
function snr = sample_snr()
    if rand() < 0.80
        snr = 5 + 0.5 * randn();
        snr = max(3, min(7, snr));
    else
        snr = 2 + 6 * rand();
    end
end

%% ---- Create reverberant 2-channel mixture ----
function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_mixture_reverb(target_sig, interf_sig, rir_target, rir_interf, L, sir_db, snr_db)
    %
    %  Convolve each mono source with its 2-mic RIR, then scale for SIR/SNR.
    %
    %  target_sig : [L×1] dry mono target
    %  interf_sig : [L×1] dry mono interference
    %  rir_target : [2×Lrir] target→mic1, target→mic2
    %  rir_interf : [2×Lrir] interf→mic1, interf→mic2
    %  L          : desired output length (samples)
    %  sir_db     : target SIR (dB)
    %  snr_db     : target SNR (dB)
    %

    target_mc = zeros(L, 2);
    interf_mc = zeros(L, 2);

    for ch = 1:2
        t_conv = fftfilt(rir_target(ch,:)', target_sig);
        i_conv = fftfilt(rir_interf(ch,:)', interf_sig);
        target_mc(:, ch) = t_conv(1:L);
        interf_mc(:, ch) = i_conv(1:L);
    end

    % ---- SIR scaling (global, preserves spatial structure) ----
    Pt = mean(target_mc(:).^2) + eps;
    Pi = mean(interf_mc(:).^2) + eps;
    sir_lin = 10^(sir_db / 10);
    interf_mc = interf_mc * sqrt(Pt / (Pi * sir_lin));

    % ---- Clean mixture (before noise) ----
    mixture_clean = target_mc + interf_mc;

    % ---- AWGN sensor noise at target SNR ----
    Ps = mean(mixture_clean(:).^2) + eps;
    snr_lin = 10^(snr_db / 10);
    noise_power = Ps / snr_lin;
    noise_mc = sqrt(noise_power) * randn(size(mixture_clean));

    mixture = mixture_clean + noise_mc;

    % ---- Peak-normalize (preserves relative levels) ----
    peak = max(abs(mixture(:)));
    if peak > 0
        sc = 0.99 / peak;
        mixture   = mixture   * sc;
        target_mc = target_mc * sc;
        interf_mc = interf_mc * sc;
        noise_mc  = noise_mc  * sc;
    end
end

%% ---- Peak-normalize mono signal to 0.99 ----
function x = safe_peak_norm(x)
    pk = max(abs(x(:)));
    if pk > 0
        x = 0.99 * x / pk;
    end
end

%% ---- Peak-normalize stereo signal to 0.99 (joint) ----
function x = safe_peak_norm_stereo(x)
    pk = max(abs(x(:)));
    if pk > 0
        x = 0.99 * x / pk;
    end
end

%% ---- Load and prepare audio file ----
function sig = load_audio_file(filepath, fs, duration)
    target_samples = fs * duration;
    try
        [sig, file_fs] = audioread(filepath);
        if file_fs ~= fs
            sig = resample(sig, fs, file_fs);
        end
        if size(sig, 2) > 1
            sig = mean(sig, 2);
        end
        if length(sig) > target_samples
            sig = sig(1:target_samples);
        elseif length(sig) < target_samples
            sig = [sig; zeros(target_samples - length(sig), 1)];
        end
        sig = sig - mean(sig);   % DC removal
    catch
        sig = zeros(target_samples, 1);
    end
end

%% ==================== ISM SHOEBOX RIR ====================
function ir = ismShoeboxRIR(roomDim, tx, rx, fs, c, order, absorption6, scattering6)
    L = double(roomDim(:)).';
    tx = double(tx);
    rx = double(rx);

    alpha = absorption6(:);
    scat  = scattering6(:);
    refl  = sqrt(max(0, 1 - alpha)) .* sqrt(max(0, 1 - scat));

    n = -order:order;
    [Nx, Ny, Nz] = ndgrid(n, n, n);
    keep = (abs(Nx) + abs(Ny) + abs(Nz)) <= order;
    Nx = Nx(keep); Ny = Ny(keep); Nz = Nz(keep);

    Ximg = ((-1).^Nx) .* tx(1) + 2 .* Nx .* L(1);
    Yimg = ((-1).^Ny) .* tx(2) + 2 .* Ny .* L(2);
    Zimg = ((-1).^Nz) .* tx(3) + 2 .* Nz .* L(3);

    ax = abs(Nx); ay = abs(Ny); az = abs(Nz);

    nLeft = zeros(size(Nx)); nRight = nLeft;
    nFront = nLeft; nBack = nLeft;
    nFloor = nLeft; nCeil = nLeft;

    pos = Nx >= 0;
    nRight(pos)  = ceil(ax(pos) ./ 2);
    nLeft(pos)   = floor(ax(pos) ./ 2);
    nLeft(~pos)  = ceil(ax(~pos) ./ 2);
    nRight(~pos) = floor(ax(~pos) ./ 2);

    pos = Ny >= 0;
    nFront(pos)  = ceil(ay(pos) ./ 2);
    nBack(pos)   = floor(ay(pos) ./ 2);
    nBack(~pos)  = ceil(ay(~pos) ./ 2);
    nFront(~pos) = floor(ay(~pos) ./ 2);

    pos = Nz >= 0;
    nCeil(pos)   = ceil(az(pos) ./ 2);
    nFloor(pos)  = floor(az(pos) ./ 2);
    nFloor(~pos) = ceil(az(~pos) ./ 2);
    nCeil(~pos)  = floor(az(~pos) ./ 2);

    reflGain = (refl(4).^nLeft) .* (refl(5).^nRight) .* ...
               (refl(2).^nFront) .* (refl(3).^nBack) .* ...
               (refl(1).^nFloor) .* (refl(6).^nCeil);

    numRx = size(rx, 1);
    ir = zeros(numRx, 1);

    for rxi = 1:numRx
        dx = Ximg - rx(rxi,1);
        dy = Yimg - rx(rxi,2);
        dz = Zimg - rx(rxi,3);
        dist = sqrt(dx.^2 + dy.^2 + dz.^2);

        valid = dist > 0;
        dist = dist(valid);
        g = reflGain(valid) ./ dist;

        sFloat = (dist ./ c) .* fs;
        s0   = floor(sFloat);
        frac = sFloat - s0;
        idx1 = s0 + 1;
        idx2 = idx1 + 1;

        Lr = max(idx2);
        if Lr > size(ir, 2)
            ir(:, end+1:Lr) = 0;
        end

        w1 = g .* (1 - frac);
        w2 = g .* frac;

        ir_r = accumarray(idx1, w1, [Lr,1], @sum, 0) + ...
               accumarray(idx2, w2, [Lr,1], @sum, 0);
        ir(rxi, 1:Lr) = ir(rxi, 1:Lr) + ir_r(:).';
    end
end

%% ==================== RT60 ESTIMATION ====================
function rt60 = estimate_rt60(ir, fs)
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
