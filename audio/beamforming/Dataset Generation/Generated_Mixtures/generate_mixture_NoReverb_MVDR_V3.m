%% ===============================(FINAL V2)=============================
%  generate_mixture_NoReverb_MVDR_V3.m 
%  Task 1 Dataset Generator  —  MVDR + Zelinski Post-Filter
%
%  Competition-focused: heavy bias toward exact test conditions
%    • Target at 90° (broadside, MATLAB cos convention)
%    • Interference at 40° ± N(0,5°)
%    • SIR = 0 dB ± N(0,0.5)
%    • SNR: 80% at 5 ± N(0,0.5) dB, 20% uniform [2,8] dB
%    • AWGN sensor noise on EVERY sample
%
%  3 interference types: Female (50%), Music (30%), Noise-file (20%)
%
%  Pipeline per sample:
%    spatialize → scale SIR → add AWGN → STFT → batch Rxx
%    → MVDR → Zelinski post-filter → ISTFT → save
%
%  Output folders:
%    clean/          = target_mc(:,1)  (mono clean reference)
%    Mixtures/       = 2-ch stereo mixture BEFORE any beamforming
%    MVDR_outputs/   = MVDR-only output (mono)
%    MVDR_filtered/  = MVDR + Zelinski post-filter output (mono)
%
%  Naming: maleAudioName_combinationType_interferenceName.wav
%
%  Date: 8 Feb 2026
% =============================================================

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

% ---------- Array geometry ----------
c   = 340;
d   = 0.08;
mic_pos = [-d/2; d/2];

% ---------- DOAs (MATLAB convention: 90°=broadside, cos delay) ----------
theta_target = 90;   % always broadside

% ---------- STFT / MVDR (must match inference pipeline) ----------
N_fft_win = 512;
hop       = 128;
nfft      = 512;
window    = sqrt(hann(N_fft_win, 'periodic'));
delta     = 1e-3;   % diagonal loading

% ---------- Post-filter ----------
alpha_pf = 0.92;
beta_pf  = 0.02;

% ---------- Source folders ----------
male_folder   = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Male';
female_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Female';
music_folder  = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Music';
noise_folder  = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Noise';

% ---------- Output ----------
output_base = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/prepared_dataset_v3';

% ---------- Interference type shares ----------
%  A = Female (50%), B = Music (30%), C = Noise-file (20%)
type_shares = struct('A', 0.50, 'B', 0.30, 'C', 0.20);

%% ======================== FILE LISTING ========================
male_files   = [dir(fullfile(male_folder, '*.flac')); dir(fullfile(male_folder, '*.wav'))];
female_files = [dir(fullfile(female_folder, '*.flac')); dir(fullfile(female_folder, '*.wav'))];
music_files  = [dir(fullfile(music_folder, '*.flac')); dir(fullfile(music_folder, '*.wav'))];
noise_files  = [dir(fullfile(noise_folder, '*.flac')); dir(fullfile(noise_folder, '*.wav'))];

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
n_C = n_total - n_A - n_B;   % remainder to avoid rounding drift

fprintf('=== Samples per type ===\n');
fprintf('  A (Female):     %d  (%.0f%%)\n', n_A, 100*n_A/n_total);
fprintf('  B (Music):      %d  (%.0f%%)\n', n_B, 100*n_B/n_total);
fprintf('  C (Noise-file): %d  (%.0f%%)\n\n', n_C, 100*n_C/n_total);

%% ======================== CREATE OUTPUT DIRS ========================
out_dirs = {'clean', 'Mixtures', 'MVDR_outputs', 'MVDR_filtered'};
for ii_d = 1:length(out_dirs)
    dpath = fullfile(output_base, out_dirs{ii_d});
    if ~exist(dpath, 'dir'), mkdir(dpath); end
end

%% ======================== BUILD SAMPLE LIST ========================
% Each entry: {interf_type, sample_index_within_type}
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
% Steering vector for target (always 90°) — reuse for every sample
freqs_vec = (0:(nfft/2))' * fs / nfft;   % numFreqs = nfft/2 + 1
dvec_target = compute_steering_vector(theta_target, freqs_vec, mic_pos, c);

%% ======================== MAIN GENERATION LOOP ========================
t_start = tic;
n_errors = 0;
used_names = containers.Map();   % track unique (male, type, interf) combos

fprintf('=== Starting generation of %d samples ===\n\n', n_total);

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
        sir_db      = sample_sir();
        snr_db      = sample_snr();
        theta_interf = sample_theta_interf();

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

        % Skip silent sources (don't reserve the name)
        if rms(target_sig) < 1e-6 || rms(interf_sig) < 1e-6
            n_errors = n_errors + 1;
            continue;
        end

        % Reserve name only after confirming valid audio
        used_names(out_name) = true;

        % ---- Create 2-channel mixture ----
        [mixture, target_mc, interf_mc, noise_mc] = create_mixture_v3( ...
            target_sig, interf_sig, theta_target, theta_interf, ...
            fs, sir_db, snr_db, c, d);

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

        % ---- Clean reference: ch1 of target-only multichannel ----
        clean_ref = target_mc(1:L, 1);

        % ---- Trim boundary artifacts (20 ms each side) ----
        %  ISTFT edge: winNorm under-normalized (few overlapping windows)
        %  Post-filter edge: EMA cold-start + zero-pad artefacts
        trim_samp = round(0.020 * fs);   % 320 samples at 16 kHz
        trim_idx  = (trim_samp + 1) : (L - trim_samp);

        clean_ref = clean_ref(trim_idx);
        y_mvdr    = y_mvdr(trim_idx);
        y_pf      = y_pf(trim_idx);
        mixture   = mixture(trim_idx, :);   % both channels

        % ---- Peak-normalize each to avoid clipping ----
        clean_ref = safe_peak_norm(clean_ref);
        y_mvdr    = safe_peak_norm(y_mvdr);
        y_pf      = safe_peak_norm(y_pf);
        mixture   = safe_peak_norm_stereo(mixture);

        % ---- Save all 4 outputs ----
        audiowrite(fullfile(output_base, 'clean',          out_name), clean_ref, fs);
        audiowrite(fullfile(output_base, 'Mixtures',       out_name), mixture,   fs);
        audiowrite(fullfile(output_base, 'MVDR_outputs',   out_name), y_mvdr,    fs);
        audiowrite(fullfile(output_base, 'MVDR_filtered',  out_name), y_pf,      fs);

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
fprintf('Errors/skipped: %d\n', n_errors);

%% ======================== SAVE METADATA ========================
T = table(meta_id, meta_type, meta_sir, meta_snr, ...
          meta_theta_int, meta_male_file, meta_int_file, ...
    'VariableNames', {'filename','interf_type','sir_db','snr_db', ...
                      'theta_interf','male_file','interf_file'});
% Remove empty rows from errors
T = T(~cellfun(@isempty, T.filename), :);

writetable(T, fullfile(output_base, 'metadata.csv'));
fprintf('Metadata saved: %s\n', fullfile(output_base, 'metadata.csv'));

% Save dataset info
fid = fopen(fullfile(output_base, 'dataset_info.txt'), 'w');
fprintf(fid, 'Dataset: NoReverb_MVDR_V3\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Total samples: %d\n', n_total - n_errors);
fprintf(fid, '\nOutput folders:\n');
fprintf(fid, '  clean/         — mono clean target reference\n');
fprintf(fid, '  Mixtures/      — 2-ch stereo mixture (before MVDR)\n');
fprintf(fid, '  MVDR_outputs/  — mono MVDR-only output\n');
fprintf(fid, '  MVDR_filtered/ — mono MVDR + Zelinski post-filter output\n');
fprintf(fid, '\nNaming: maleAudioName_combinationType_interferenceName.wav\n');
fprintf(fid, '\nConditions:\n');
fprintf(fid, '  theta_target = 90° (broadside, cos convention)\n');
fprintf(fid, '  theta_interf = 40° ± N(0,5°), clamped [25°,55°]\n');
fprintf(fid, '  SIR = 0 dB ± N(0,0.5), clamped [-2,2]\n');
fprintf(fid, '  SNR: 80%% at 5±N(0,0.5) dB, 20%% uniform [2,8]\n');
fprintf(fid, '  Sensor noise: AWGN on every sample\n');
fprintf(fid, '\nInterference types:\n');
fprintf(fid, '  A (Female): 50%%\n');
fprintf(fid, '  B (Music):  30%%\n');
fprintf(fid, '  C (Noise):  20%%\n');
fprintf(fid, '\nPipeline: spatialize → SIR scale → AWGN → STFT(512) → batch Rxx → MVDR(δ=1e-3) → Zelinski PF(α=0.92,β=0.02) → ISTFT\n');
fprintf(fid, 'Clean ref: target_mc(:,1)\n');
fclose(fid);
fprintf('Dataset info saved.\n');


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
        snr = max(3, min(7, snr));   % soft clamp around center
    else
        snr = 2 + 6 * rand();        % uniform [2, 8]
    end
end

%% ---- Sample θ_interf: 40° ± N(0, 5°), clamped [25°, 55°] ----
function theta = sample_theta_interf()
    theta = 40 + 5 * randn();
    theta = max(25, min(55, theta));
end

%% ---- Create 2-channel mixture with explicit SIR & SNR control ----
function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_mixture_v3(target, interf, theta_target, theta_interf, ...
                      fs, sir_db, snr_db, c, d)
    %
    %  MATLAB convention: delay = micPos * cos(azimuth) / c
    %  90° = broadside (zero delay), 0°/180° = endfire
    %

    micPos = [-d/2; d/2];

    % Length match
    Lsig = min(length(target), length(interf));
    target = target(1:Lsig);
    interf = interf(1:Lsig);

    % Fractional-delay spatialization (plane wave, cos convention)
    apply_delay = @(x, tau) interp1( ...
        (0:length(x)-1)', x, ...
        (0:length(x)-1)' - tau*fs, ...
        'linear', 0);

    make_mc = @(sig, az) ...
        [ apply_delay(sig, micPos(1)*cos(deg2rad(az))/c), ...
          apply_delay(sig, micPos(2)*cos(deg2rad(az))/c) ];

    target_mc = make_mc(target, theta_target);
    interf_mc = make_mc(interf, theta_interf);

    % ---- SIR control (across both channels) ----
    Pt = mean(target_mc(:).^2) + eps;
    Pi = mean(interf_mc(:).^2) + eps;
    sir_lin = 10^(sir_db / 10);
    interf_mc = interf_mc * sqrt(Pt / (Pi * sir_lin));

    % ---- Clean mixture (before noise) ----
    mixture_clean = target_mc + interf_mc;

    % ---- Add AWGN sensor noise at target SNR (per channel) ----
    Ps = mean(mixture_clean(:).^2) + eps;
    snr_lin = 10^(snr_db / 10);
    noise_power = Ps / snr_lin;
    noise_mc = sqrt(noise_power) * randn(size(mixture_clean));

    mixture = mixture_clean + noise_mc;

    % ---- Peak-normalize mixture (preserves relative levels) ----
    peak = max(abs(mixture(:)));
    if peak > 0
        scale = 0.99 / peak;
        mixture   = mixture   * scale;
        target_mc = target_mc * scale;
        interf_mc = interf_mc * scale;
        noise_mc  = noise_mc  * scale;
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

        % Resample if needed
        if file_fs ~= fs
            sig = resample(sig, fs, file_fs);
        end

        % Mono
        if size(sig, 2) > 1
            sig = mean(sig, 2);
        end

        % Ensure exact length
        if length(sig) > target_samples
            sig = sig(1:target_samples);
        elseif length(sig) < target_samples
            sig = [sig; zeros(target_samples - length(sig), 1)];
        end

        % Remove DC
        sig = sig - mean(sig);
    catch
        sig = zeros(target_samples, 1);
    end
end
