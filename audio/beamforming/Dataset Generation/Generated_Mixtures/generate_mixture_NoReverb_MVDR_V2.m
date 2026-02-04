%% Mixture Generation (Augmented) - No Reverb + Optional MVDR
% Creates training mixtures with richer SNR/SIR variation and mild realism
% augmentations. Emphasizes SNR = 5 dB and SIR = 0 dB.
%
% Notes:
% - This script is designed to be drop-in compatible with your existing
%   beamforming MVDR helpers (stft_multichannel, compute_mvdr_weights, etc.).
% - By default, it assumes you run it from your dataset-generation folder
%   where '../Male_clean', '../Female', '../Music', '../Noise' exist.
%   If you run from elsewhere, set cfg.data.* paths below.
%
% Current date: 31 Jan 2026

clear; clc; close all;

%% ---------------- Config ----------------
cfg = struct();

% Random seed
cfg.rng = 'shuffle'; % 'shuffle' | numeric seed

% How many targets to process (random sampling with replacement)
cfg.n_to_process = 2;

% Audio
cfg.fs = 16000;
cfg.duration_sec = 3;

% Array geometry
cfg.c = 340;
cfg.d = 0.08;
cfg.mic_pos = [-cfg.d/2; cfg.d/2];

% DOAs (degrees)
cfg.theta_target_actual = 0;   % keep target at 0° to match your MVDR steering
cfg.theta_target_mvdr   = 0;   % steering angle used by MVDR
cfg.theta_interf_range  = [20 80];

% MVDR
cfg.apply_mvdr_to_mixture = true;
cfg.mvdr_delta = 1e-3;
cfg.mvdr_alpha = 0.98;

% STFT params (MVDR)
cfg.stft.N = 256;
cfg.stft.hop = 128;
cfg.stft.nfft = 512;
cfg.stft.window = sqrt(hann(cfg.stft.N,'periodic'));

% Output
cfg.output_dir = './NoReverb_V2'; % default: current folder
cfg.save_clean_reference = false; % saves first-mic target as clean ref
cfg.save_metadata_mat = false;

% Data folders (relative to where you run the script)
cfg.data.male_folder   = '../Male_clean';
cfg.data.female_folder = '../Female';
cfg.data.music_folder  = '../Music';
cfg.data.noise_folder  = '../Noise';

% How many variants per target per "combo" (A-E)
cfg.variants_per_combo = 2; % increase if you want more diversity

% Emphasize SNR=5 dB and SIR=0 dB via biased sampling
cfg.sampling = struct();

% SIR: mostly around 0 dB, sometimes harder/easier
cfg.sampling.sir.center_db = 0;
cfg.sampling.sir.center_prob = 0.65;
cfg.sampling.sir.center_std_db = 1.0;
cfg.sampling.sir.other_values_db = [-10 -5 -2.5 2.5 5 10];
cfg.sampling.sir.other_probs = [0.07 0.10 0.10 0.10 0.10 0.08];

% SNR: strongly around 5 dB; also include harder and easier cases
cfg.sampling.snr.center_db = 5;
cfg.sampling.snr.center_prob = 0.55;
cfg.sampling.snr.center_std_db = 1.2;
cfg.sampling.snr.other_values_db = [-5 0 2.5 7.5 10 15 20];
cfg.sampling.snr.other_probs = [0.05 0.10 0.10 0.08 0.07 0.03 0.02];

% Clamp ranges (prevents extreme outliers)
cfg.sampling.sir_clamp_db = [-12 12];
cfg.sampling.snr_clamp_db = [-8 22];

% Mild augmentations (kept conservative to avoid hurting STOI/PESQ)
cfg.aug = struct();

% Random crop (prevents always taking first 3s)
cfg.aug.random_crop = false;

% Random overall level for each source (dBFS RMS)
cfg.aug.random_source_level = true;
cfg.aug.source_level_dbfs_range = [-30 -20];

% Mild spectral shaping per source (simulates mic/codec coloration)
cfg.aug.apply_mild_filter = true;
cfg.aug.filter_prob = 0.35;

% Optional soft clipping (rare)
cfg.aug.soft_clip_prob = 0.05;

% Optional time-varying noise envelope (helps nonstationary noise robustness)
cfg.aug.dynamic_noise_prob = 0.25;

% Add a second interferer sometimes (multi-talker / multi-source)
cfg.aug.add_second_interferer_prob = 0.25;

% Use real noise file as noise component when available
% (preferred over AWGN; better for PESQ/ViSQOL realism)
cfg.noise = struct();
cfg.noise.use_real_noise_prob = 0.80;

% Peak normalization safety
cfg.peak_target = 0.99;





%% ---------------- Setup ----------------
if isequal(cfg.rng, 'shuffle')
    rng('shuffle');
else
    rng(cfg.rng);
end

if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

% Add beamforming helpers (edit this if your path differs)
% This is needed only when cfg.apply_mvdr_to_mixture = true.
try
    addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
catch
end

fprintf('=== Augmented Mixture Generation (No Reverb) ===\n');
fprintf('Targets to process: %d\n', cfg.n_to_process);
fprintf('Variants per combo: %d\n', cfg.variants_per_combo);
fprintf('Emphasis: SNR≈%g dB, SIR≈%g dB\n\n', cfg.sampling.snr.center_db, cfg.sampling.sir.center_db);

% Get file lists
male_files = dir(fullfile(cfg.data.male_folder, '*.flac'));
if isempty(male_files)
    male_files = dir(fullfile(cfg.data.male_folder, '*flac'));
end
female_files = dir(fullfile(cfg.data.female_folder, '*.flac'));
music_files  = dir(fullfile(cfg.data.music_folder, '*.flac'));
noise_files  = dir(fullfile(cfg.data.noise_folder, '*.flac'));

if isempty(male_files)
    error('No male files found in %s', cfg.data.male_folder);
end
if isempty(female_files)
    error('No female files found in %s', cfg.data.female_folder);
end
if isempty(music_files)
    error('No music files found in %s', cfg.data.music_folder);
end
if isempty(noise_files)
    warning('No noise files found in %s (will fall back to AWGN)', cfg.data.noise_folder);
end

fprintf('Found files:\n  Male: %d\n  Female: %d\n  Music: %d\n  Noise: %d\n\n', ...
    numel(male_files), numel(female_files), numel(music_files), numel(noise_files));

%% ---------------- Main loop ----------------
processed = 0;

for iter = 1:min(cfg.n_to_process, numel(male_files))
    male_idx = randi(numel(male_files));
    male_file = male_files(male_idx);
    male_path = fullfile(cfg.data.male_folder, male_file.name);

    target_sig = load_audio_file_random(male_path, cfg.fs, cfg.duration_sec, cfg.aug.random_crop);
    if all(target_sig == 0)
        continue;
    end

    [~, male_base, ~] = fileparts(male_file.name);

    % Pick sources
    female_idx = randi(numel(female_files));
    music_idx  = randi(numel(music_files));

    female_sig = load_audio_file_random(fullfile(cfg.data.female_folder, female_files(female_idx).name), cfg.fs, cfg.duration_sec, cfg.aug.random_crop);
    music_sig  = load_audio_file_random(fullfile(cfg.data.music_folder,  music_files(music_idx).name),  cfg.fs, cfg.duration_sec, cfg.aug.random_crop);

    if ~isempty(noise_files)
        noise_idx = randi(numel(noise_files));
        noise_sig = load_audio_file_random(fullfile(cfg.data.noise_folder, noise_files(noise_idx).name), cfg.fs, cfg.duration_sec, cfg.aug.random_crop);
        noise_name = noise_files(noise_idx).name;
    else
        noise_sig = zeros(cfg.fs * cfg.duration_sec, 1);
        noise_name = 'NONE';
    end

    % Optional: normalize each source to a random RMS level
    if cfg.aug.random_source_level
        target_sig = normalize_rms_dbfs(target_sig, rand_range(cfg.aug.source_level_dbfs_range));
        female_sig = normalize_rms_dbfs(female_sig, rand_range(cfg.aug.source_level_dbfs_range));
        music_sig  = normalize_rms_dbfs(music_sig,  rand_range(cfg.aug.source_level_dbfs_range));
        if any(noise_sig)
            noise_sig  = normalize_rms_dbfs(noise_sig,  rand_range(cfg.aug.source_level_dbfs_range));
        end
    end

    % Optional mild coloration
    if cfg.aug.apply_mild_filter
        if rand < cfg.aug.filter_prob
            target_sig = mild_channel_filter(target_sig, cfg.fs);
        end
        if rand < cfg.aug.filter_prob
            female_sig = mild_channel_filter(female_sig, cfg.fs);
        end
        if rand < cfg.aug.filter_prob
            music_sig  = mild_channel_filter(music_sig, cfg.fs);
        end
        if any(noise_sig) && rand < cfg.aug.filter_prob
            noise_sig  = mild_channel_filter(noise_sig, cfg.fs);
        end
    end

    % Build combos where SIR controls interferer(s) and SNR controls additive noise
    combos = {
        struct('tag','A','interf', female_sig, 'noise', [] , 'desc','female'), ...
        struct('tag','B','interf', female_sig + music_sig, 'noise', [] , 'desc','female+music'), ...
        struct('tag','C','interf', female_sig, 'noise', noise_sig, 'desc','female+realnoise'), ...
        struct('tag','D','interf', music_sig,  'noise', noise_sig, 'desc','music+realnoise'), ...
        struct('tag','E','interf', female_sig + music_sig, 'noise', noise_sig, 'desc','female+music+realnoise') ...
    };

    for ci = 1:numel(combos)
        combo = combos{ci};

        for vi = 1:cfg.variants_per_combo
            [sir_db, snr_db] = sample_sir_snr(cfg.sampling, cfg.sampling.sir_clamp_db, cfg.sampling.snr_clamp_db);

            theta_interf = rand_range(cfg.theta_interf_range);
            theta_target_actual = cfg.theta_target_actual;

            % Sometimes add a second interferer (multi-source), while keeping overall SIR defined
            interf_sig = combo.interf;
            if rand < cfg.aug.add_second_interferer_prob
                % pick another talker/music as an extra interferer
                if rand < 0.6
                    female_idx2 = randi(numel(female_files));
                    extra = load_audio_file_random(fullfile(cfg.data.female_folder, female_files(female_idx2).name), cfg.fs, cfg.duration_sec, cfg.aug.random_crop);
                else
                    music_idx2 = randi(numel(music_files));
                    extra = load_audio_file_random(fullfile(cfg.data.music_folder, music_files(music_idx2).name), cfg.fs, cfg.duration_sec, cfg.aug.random_crop);
                end
                if cfg.aug.random_source_level
                    extra = normalize_rms_dbfs(extra, rand_range(cfg.aug.source_level_dbfs_range));
                end
                if cfg.aug.apply_mild_filter && rand < cfg.aug.filter_prob
                    extra = mild_channel_filter(extra, cfg.fs);
                end

                % keep extra quieter so it doesn't dominate
                extra = extra * db2mag(-6 + 6*randn());
                interf_sig = interf_sig + extra;

                % spread the extra interferer spatially (random within range)
                theta_interf = rand_range(cfg.theta_interf_range);
            end

            % Decide whether to use real noise as additive noise (SNR-controlled)
            use_real_noise = (~isempty(noise_files)) && (rand < cfg.noise.use_real_noise_prob) && any(combo.noise);
            if use_real_noise
                noise_src = combo.noise;
            else
                noise_src = [];
            end

            [mix_mc, target_mc, interf_mc, noise_mc, meta] = create_mixture_no_reverb_aug( ...
                target_sig, interf_sig, noise_src, ...
                theta_target_actual, theta_interf, ...
                cfg.fs, sir_db, snr_db, ...
                cfg.c, cfg.d, cfg.mic_pos, ...
                cfg.aug.dynamic_noise_prob);

            % Optional rare soft clipping on the final mixture (simulates device saturation)
            if rand < cfg.aug.soft_clip_prob
                mix_mc = soft_clip_tanh(mix_mc, 1.5 + 0.5*rand);
            end

            % Safety peak normalization
            [mix_mc, target_mc, interf_mc, noise_mc] = peak_normalize_all(mix_mc, target_mc, interf_mc, noise_mc, cfg.peak_target);

            % Optional MVDR (outputs single-channel)
            mvdr_wav = [];
            if cfg.apply_mvdr_to_mixture
                mvdr_wav = apply_mvdr_pipeline(mix_mc, cfg);
                if ~isempty(mvdr_wav)
                    mvdr_wav = mvdr_wav(:);
                    mvdr_wav = peak_normalize(mvdr_wav, cfg.peak_target);
                end
            end

            % Save
            snr_tag = fmt_db_tag(snr_db);
            sir_tag = fmt_db_tag(sir_db);
            base = sprintf('%s_%s_v%03d_SIR%s_SNR%s', male_base, combo.tag, vi, sir_tag, snr_tag);

            out_mix = fullfile(cfg.output_dir, sprintf('%s_mix.wav', base));
            audiowrite(out_mix, mix_mc, cfg.fs);

            if cfg.save_clean_reference
                out_clean = fullfile(cfg.output_dir, sprintf('%s_clean.wav', base));
                % Reference: first mic target channel
                audiowrite(out_clean, target_mc(:,1), cfg.fs);
            end

            if cfg.apply_mvdr_to_mixture && ~isempty(mvdr_wav)
                out_mvdr = fullfile(cfg.output_dir, sprintf('%s_mvdr.wav', base));
                audiowrite(out_mvdr, mvdr_wav, cfg.fs);
            end

            if cfg.save_metadata_mat
                meta_out = fullfile(cfg.output_dir, sprintf('%s_meta.mat', base));
                meta.combo = combo;
                meta.files.male   = male_file.name;
                meta.files.female = female_files(female_idx).name;
                meta.files.music  = music_files(music_idx).name;
                meta.files.noise  = noise_name;
                meta.sir_db = sir_db;
                meta.snr_db = snr_db;
                meta.theta_target_actual = theta_target_actual;
                meta.theta_interf = theta_interf;
                meta.used_real_noise = use_real_noise;
                save(meta_out, 'meta');
            end
        end
    end

    processed = processed + 1;
    if mod(processed, 50) == 0
        fprintf('Processed %d targets...\n', processed);
    end
end

fprintf('\n=== COMPLETE ===\n');
fprintf('Processed targets: %d\n', processed);
fprintf('Output dir: %s\n', cfg.output_dir);

%% ---------------- Local functions ----------------

function [sir_db, snr_db] = sample_sir_snr(sampling, sir_clamp_db, snr_clamp_db)
    sir_db = sample_biased_db(sampling.sir.center_db, sampling.sir.center_prob, sampling.sir.center_std_db, ...
        sampling.sir.other_values_db, sampling.sir.other_probs);
    snr_db = sample_biased_db(sampling.snr.center_db, sampling.snr.center_prob, sampling.snr.center_std_db, ...
        sampling.snr.other_values_db, sampling.snr.other_probs);

    sir_db = clamp(sir_db, sir_clamp_db(1), sir_clamp_db(2));
    snr_db = clamp(snr_db, snr_clamp_db(1), snr_clamp_db(2));
end

function v = sample_biased_db(center_db, center_prob, center_std_db, other_values_db, other_probs)
    r = rand;
    if r < center_prob
        v = center_db + center_std_db * randn;
    else
        other_probs = other_probs(:)';
        other_probs = other_probs / sum(other_probs);
        idx = sample_discrete(other_probs);
        v = other_values_db(idx) + 0.6 * randn;
    end
end

function idx = sample_discrete(probs)
    cdf = cumsum(probs);
    r = rand;
    idx = find(r <= cdf, 1, 'first');
    if isempty(idx)
        idx = numel(probs);
    end
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end

function val = rand_range(rng2)
    val = rng2(1) + (rng2(2)-rng2(1)) * rand;
end

function sig = load_audio_file_random(filepath, fs, duration_sec, random_crop)
    target_samples = round(fs * duration_sec);

    try
        [sig, file_fs] = audioread(filepath);
        if file_fs ~= fs
            sig = resample(sig, fs, file_fs);
        end
        if size(sig,2) > 1
            sig = mean(sig,2);
        end
        sig = sig - mean(sig);

        if length(sig) >= target_samples
            if random_crop
                max_start = length(sig) - target_samples + 1;
                start = randi(max_start);
                sig = sig(start:start+target_samples-1);
            else
                sig = sig(1:target_samples);
            end
        else
            sig = [sig; zeros(target_samples - length(sig), 1)];
        end

        if all(sig == 0)
            sig = zeros(target_samples, 1);
        end

    catch
        sig = zeros(target_samples, 1);
    end
end

function sig = normalize_rms_dbfs(sig, target_dbfs)
    % Normalize to RMS target in dBFS, ignoring silence.
    eps_pow = 1e-12;
    p = mean(sig.^2);
    if p < eps_pow
        return;
    end
    rms = sqrt(p);

    % dBFS assumes full-scale = 1.0
    current_dbfs = 20*log10(rms + eps);
    gain_db = target_dbfs - current_dbfs;
    sig = sig * db2mag(gain_db);
end

function x = mild_channel_filter(x, fs)
    % Mild coloration to emulate mic / codec response.
    % Uses simple 1st/2nd order IIR to avoid requiring extra toolboxes.

    % Random gentle highpass (30-120 Hz)
    if rand < 0.6
        fc = 30 + 90*rand;
        try
            [b,a] = butter(1, fc/(fs/2), 'high');
            x = filter(b,a,x);
        catch
        end
    end

    % Random gentle lowpass (6.5-7.9 kHz)
    if rand < 0.6
        fc = 6500 + 1400*rand;
        try
            [b,a] = butter(2, fc/(fs/2), 'low');
            x = filter(b,a,x);
        catch
        end
    end

    % Small tilt / EQ via shelving-like effect (approx using 1st order)
    if rand < 0.4
        % boost/cut highs slightly
        g = -2 + 4*rand; % dB
        alpha = 0.95; % closer to 1 = more high emphasis
        y = filter([1 -alpha], 1, x);
        x = x + db2mag(g) * y;
    end
end

function [mix_mc, target_mc, interf_mc, noise_mc, meta] = create_mixture_no_reverb_aug( ...
    target, interf, noise_src, theta_target, theta_interf, fs, sir_db, snr_db, c, d, micPos, dynamic_noise_prob)

    L = min([length(target), length(interf)]);
    if ~isempty(noise_src)
        L = min(L, length(noise_src));
    end

    target = target(1:L);
    interf = interf(1:L);
    if ~isempty(noise_src)
        noise_src = noise_src(1:L);
    end

    target_mc = make_multichannel(target, theta_target, fs, c, micPos);
    interf_mc = make_multichannel(interf, theta_interf, fs, c, micPos);

    % Set SIR by scaling the total interferer energy vs target energy
    Pt = mean(target_mc(:).^2) + 1e-12;
    Pi = mean(interf_mc(:).^2) + 1e-12;
    desired_ratio = 10^(-sir_db/10); % Pi/Pt
    interf_mc = interf_mc * sqrt((Pt * desired_ratio) / Pi);

    mix_clean = target_mc + interf_mc;

    % Build additive noise (real noise if provided, else white)
    if ~isempty(noise_src) && any(noise_src)
        noise_base_mc = make_multichannel(noise_src, -theta_interf, fs, c, micPos);
    else
        noise_base_mc = randn(size(mix_clean));
    end

    % Optionally make noise nonstationary via a smooth time-varying envelope
    if rand < dynamic_noise_prob
        noise_base_mc = apply_smooth_gain_envelope(noise_base_mc);
    end

    % Scale noise to achieve target SNR where signal = (target+interf)
    Ps = mean(mix_clean(:).^2) + 1e-12;
    Pn = mean(noise_base_mc(:).^2) + 1e-12;
    desired_noise_pow = Ps / (10^(snr_db/10));
    noise_mc = noise_base_mc * sqrt(desired_noise_pow / Pn);

    mix_mc = mix_clean + noise_mc;

    meta = struct();
    meta.Pt = Pt;
    meta.Pi = mean(interf_mc(:).^2);
    meta.Ps = Ps;
    meta.Pn = mean(noise_mc(:).^2);
end

function mc = make_multichannel(sig, theta_deg, fs, c, micPos)
    % Fractional delay (plane wave) per mic
    apply_delay = @(x, tau) interp1((0:length(x)-1)', x, (0:length(x)-1)' - tau*fs, 'linear', 0);

    tau1 = micPos(1)*sin(deg2rad(theta_deg))/c;
    tau2 = micPos(2)*sin(deg2rad(theta_deg))/c;

    mc = [apply_delay(sig, tau1), apply_delay(sig, tau2)];
end

function x = apply_smooth_gain_envelope(x)
    % Smooth random envelope in dB, converted to linear gain.
    L = size(x,1);
    nKnots = 5;
    knot_idx = round(linspace(1, L, nKnots));
    knot_db = -3 + 6*randn(1, nKnots); % variability
    env_db = interp1(knot_idx, knot_db, 1:L, 'pchip');
    g = db2mag(env_db(:));
    x = x .* g;
end

function y = soft_clip_tanh(x, drive)
    y = tanh(drive * x) ./ tanh(drive);
end

function [mix_mc, target_mc, interf_mc, noise_mc] = peak_normalize_all(mix_mc, target_mc, interf_mc, noise_mc, peak_target)
    p = max(abs(mix_mc(:)));
    if p > 0
        s = peak_target / p;
        mix_mc = mix_mc * s;
        target_mc = target_mc * s;
        interf_mc = interf_mc * s;
        noise_mc = noise_mc * s;
    end
end

function y = peak_normalize(y, peak_target)
    p = max(abs(y(:)));
    if p > 0
        y = (peak_target / p) * y;
    end
end

function out = apply_mvdr_pipeline(x_mc, cfg)
    out = [];

    try
        X = stft_multichannel(x_mc, cfg.stft.window, cfg.stft.hop, cfg.stft.nfft);
        [numFreqs, numFrames, numMics] = size(X);
        freqs = (0:numFreqs-1)' * cfg.fs / cfg.stft.nfft;

        Rxx = init_covariance(numFreqs, numMics);
        for n = 1:numFrames
            X_frame = squeeze(X(:,n,:));
            Rxx = update_covariance(Rxx, X_frame, cfg.mvdr_alpha);
        end

        dvec = compute_steering_vector(cfg.theta_target_mvdr, freqs, cfg.mic_pos, cfg.c);
        W = compute_mvdr_weights(Rxx, dvec, cfg.mvdr_delta);
        Y = apply_mvdr(X, W);

        L = round(cfg.duration_sec * cfg.fs);
        y = istft_single_channel(Y, cfg.stft.window, cfg.stft.hop, cfg.stft.nfft, L);
        out = real(y);

    catch ME
        fprintf('MVDR failed: %s\n', ME.message);
    end
end

function tag = fmt_db_tag(x_db)
    % e.g., -2.5 -> m2p5, 5 -> p5
    s = sprintf('%.1f', x_db);
    s = strrep(s, '-', 'm');
    s = strrep(s, '+', 'p');
    s = strrep(s, '.', 'p');
    if startsWith(s, 'm')
        tag = s;
    else
        tag = ['p' s];
    end
end

function m = db2mag(db)
    % Local fallback to avoid toolbox dependency.
    m = 10.^(db./20);
end
