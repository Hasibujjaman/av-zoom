%% Generate Competition Mixtures (RANDOMIZED male, interferences selection)
% Uses generate_reverberant_mixture_custom (shoebox ISM) instead of create_competition_mixture.

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

clear; clc; close all;

%% Randomization Seed
% rng(2);
rng('shuffle');

%% Process N male files
n_to_process = 2;

%% apply MVDR?
apply_mvdr_to_mixture = true;

%% Parameters (as per competition requirements)
fs = 16000;                 % Sampling rate (Hz)
c = 340;                    % Speed of sound (m/s)
d = 0.08;                   % Microphone spacing (m)
RT60_target = 0.5;          % Reverberation time (s)
SIR_target = 0;             % Signal-to-Interference Ratio (dB)
SNR_target = 5;             % Signal-to-Noise Ratio (dB)
duration_sec = 3;
Lsig = duration_sec * fs;

%% -------- MVDR Params --------
theta_target_test = 0;
mic_pos = [-d/2; d/2];
% STFT params
N = 256;
hop = 128;
nfft = 512;
window = sqrt(hann(N,'periodic'));

fprintf('=== Competition Mixture Generation (Custom ISM Reverb) ===\n');
fprintf('Target Parameters:\n');
fprintf('  RT60: %.1f s\n', RT60_target);
fprintf('  SIR: %d dB\n', SIR_target);
fprintf('  SNR: %d dB\n', SNR_target);
fprintf('  Fs: %d Hz\n', fs);
fprintf('  c: %.0f m/s\n', c);
fprintf('  d: %.2f m\n\n', d);

%% Main processing
male_folder = '../Male_clean';
female_folder = '../Female';
music_folder = '../Music';
noise_folder = '../Noise';

male_files = dir(fullfile(male_folder, '*.flac'));
female_files = dir(fullfile(female_folder, '*.flac'));
music_files = dir(fullfile(music_folder, '*.flac'));
noise_files = dir(fullfile(noise_folder, '*.flac'));

fprintf('Found files:\n');
fprintf('  Male: %d\n', length(male_files));
fprintf('  Female: %d\n', length(female_files));
fprintf('  Music: %d\n', length(music_files));
fprintf('  Noise: %d\n\n', length(noise_files));

success_count = 0;

for i = 1:min(n_to_process, length(male_files))
    male_idx = randi(length(male_files));
    fprintf('=== Processing Male File %d/%d ===\n', male_idx, min(n_to_process, length(male_files)));

    male_file = male_files(male_idx);
    male_path = fullfile(male_folder, male_file.name);
    target_sig = load_audio_file(male_path, fs, duration_sec);

    if all(target_sig == 0)
        fprintf('Skipping %s (empty or error)\n', male_file.name);
        continue;
    end

    fprintf('Target: %s (%.1f dB)\n', male_file.name, 10*log10(mean(target_sig.^2) + eps));
    [~, male_base_name, ~] = fileparts(male_file.name);

    female_idx = randi(length(female_files));
    music_idx  = randi(length(music_files));
    noise_idx  = randi(length(noise_files));

    female_sig = load_audio_file(fullfile(female_folder, female_files(female_idx).name), fs, duration_sec);
    music_sig  = load_audio_file(fullfile(music_folder, music_files(music_idx).name), fs, duration_sec);
    noise_sig  = load_audio_file(fullfile(noise_folder, noise_files(noise_idx).name), fs, duration_sec);

    fprintf('Interferences:\n');
    fprintf('  Female: %s\n', female_files(female_idx).name);
    fprintf('  Music: %s\n', music_files(music_idx).name);
    fprintf('  Noise: %s\n\n', noise_files(noise_idx).name);

    P_female = mean(female_sig.^2) + eps;
    P_music = mean(music_sig.^2) + eps;
    P_noise_base = mean(noise_sig.^2) + eps;

    % Pre-scale auxiliary components so combos are balanced (same as your current script intent)
    music_scaled = music_sig * sqrt(P_female / P_music);
    noise_scaled = noise_sig * sqrt(P_female / P_noise_base);
    noise_scaled2 = noise_sig * sqrt(P_music / P_noise_base);

    %% Combination A: Female speech only
    fprintf('--- Combination A (Female only) ---\n');
    interf_sig_A = female_sig;
    [mixture_A, target_mc_A, interf_mc_A, noise_mc_A, rt60_A] = create_competition_mixture_customRIR(...
        target_sig, interf_sig_A, fs, c, RT60_target, SIR_target, SNR_target);
    [SIR_A, SNR_A, max_amp_A] = verify_from_components(mixture_A, target_mc_A, interf_mc_A, noise_mc_A);
    display_verification_results(SIR_A, SNR_A, rt60_A, max_amp_A, RT60_target, SIR_target, SNR_target);

    output_name_A = sprintf('%s_A_female_only.wav', male_base_name);
    mixture_A = apply_mvdr_if_enabled(mixture_A, apply_mvdr_to_mixture, window, hop, nfft, Lsig, fs, mic_pos, c, theta_target_test);
    audiowrite(output_name_A, mixture_A, fs);
    fprintf('Saved: %s\n\n', output_name_A);

    %% Combination B: Female + Music
    fprintf('--- Combination B (Female + Music) ---\n');
    interf_sig_B = female_sig + music_scaled;
    [mixture_B, target_mc_B, interf_mc_B, noise_mc_B, rt60_B] = create_competition_mixture_customRIR(...
        target_sig, interf_sig_B, fs, c, RT60_target, SIR_target, SNR_target);
    [SIR_B, SNR_B, max_amp_B] = verify_from_components(mixture_B, target_mc_B, interf_mc_B, noise_mc_B);
    display_verification_results(SIR_B, SNR_B, rt60_B, max_amp_B, RT60_target, SIR_target, SNR_target);

    output_name_B = sprintf('%s_B_female_music.wav', male_base_name);
    mixture_B = apply_mvdr_if_enabled(mixture_B, apply_mvdr_to_mixture, window, hop, nfft, Lsig, fs, mic_pos, c, theta_target_test);
    audiowrite(output_name_B, mixture_B, fs);
    fprintf('Saved: %s\n\n', output_name_B);

    %% Combination C: Female + Noise
    fprintf('--- Combination C (Female + Noise) ---\n');
    interf_sig_C = female_sig + noise_scaled;
    [mixture_C, target_mc_C, interf_mc_C, noise_mc_C, rt60_C] = create_competition_mixture_customRIR(...
        target_sig, interf_sig_C, fs, c, RT60_target, SIR_target, SNR_target);
    [SIR_C, SNR_C, max_amp_C] = verify_from_components(mixture_C, target_mc_C, interf_mc_C, noise_mc_C);
    display_verification_results(SIR_C, SNR_C, rt60_C, max_amp_C, RT60_target, SIR_target, SNR_target);

    output_name_C = sprintf('%s_C_female_noise.wav', male_base_name);
    mixture_C = apply_mvdr_if_enabled(mixture_C, apply_mvdr_to_mixture, window, hop, nfft, Lsig, fs, mic_pos, c, theta_target_test);
    audiowrite(output_name_C, mixture_C, fs);
    fprintf('Saved: %s\n\n', output_name_C);

    %% Combination D: Music + Noise
    fprintf('--- Combination D (Music + Noise) ---\n');
    interf_sig_D = music_sig + noise_scaled2;
    [mixture_D, target_mc_D, interf_mc_D, noise_mc_D, rt60_D] = create_competition_mixture_customRIR(...
        target_sig, interf_sig_D, fs, c, RT60_target, SIR_target, SNR_target);
    [SIR_D, SNR_D, max_amp_D] = verify_from_components(mixture_D, target_mc_D, interf_mc_D, noise_mc_D);
    display_verification_results(SIR_D, SNR_D, rt60_D, max_amp_D, RT60_target, SIR_target, SNR_target);

    output_name_D = sprintf('%s_D_music_noise.wav', male_base_name);
    mixture_D = apply_mvdr_if_enabled(mixture_D, apply_mvdr_to_mixture, window, hop, nfft, Lsig, fs, mic_pos, c, theta_target_test);
    audiowrite(output_name_D, mixture_D, fs);
    fprintf('Saved: %s\n\n', output_name_D);

    %% Combination E: Female + Music + Noise
    fprintf('--- Combination E (Female + Music + Noise) ---\n');
    interf_sig_E = female_sig + music_scaled + noise_scaled;
    [mixture_E, target_mc_E, interf_mc_E, noise_mc_E, rt60_E] = create_competition_mixture_customRIR(...
        target_sig, interf_sig_E, fs, c, RT60_target, SIR_target, SNR_target);
    [SIR_E, SNR_E, max_amp_E] = verify_from_components(mixture_E, target_mc_E, interf_mc_E, noise_mc_E);
    display_verification_results(SIR_E, SNR_E, rt60_E, max_amp_E, RT60_target, SIR_target, SNR_target);

    output_name_E = sprintf('%s_E_female_music_noise.wav', male_base_name);
    mixture_E = apply_mvdr_if_enabled(mixture_E, apply_mvdr_to_mixture, window, hop, nfft, Lsig, fs, mic_pos, c, theta_target_test);
    audiowrite(output_name_E, mixture_E, fs);
    fprintf('Saved: %s\n\n', output_name_E);

    success_count = success_count + 1;
end

fprintf('\n=== GENERATION COMPLETE ===\n');
fprintf('Successfully processed %d male files.\n', success_count);
fprintf('Generated %d mixtures in total.\n', success_count * 5);

%% -------- Local functions --------

function [mixture, target_mc, interf_mc, noise_mc, rt60_avg] = create_competition_mixture_customRIR(target_sig, interf_sig, fs, c, RT60_target, SIR_dB, SNR_dB)
    opts = struct();
    opts.fs = fs;
    opts.soundSpeed = c;
    opts.RT60_target = RT60_target;
    opts.SIR_dB = SIR_dB;
    opts.SNR_dB = SNR_dB;

    opts.targetSignal = target_sig(:);
    opts.interferenceSignal = interf_sig(:);

    out = generate_reverberant_mixture_custom(opts);

    mixture = out.signals.stereo_out;
    target_mc = out.signals.target_mc;
    interf_mc = out.signals.interf_mc;
    noise_mc = out.signals.noise_mc;
    rt60_avg = out.rt60.average;

    L = 3 * fs;
    mixture = fix_length(mixture, L);
    target_mc = fix_length(target_mc, L);
    interf_mc = fix_length(interf_mc, L);
    noise_mc = fix_length(noise_mc, L);
end

function y = fix_length(x, L)
    if isempty(x)
        y = zeros(L, 1);
        return;
    end
    if size(x, 1) > L
        y = x(1:L, :);
    elseif size(x, 1) < L
        y = [x; zeros(L - size(x, 1), size(x, 2))];
    else
        y = x;
    end
end

function [SIR_actual, SNR_actual, max_amp] = verify_from_components(mixture, target_mc, interf_mc, noise_mc)
    P_target = mean(target_mc(:).^2) + eps;
    P_interf = mean(interf_mc(:).^2) + eps;
    SIR_actual = 10 * log10(P_target / P_interf);

    mixture_clean = target_mc + interf_mc;
    P_signal_clean = mean(mixture_clean(:).^2) + eps;
    P_noise = mean(noise_mc(:).^2) + eps;
    SNR_actual = 10 * log10(P_signal_clean / P_noise);

    max_amp = max(abs(mixture(:)));
end

function display_verification_results(SIR_actual, SNR_actual, rt60_est, max_amp, RT60_target, SIR_target, SNR_target)
    fprintf('SIR: %.2f dB (target: %.0f dB) ', SIR_actual, SIR_target);
    if abs(SIR_actual - SIR_target) < 0.1
        fprintf('✓\n');
    else
        fprintf('✗ (error: %.2f dB)\n', abs(SIR_actual - SIR_target));
    end

    fprintf('SNR: %.2f dB (target: %.0f dB) ', SNR_actual, SNR_target);
    if abs(SNR_actual - SNR_target) < 0.1
        fprintf('✓\n');
    else
        fprintf('✗ (error: %.2f dB)\n', abs(SNR_actual - SNR_target));
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

function mixture_out = apply_mvdr_if_enabled(mixture_in, enabled, window, hop, nfft, L, fs, mic_pos, c, theta_target)
    if ~enabled
        mixture_out = mixture_in;
        return;
    end

    x = mixture_in;
    X = stft_multichannel(x, window, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    freqs = (0:numFreqs-1)' * fs / nfft;

    Rxx = init_covariance(numFreqs, numMics);
    alpha = 0.98;
    for n = 1:numFrames
        X_frame = squeeze(X(:, n, :));
        Rxx = update_covariance(Rxx, X_frame, alpha);
    end

    dvec = compute_steering_vector(theta_target, freqs, mic_pos, c);
    delta = 1e-3;
    W = compute_mvdr_weights(Rxx, dvec, delta);
    Y = apply_mvdr(X, W);

    y_mvdr = istft_single_channel(Y, window, hop, nfft, L);
    mixture_out = real(y_mvdr);
end

function sig = load_audio_file(filepath, fs, duration)
    try
        [sig, file_fs] = audioread(filepath);
        if file_fs ~= fs
            sig = resample(sig, fs, file_fs);
        end
        if size(sig, 2) > 1
            sig = mean(sig, 2);
        end

        target_samples = fs * duration;
        if length(sig) > target_samples
            sig = sig(1:target_samples);
        elseif length(sig) < target_samples
            sig = [sig; zeros(target_samples - length(sig), 1)];
        end

        sig = sig - mean(sig);
    catch ME
        fprintf('Error loading %s: %s\n', filepath, ME.message);
        sig = zeros(fs * duration, 1);
    end
end








function out = generate_reverberant_mixture_custom(opts)
%GENERATE_REVERBERANT_MIXTURE_CUSTOM Generate a 2-mic reverberant mixture using a custom shoebox ISM.
%
%   out = generate_reverberant_mixture_custom(opts)
%
% Inputs
%   opts (struct) with fields (all optional; defaults match your setup):
%     .fs (default 16000)
%     .soundSpeed (default 340)
%     .roomDimensions (1x3, default [4.9 4.9 4.9])
%     .rx (Nx3 mic positions, default two-mic array at center)
%     .source1 (1x3 target position)
%     .source2 (1x3 interferer position)
%     .RT60_target (default 0.5)
%     .SIR_dB (default 0)
%     .SNR_dB (default 5)
%     .imageSourceOrder (default 15; may be auto-increased)
%     .autoIncreaseOrder (default true)
%     .ms0 (initial scattering for tuning, default 0.05)
%     .minIrDurationFactor (default 1.1)  % prefer IR >= factor*RT60
%     .absorptionCandidates (vector)
%     .scatteringCandidates (vector)
%     .fineSpan (default 0.03)
%     .fineStep (default 0.0025)
%     .targetSignal / .interferenceSignal (vectors) OR
%     .targetPath / .interferencePath (strings)
%     .normalizeOutput (default true)
%     .headroom (default 0.95)
%     .save (struct):
%         .enable (default false)
%         .outDir (default pwd)
%         .prefix (default "")
%         .saveMat (default true)
%         .saveWav (default true)
%
% Output
%   out struct containing:
%     .rir.source1 (NxL), .rir.source2 (NxL)
%     .params (chosen coefficients, order, etc)
%     .rt60 (per-path and average)
%     .signals (mic received/noisy/normalized, stereo)
%
% Notes
% - Uses image-source method only.
% - Uses a local ISM implementation (no Audio Toolbox dependency).

    if nargin < 1
        opts = struct();
    end

    opts = applyDefaults(opts);

    % Load/resample input signals
    [target, fs_target] = loadSignal(opts.targetSignal, opts.targetPath);
    [interf, fs_interf] = loadSignal(opts.interferenceSignal, opts.interferencePath);

    if isempty(fs_target)
        fs_target = opts.fs;
    end
    if isempty(fs_interf)
        fs_interf = opts.fs;
    end

    if fs_target ~= opts.fs
        target = resample(target, opts.fs, fs_target);
    end
    if fs_interf ~= opts.fs
        interf = resample(interf, opts.fs, fs_interf);
    end

    target = target(:);
    interf = interf(:);

    % Auto-increase order so the maximum delay can plausibly exceed RT60.
    if opts.autoIncreaseOrder
        minOrder = ceil((opts.RT60_target * opts.soundSpeed) / (2 * min(opts.roomDimensions))) + 2;
        if opts.imageSourceOrder < minOrder
            opts.imageSourceOrder = minOrder;
        end
    end

    % Sabine absorption as reference (printed/returned, not forced)
    V = prod(opts.roomDimensions);
    S = 2*(opts.roomDimensions(1)*opts.roomDimensions(2) + ...
           opts.roomDimensions(1)*opts.roomDimensions(3) + ...
           opts.roomDimensions(2)*opts.roomDimensions(3));
    alpha_sabine = 0.161 * V / (opts.RT60_target * S);

    % Tune uniform absorption/scattering
    min_ir_duration_s = opts.minIrDurationFactor * opts.RT60_target;

    best = struct('alpha', alpha_sabine, 'ms', opts.ms0, 'rt60', NaN, 'err', Inf, 'ir_dur', 0);

    for si = 1:numel(opts.scatteringCandidates)
        s = opts.scatteringCandidates(si);
        for ai = 1:numel(opts.absorptionCandidates)
            a = opts.absorptionCandidates(ai);
            ir_test = ismShoeboxRIR(opts.roomDimensions, opts.source1, opts.rx(1,:), ...
                opts.fs, opts.soundSpeed, opts.imageSourceOrder, a*ones(6,1), s*ones(6,1));

            ir_dur = size(ir_test,2) / opts.fs;
            rt = estimate_rt60(ir_test(1,:), opts.fs);
            err = abs(rt - opts.RT60_target);

            if ir_dur >= min_ir_duration_s
                if ~isnan(rt) && err < best.err
                    best = struct('alpha', a, 'ms', s, 'rt60', rt, 'err', err, 'ir_dur', ir_dur);
                end
            else
                if best.ir_dur < min_ir_duration_s && ir_dur > best.ir_dur
                    best = struct('alpha', a, 'ms', s, 'rt60', rt, 'err', err, 'ir_dur', ir_dur);
                end
            end

            if best.ir_dur >= min_ir_duration_s && best.err <= 0.03
                break;
            end
        end
        if best.ir_dur >= min_ir_duration_s && best.err <= 0.03
            break;
        end
    end

    % Fine search around best.alpha
    if ~isnan(best.rt60)
        fine_a = max(0, best.alpha - opts.fineSpan) : opts.fineStep : min(0.95, best.alpha + opts.fineSpan);
        for ai = 1:numel(fine_a)
            a = fine_a(ai);
            ir_test = ismShoeboxRIR(opts.roomDimensions, opts.source1, opts.rx(1,:), ...
                opts.fs, opts.soundSpeed, opts.imageSourceOrder, a*ones(6,1), best.ms*ones(6,1));

            ir_dur = size(ir_test,2) / opts.fs;
            rt = estimate_rt60(ir_test(1,:), opts.fs);
            err = abs(rt - opts.RT60_target);

            if ir_dur >= min_ir_duration_s
                if ~isnan(rt) && err < best.err
                    best = struct('alpha', a, 'ms', best.ms, 'rt60', rt, 'err', err, 'ir_dur', ir_dur);
                end
            end

            if best.err <= 0.01
                break;
            end
        end
    end

    absorption = best.alpha;
    scattering = best.ms;
    reflection = sqrt(max(0, 1 - absorption));

    % Generate RIRs
    ir_source1 = ismShoeboxRIR(opts.roomDimensions, opts.source1, opts.rx, ...
        opts.fs, opts.soundSpeed, opts.imageSourceOrder, absorption*ones(6,1), scattering*ones(6,1));
    ir_source2 = ismShoeboxRIR(opts.roomDimensions, opts.source2, opts.rx, ...
        opts.fs, opts.soundSpeed, opts.imageSourceOrder, absorption*ones(6,1), scattering*ones(6,1));

    % Convolve + mix with SIR
    target_mic1 = conv(target, ir_source1(1,:).');
    target_mic2 = conv(target, ir_source1(2,:).');
    interf_mic1 = conv(interf, ir_source2(1,:).');
    interf_mic2 = conv(interf, ir_source2(2,:).');

    min_len = min([numel(target_mic1), numel(interf_mic1), numel(target_mic2), numel(interf_mic2)]);
    target_mic1 = target_mic1(1:min_len);
    target_mic2 = target_mic2(1:min_len);
    interf_mic1 = interf_mic1(1:min_len);
    interf_mic2 = interf_mic2(1:min_len);

    % Scale interferer to achieve desired SIR per mic
    sir_lin = 10^(opts.SIR_dB/10);
    % SIR = P_target / P_interf  =>  P_interf_desired = P_target / sir_lin
    P_t1 = mean(target_mic1.^2);
    P_i1 = mean(interf_mic1.^2);
    P_t2 = mean(target_mic2.^2);
    P_i2 = mean(interf_mic2.^2);

    scale1 = sqrt((P_t1 / sir_lin) / max(P_i1, eps));
    scale2 = sqrt((P_t2 / sir_lin) / max(P_i2, eps));

    interf_mic1 = interf_mic1 * scale1;
    interf_mic2 = interf_mic2 * scale2;

    mic1_received = target_mic1 + interf_mic1;
    mic2_received = target_mic2 + interf_mic2;

    target_mc = [target_mic1(:), target_mic2(:)];
    interf_mc = [interf_mic1(:), interf_mic2(:)];
    mixture_clean = target_mc + interf_mc;

    % Add sensor noise
    if isfinite(opts.SNR_dB)
        snr_lin = 10^(opts.SNR_dB/10);
        P_s1 = mean(mic1_received.^2);
        P_s2 = mean(mic2_received.^2);
        P_n1 = P_s1 / snr_lin;
        P_n2 = P_s2 / snr_lin;
        noise1 = sqrt(P_n1) * randn(size(mic1_received));
        noise2 = sqrt(P_n2) * randn(size(mic2_received));
        mic1_noisy = mic1_received + noise1;
        mic2_noisy = mic2_received + noise2;
    else
        noise1 = zeros(size(mic1_received));
        noise2 = zeros(size(mic2_received));
        mic1_noisy = mic1_received;
        mic2_noisy = mic2_received;
    end

    noise_mc = [noise1(:), noise2(:)];

    % RT60 estimates
    rt60_est = zeros(4,1);
    rt60_est(1) = estimate_rt60(ir_source1(1,:), opts.fs);
    rt60_est(2) = estimate_rt60(ir_source1(2,:), opts.fs);
    rt60_est(3) = estimate_rt60(ir_source2(1,:), opts.fs);
    rt60_est(4) = estimate_rt60(ir_source2(2,:), opts.fs);

    % Normalize
    if opts.normalizeOutput
        max_amp = max([max(abs(mic1_noisy(:))), max(abs(mic2_noisy(:)))]);
        if max_amp > 0
            normScale = (opts.headroom / max_amp);
            mic1_out = mic1_noisy * normScale;
            mic2_out = mic2_noisy * normScale;

            target_mc = target_mc * normScale;
            interf_mc = interf_mc * normScale;
            noise_mc = noise_mc * normScale;
            mixture_clean = mixture_clean * normScale;
        else
            mic1_out = mic1_noisy;
            mic2_out = mic2_noisy;
        end
    else
        max_amp = max([max(abs(mic1_noisy(:))), max(abs(mic2_noisy(:)))]);
        mic1_out = mic1_noisy;
        mic2_out = mic2_noisy;
    end

    stereo = [mic1_out(:), mic2_out(:)];

    out = struct();
    out.rir = struct('source1', ir_source1, 'source2', ir_source2);
    out.params = struct(...
        'fs', opts.fs, ...
        'soundSpeed', opts.soundSpeed, ...
        'roomDimensions', opts.roomDimensions, ...
        'rx', opts.rx, ...
        'source1', opts.source1, ...
        'source2', opts.source2, ...
        'RT60_target', opts.RT60_target, ...
        'alpha_sabine', alpha_sabine, ...
        'absorption', absorption, ...
        'reflection', reflection, ...
        'scattering', scattering, ...
        'imageSourceOrder', opts.imageSourceOrder, ...
        'min_ir_duration_s', min_ir_duration_s, ...
        'best_rt60_test', best.rt60, ...
        'best_err', best.err);

    out.rt60 = struct(...
        'source1_mic1', rt60_est(1), ...
        'source1_mic2', rt60_est(2), ...
        'source2_mic1', rt60_est(3), ...
        'source2_mic2', rt60_est(4), ...
        'average', mean(rt60_est));

    out.signals = struct(...
        'mic1_received', mic1_received, ...
        'mic2_received', mic2_received, ...
        'mic1_noisy', mic1_noisy, ...
        'mic2_noisy', mic2_noisy, ...
        'mic1_out', mic1_out, ...
        'mic2_out', mic2_out, ...
        'target_mc', target_mc, ...
        'interf_mc', interf_mc, ...
        'noise_mc', noise_mc, ...
        'mixture_clean', mixture_clean, ...
        'stereo_out', stereo, ...
        'max_amp_before_norm', max_amp);

    out.info = struct('rir_length_samples', size(ir_source1,2), 'rir_length_seconds', size(ir_source1,2)/opts.fs);

    if opts.save.enable
        saveOutputs(opts, out);
    end
end

function opts = applyDefaults(opts)
    opts = setDefault(opts, 'fs', 16000);
    opts = setDefault(opts, 'soundSpeed', 343);
    opts = setDefault(opts, 'roomDimensions', [4.9 4.9 4.9]);

    defaultRx = [2.41, 2.45, 1.5; 2.49, 2.45, 1.5];
    opts = setDefault(opts, 'rx', defaultRx);

    opts = setDefault(opts, 'source1', [2.45, 3.45, 1.5]);
    opts = setDefault(opts, 'source2', [3.22, 3.06, 1.5]);

    opts = setDefault(opts, 'RT60_target', 0.5);
    opts = setDefault(opts, 'SIR_dB', 0);
    opts = setDefault(opts, 'SNR_dB', 5);

    opts = setDefault(opts, 'imageSourceOrder', 15);
    opts = setDefault(opts, 'autoIncreaseOrder', true);
    opts = setDefault(opts, 'ms0', 0.05);
    opts = setDefault(opts, 'minIrDurationFactor', 1.1);

    opts = setDefault(opts, 'absorptionCandidates', [0.0 0.0005 0.001 0.002 0.005 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.12 0.14 0.16 0.20 0.24 0.28]);
    opts = setDefault(opts, 'scatteringCandidates', [0.0 0.01 0.05 0.2 0.5]);
    opts = setDefault(opts, 'fineSpan', 0.03);
    opts = setDefault(opts, 'fineStep', 0.0025);

    opts = setDefault(opts, 'normalizeOutput', true);
    opts = setDefault(opts, 'headroom', 0.95);

    % Signals/paths: allow either (signal or path). Paths default to empty.
    opts = setDefault(opts, 'targetSignal', []);
    opts = setDefault(opts, 'interferenceSignal', []);
    opts = setDefault(opts, 'targetPath', "");
    opts = setDefault(opts, 'interferencePath', "");

    if isempty(opts.targetSignal) && strlength(string(opts.targetPath)) == 0
        error('opts.targetSignal or opts.targetPath is required.');
    end
    if isempty(opts.interferenceSignal) && strlength(string(opts.interferencePath)) == 0
        error('opts.interferenceSignal or opts.interferencePath is required.');
    end

    if ~isfield(opts, 'save') || ~isstruct(opts.save)
        opts.save = struct();
    end
    opts.save = setDefault(opts.save, 'enable', false);
    opts.save = setDefault(opts.save, 'outDir', pwd);
    opts.save = setDefault(opts.save, 'prefix', "");
    opts.save = setDefault(opts.save, 'saveMat', true);
    opts.save = setDefault(opts.save, 'saveWav', true);
end

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end

function [x, fs] = loadSignal(xIn, pathIn)
    if ~isempty(xIn)
        x = xIn;
        fs = [];
        return;
    end
    [x, fs] = audioread(char(pathIn));
end

function saveOutputs(opts, out)
    outDir = char(opts.save.outDir);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    prefix = char(opts.save.prefix);

    if opts.save.saveWav
        audiowrite(fullfile(outDir, [prefix 'mic1_received.wav']), out.signals.mic1_out(:), opts.fs);
        audiowrite(fullfile(outDir, [prefix 'mic2_received.wav']), out.signals.mic2_out(:), opts.fs);
        audiowrite(fullfile(outDir, [prefix 'stereo_output.wav']), out.signals.stereo_out, opts.fs);
    end

    if opts.save.saveMat
        rir = out.rir; %#ok<NASGU>
        params = out.params; %#ok<NASGU>
        rt60 = out.rt60; %#ok<NASGU>
        save(fullfile(outDir, [prefix 'room_impulse_responses.mat']), 'rir', 'params', 'rt60');
    end
end

function rt60 = estimate_rt60(ir, fs)
    ir = ir(:);

    energy = cumsum(ir(end:-1:1).^2);
    energy = energy(end:-1:1);
    energy = energy / max(energy + eps);
    energy_db = 10*log10(energy + eps);

    idx_5db = find(energy_db <= -5, 1, 'first');
    if isempty(idx_5db)
        rt60 = NaN;
        return;
    end

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

function ir = ismShoeboxRIR(roomDimensions, tx, rx, fs, c, order, materialAbsorption, materialScattering)
%ISM SHOEBOX RIR deterministic image-source method for shoebox rooms.
% Returns N-by-L (N microphones).

    L = double(roomDimensions(:)).';
    tx = double(tx);
    rx = double(rx);

    validateInRoom(tx, L, 'tx');
    for rxi = 1:size(rx,1)
        validateInRoom(rx(rxi,:), L, sprintf('rx(%d,:)', rxi));
    end

    alpha = materialAbsorption;
    if isscalar(alpha), alpha = repmat(alpha, 6, 1); else, alpha = alpha(:); end
    scat = materialScattering;
    if isscalar(scat), scat = repmat(scat, 6, 1); else, scat = scat(:); end

    % Effective per-surface reflection (simple model)
    refl = sqrt(max(0, 1 - alpha)) .* sqrt(max(0, 1 - scat));

    n = -order:order;
    [Nx, Ny, Nz] = ndgrid(n, n, n);
    keep = (abs(Nx) + abs(Ny) + abs(Nz)) <= order;
    Nx = Nx(keep);
    Ny = Ny(keep);
    Nz = Nz(keep);

    Ximg = ((-1) .^ Nx) .* tx(1) + 2 .* Nx .* L(1);
    Yimg = ((-1) .^ Ny) .* tx(2) + 2 .* Ny .* L(2);
    Zimg = ((-1) .^ Nz) .* tx(3) + 2 .* Nz .* L(3);

    ax = abs(Nx);
    ay = abs(Ny);
    az = abs(Nz);

    nLeft = zeros(size(Nx));
    nRight = zeros(size(Nx));
    nFront = zeros(size(Ny));
    nBack = zeros(size(Ny));
    nFloor = zeros(size(Nz));
    nCeil = zeros(size(Nz));

    pos = Nx >= 0;
    nRight(pos) = ceil(ax(pos) ./ 2);
    nLeft(pos)  = floor(ax(pos) ./ 2);
    nLeft(~pos) = ceil(ax(~pos) ./ 2);
    nRight(~pos)= floor(ax(~pos) ./ 2);

    pos = Ny >= 0;
    nFront(pos) = ceil(ay(pos) ./ 2);
    nBack(pos)  = floor(ay(pos) ./ 2);
    nBack(~pos) = ceil(ay(~pos) ./ 2);
    nFront(~pos)= floor(ay(~pos) ./ 2);

    pos = Nz >= 0;
    nCeil(pos)  = ceil(az(pos) ./ 2);
    nFloor(pos) = floor(az(pos) ./ 2);
    nFloor(~pos)= ceil(az(~pos) ./ 2);
    nCeil(~pos) = floor(az(~pos) ./ 2);

    % Surface order: floor(1), front(2), back(3), left(4), right(5), ceiling(6)
    reflGain = (refl(4) .^ nLeft) .* (refl(5) .^ nRight) .* ...
               (refl(2) .^ nFront) .* (refl(3) .^ nBack) .* ...
               (refl(1) .^ nFloor) .* (refl(6) .^ nCeil);

    numRx = size(rx,1);
    ir = zeros(numRx, 1);
    maxIdx2 = 1;

    for rxi = 1:numRx
        dx = Ximg - rx(rxi,1);
        dy = Yimg - rx(rxi,2);
        dz = Zimg - rx(rxi,3);
        dist = sqrt(dx.^2 + dy.^2 + dz.^2);

        valid = dist > 0;
        dist = dist(valid);
        g = (reflGain(valid) ./ dist);

        sFloat = (dist ./ c) .* fs;
        s0 = floor(sFloat);
        frac = sFloat - s0;
        idx1 = s0 + 1;
        idx2 = idx1 + 1;

        Lr = max(idx2);
        if Lr > size(ir,2)
            ir(:, end+1:Lr) = 0;
        end
        if Lr > maxIdx2
            maxIdx2 = Lr;
        end

        w1 = g .* (1 - frac);
        w2 = g .* frac;

        ir_r = accumarray(idx1, w1, [Lr, 1], @sum, 0) + ...
               accumarray(idx2, w2, [Lr, 1], @sum, 0);
        ir(rxi, 1:Lr) = ir(rxi, 1:Lr) + ir_r(:).';
    end

    ir = ir(:, 1:maxIdx2);
end

function validateInRoom(pos, L, name)
    if any(pos <= 0) || any(pos >= L)
        error('%s must be strictly inside the room bounds (0 < coord < dimension).', name);
    end
end
