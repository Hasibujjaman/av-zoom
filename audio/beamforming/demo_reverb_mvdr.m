%% ================================================================
%  demo_reverb_mvdr.m
%  Task 2 Demo: Reverberant MVDR + Zelinski Post-Filter
%
%  Competition parameters (SP Cup 2026):
%    Room:  4.9 × 4.9 × 4.9 m shoebox
%    RT60:  0.5 s (± 0.05 s tolerance)
%    Array: 2-mic ULA, d = 0.08 m, at room centre, height 1.5 m
%    Target:       broadside (90° cos convention), 1 m distance
%    Interference: ~40° off-axis, ~1 m distance
%    SIR = 0 dB,  SNR = 5 dB
%
%  Pipeline:
%    ISM RIR → convolve → SIR scale → AWGN → STFT → batch Rxx
%    → MVDR(δ=1e-3) → Zelinski PF(α=0.92,β=0.02) → ISTFT
%
%  Outputs:
%    Audio files for listening (6 WAVs in Test_output/reverb_demo/)
%    Metrics: SI-SDR, STOI, ViSQOL
%    RT60 verification (Schroeder backward integration)
%    Plots: RIR / EDC / waveforms / spectrograms
%
%  Date: 10 Feb 2026
% =================================================================

clear; clc; close all;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');

%% ===================== COMPETITION PARAMETERS =====================
fs       = 16000;
c        = 340;          % m/s
duration = 3;            % seconds
L        = fs * duration;

% ---- Room (competition spec) ----
room_dim = [4.9, 4.9, 4.9];   % metres

% ---- 2-mic ULA ----
d = 0.08;                        % mic spacing (m)
mic_pos_1d = [-d/2; d/2];       % for steering vector

% 3D positions (array along x-axis at room centre, height 1.5 m)
array_centre = [2.45, 2.45, 1.50];
mic_pos_3d   = [array_centre(1)-d/2, array_centre(2), array_centre(3);
                array_centre(1)+d/2, array_centre(2), array_centre(3)];

% ---- Source positions (competition spec) ----
source_target = [2.45, 3.45, 1.50];   % broadside, 1 m in front
source_interf = [3.22, 3.06, 1.50];   % ~40° off-axis

% ---- Compute actual DOA azimuths (cos convention) ----
dx_t = source_target(1) - array_centre(1);
dy_t = source_target(2) - array_centre(2);
az_target = atan2d(dy_t, dx_t);   % should be 90°

dx_i = source_interf(1) - array_centre(1);
dy_i = source_interf(2) - array_centre(2);
az_interf = atan2d(dy_i, dx_i);   % should be ~38°

dist_target = norm(source_target - array_centre);
dist_interf = norm(source_interf - array_centre);

fprintf('============================================\n');
fprintf('  COMPETITION GEOMETRY\n');
fprintf('============================================\n');
fprintf('Room: [%.1f × %.1f × %.1f] m\n', room_dim);
fprintf('Target:  [%.2f, %.2f, %.2f]  az = %.1f°  dist = %.2f m\n', ...
    source_target, az_target, dist_target);
fprintf('Interf:  [%.2f, %.2f, %.2f]  az = %.1f°  dist = %.2f m\n', ...
    source_interf, az_interf, dist_interf);
fprintf('Separation from broadside:  target = %.1f°,  interf = %.1f°\n', ...
    90 - az_target, 90 - az_interf);
fprintf('\n');

% ---- Conditions ----
RT60_target = 0.5;   % seconds
SIR_dB      = 0;
SNR_dB      = 5;

% ---- STFT / MVDR ----
N_fft_win = 512;
hop       = 128;
nfft      = 512;
win       = sqrt(hann(N_fft_win, 'periodic'));
delta     = 1e-3;    % diagonal loading

% ---- ISM ----
ism_order     = 15;   % auto-increased below if needed
RT60_tol      = 0.05; % ±0.05 s tolerance

% ---- Output ----
out_dir = fullfile('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming', ...
    'Test_output', 'reverb_demo');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ===================== LOAD AUDIO =====================
male_folder   = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Male';
female_folder = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/DATASET/Pre_MVDR/Female';

male_files   = [dir(fullfile(male_folder, '*.flac')); dir(fullfile(male_folder, '*.wav'))];
female_files = [dir(fullfile(female_folder, '*.flac')); dir(fullfile(female_folder, '*.wav'))];

rng(42);   % reproducible demo
m_idx = randi(length(male_files));
f_idx = randi(length(female_files));

target_sig = load_audio(fullfile(male_folder, male_files(m_idx).name), fs, duration);
interf_sig = load_audio(fullfile(female_folder, female_files(f_idx).name), fs, duration);

fprintf('Target file: %s\n', male_files(m_idx).name);
fprintf('Interf file: %s\n\n', female_files(f_idx).name);

%% ===================== GENERATE RIRs (ISM) =====================
fprintf('============================================\n');
fprintf('  ROOM IMPULSE RESPONSE GENERATION\n');
fprintf('============================================\n');
fprintf('Target RT60: %.2f s  (tolerance ± %.2f s)\n\n', RT60_target, RT60_tol);

% ---- Room acoustics ----
V      = prod(room_dim);
S_area = 2*(room_dim(1)*room_dim(2) + room_dim(1)*room_dim(3) + ...
             room_dim(2)*room_dim(3));

% Sabine absorption estimate
alpha_sabine = 0.161 * V / (RT60_target * S_area);
% Eyring absorption (more accurate for moderate absorption)
alpha_eyring = 1 - exp(-0.161 * V / (S_area * RT60_target));

fprintf('  Sabine  α = %.4f\n', alpha_sabine);
fprintf('  Eyring  α = %.4f\n', alpha_eyring);

% ---- ISM order: must be high enough for the full reverberant decay ----
%  The reflection coefficient per bounce is refl = sqrt(1-α).
%  For 60 dB decay: refl^N = 10^-6  →  N = 6 / -log10(refl)
%  The ISM order determines the MAX number of wall bounces (Manhattan dist).
%  For T30 measurement we need at least 35 dB of decay.
refl_eyring = sqrt(1 - alpha_eyring);
bounces_60dB = ceil(6 / (-log10(refl_eyring + eps)));   % bounces for 60 dB
bounces_35dB = ceil(3.5 / (-log10(refl_eyring + eps))); % bounces for T30
min_order = max(bounces_35dB + 5, 30);  % +5 safety margin, at least 30

fprintf('  Reflection per bounce (Eyring): %.4f\n', refl_eyring);
fprintf('  Bounces for 60 dB decay: %d   (for T30: %d)\n', bounces_60dB, bounces_35dB);

if ism_order < min_order
    fprintf('  Auto-increasing ISM order: %d → %d\n', ism_order, min_order);
    ism_order = min_order;
end
fprintf('  ISM order: %d\n', ism_order);

% ---- Binary search for absorption that gives RT60 = target ----
%  RT60 is monotonically decreasing with absorption.
%  Cubic rooms have degenerate modes → Eyring under-predicts the needed α.
%  Binary search is fast and robust.
fprintf('\n  Tuning absorption via binary search (target RT60 = %.2f s) ...\n', RT60_target);
t_tune = tic;

scatter_final = 0.0;   % no scattering — simpler, well-defined model
alpha_lo = 0.05;       % very reverberant end
alpha_hi = 0.80;       % very dry end
max_iter = 30;         % log2(0.75/0.001) ≈ 10, so 30 is plenty

% First verify that the bracket is valid
ir_lo = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
    fs, c, ism_order, alpha_lo*ones(6,1), scatter_final*ones(6,1));
rt60_lo = estimate_rt60(ir_lo(1,:), fs);

ir_hi = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
    fs, c, ism_order, alpha_hi*ones(6,1), scatter_final*ones(6,1));
rt60_hi = estimate_rt60(ir_hi(1,:), fs);

fprintf('    α = %.3f → RT60 = %.3f s\n', alpha_lo, rt60_lo);
fprintf('    α = %.3f → RT60 = %.3f s\n', alpha_hi, rt60_hi);

if isnan(rt60_lo) || isnan(rt60_hi)
    warning('Cannot bracket RT60 — using Eyring estimate');
    absorption_final = alpha_eyring;
    best_rt60 = NaN;
    best_err  = Inf;
else
    for iter = 1:max_iter
        alpha_mid = (alpha_lo + alpha_hi) / 2;
        ir_mid = ismShoeboxRIR(room_dim, source_target, mic_pos_3d(1,:), ...
            fs, c, ism_order, alpha_mid*ones(6,1), scatter_final*ones(6,1));
        rt60_mid = estimate_rt60(ir_mid(1,:), fs);

        if isnan(rt60_mid)
            % If NaN, move toward less absorption (longer RIR)
            alpha_hi = alpha_mid;
            continue;
        end

        if rt60_mid > RT60_target
            alpha_lo = alpha_mid;   % need more absorption
        else
            alpha_hi = alpha_mid;   % need less absorption
        end

        if abs(rt60_mid - RT60_target) < 0.005  % within 5 ms
            break;
        end
    end

    absorption_final = alpha_mid;
    best_rt60 = rt60_mid;
    best_err  = abs(rt60_mid - RT60_target);

    fprintf('    Binary search: %d iterations\n', iter);
end

refl_final = sqrt(max(0, 1 - absorption_final));

fprintf('  Result:  α = %.4f   scatter = %.3f   refl/bounce = %.4f\n', ...
    absorption_final, scatter_final, refl_final);
fprintf('  RT60 = %.3f s   (err = %.3f s)  %s\n', ...
    best_rt60, best_err, ternary(best_err <= RT60_tol, '✓', '✗'));
fprintf('  Eyring predicted α = %.4f,  ISM needed α = %.4f  (ratio = %.2f)\n', ...
    alpha_eyring, absorption_final, absorption_final / alpha_eyring);
fprintf('  Tuning time: %.1f s\n', toc(t_tune));

% ---- Check tolerance ----
if best_err > RT60_tol
    warning('RT60 error (%.3f s) exceeds ±%.2f s tolerance!', best_err, RT60_tol);
end

% ---- Generate final RIRs for all source→mic paths ----
fprintf('\nGenerating final RIRs (order = %d) ...\n', ism_order);
rir_target = ismShoeboxRIR(room_dim, source_target, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));
rir_interf = ismShoeboxRIR(room_dim, source_interf, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));

fprintf('  RIR length: %d samples (%.3f s)\n\n', size(rir_target,2), size(rir_target,2)/fs);

% ---- RT60 verification (Schroeder backward integration) ----
rt60_vals = zeros(4,1);
rt60_vals(1) = estimate_rt60(rir_target(1,:), fs);
rt60_vals(2) = estimate_rt60(rir_target(2,:), fs);
rt60_vals(3) = estimate_rt60(rir_interf(1,:), fs);
rt60_vals(4) = estimate_rt60(rir_interf(2,:), fs);

fprintf('RT60 Verification (Schroeder backward integration):\n');
fprintf('  Target  → Mic1:  %.3f s\n', rt60_vals(1));
fprintf('  Target  → Mic2:  %.3f s\n', rt60_vals(2));
fprintf('  Interf  → Mic1:  %.3f s\n', rt60_vals(3));
fprintf('  Interf  → Mic2:  %.3f s\n', rt60_vals(4));
fprintf('  Average:          %.3f s\n', mean(rt60_vals));
fprintf('  Competition spec: %.3f s   |error| = %.3f s  %s\n', ...
    RT60_target, abs(mean(rt60_vals) - RT60_target), ...
    ternary(abs(mean(rt60_vals) - RT60_target) <= RT60_tol, '✓', '✗'));

% ---- Direct-to-Reverberant Ratio from RIR ----
%  Direct path energy: RIR peak ± 1 ms (accounts for propagation delay)
%  Early reflections: everything up to 50 ms from RIR start
%  Late reverb: > 50 ms from RIR start
ir1 = rir_target(1,:);
[~, pk_rir] = max(abs(ir1));
direct_win_rir = max(1, pk_rir - round(0.001*fs)) : min(length(ir1), pk_rir + round(0.001*fs));
E_direct = sum(ir1(direct_win_rir).^2);
E_rest   = sum(ir1.^2) - E_direct;
DRR = 10*log10(E_direct / (E_rest + eps));

early_cutoff = round(0.050 * fs);  % 50 ms
E_early = sum(ir1(1:min(early_cutoff, length(ir1))).^2);
E_late  = sum(ir1(min(early_cutoff+1, length(ir1)):end).^2);
DRR_early = 10*log10(E_early / (E_late + eps));  % C50: early vs late

% Theoretical DRR (Sabine): DRR = Q*R / (16πr²) where R = Sα/(1-α)
R_room = S_area * absorption_final / (1 - absorption_final);
DRR_theory = 10*log10(R_room / (16*pi*dist_target^2));

fprintf('\n  DRR (direct vs all reflections):   %+.1f dB\n', DRR);
fprintf('  DRR+early (direct+early vs late):  %+.1f dB\n', DRR_early);
fprintf('  DRR theoretical (Sabine):          %+.1f dB\n', DRR_theory);
fprintf('  α = %.3f  →  refl/bounce = %.3f  →  R_room = %.1f m²\n\n', ...
    absorption_final, refl_final, R_room);

%% ===================== CREATE REVERBERANT MIXTURE =====================
fprintf('============================================\n');
fprintf('  REVERBERANT MIXTURE CREATION\n');
fprintf('============================================\n');

% Convolve each source with its RIRs
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
sir_lin = 10^(SIR_dB / 10);
interf_mc = interf_mc * sqrt(Pt / (Pi * sir_lin));

Pi_new = mean(interf_mc(:).^2);
sir_actual = 10*log10(Pt / Pi_new);
fprintf('SIR: target = %.1f dB   actual = %.2f dB\n', SIR_dB, sir_actual);

% ---- Clean mixture (before sensor noise) ----
mixture_clean = target_mc + interf_mc;

% ---- AWGN sensor noise ----
Ps = mean(mixture_clean(:).^2) + eps;
snr_lin = 10^(SNR_dB / 10);
noise_power = Ps / snr_lin;
noise_mc = sqrt(noise_power) * randn(size(mixture_clean));
mixture = mixture_clean + noise_mc;

snr_actual = 10*log10(Ps / mean(noise_mc(:).^2));
fprintf('SNR: target = %.1f dB   actual = %.2f dB\n', SNR_dB, snr_actual);

% ---- Peak-normalise (preserves relative levels) ----
peak = max(abs(mixture(:)));
if peak > 0
    sc = 0.99 / peak;
    mixture   = mixture   * sc;
    target_mc = target_mc * sc;
    interf_mc = interf_mc * sc;
    noise_mc  = noise_mc  * sc;
end
fprintf('Peak after normalisation: %.4f\n\n', max(abs(mixture(:))));

%% ===================== MVDR BEAMFORMING =====================
fprintf('============================================\n');
fprintf('  MVDR + ZELINSKI POST-FILTER\n');
fprintf('============================================\n');

% ---- STFT ----
X = stft_multichannel(mixture, win, hop, nfft);
[numFreqs, numFrames, numMics] = size(X);
fprintf('STFT: %d freqs × %d frames × %d mics\n', numFreqs, numFrames, numMics);

% ---- Steering vector for target DOA ----
freqs_vec   = (0:(nfft/2))' * fs / nfft;
dvec_target = compute_steering_vector(az_target, freqs_vec, mic_pos_1d, c);
fprintf('Steering vector: az = %.1f° (target)\n', az_target);

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

% ---- Apply MVDR ----
Y_mvdr = apply_mvdr(X, W);

% ---- Apply MVDR + Zelinski post-filter ----
Y_pf = mvdr_postfilter(X, W, dvec_target);

% ---- ISTFT ----
y_mvdr = real(istft_single_channel(Y_mvdr, win, hop, nfft, L));
y_mvdr = y_mvdr(1:L);

y_pf = real(istft_single_channel(Y_pf, win, hop, nfft, L));
y_pf = y_pf(1:L);

% ---- Trim 20 ms boundary artefacts ----
trim_samp = round(0.020 * fs);   % 320 samples
trim_idx  = (trim_samp + 1) : (L - trim_samp);

y_mvdr = y_mvdr(trim_idx);
y_pf   = y_pf(trim_idx);

% ---- References (trimmed to match) ----
x_mic      = mixture(trim_idx, 1);         % mic ch1 (reverb + interf + noise)
clean_dry  = target_sig(trim_idx);         % original dry speech (competition ref)
clean_rev  = target_mc(trim_idx, 1);       % reverberant target at mic1
mix_stereo = mixture(trim_idx, :);

% ---- Peak-normalise outputs ----
y_mvdr = safe_norm(y_mvdr);
y_pf   = safe_norm(y_pf);

% ---- Propagation delay compensation ----
%  The RIR introduces a propagation delay = dist/c.
%  SI-SDR is NOT shift-invariant — even a 3 ms shift can cause −20 dB loss.
%  We find the actual delay from the RIR peak and align all signals.
prop_delay_s   = dist_target / c;           % theoretical (s)
[~, peak_idx]  = max(abs(rir_target(1,:))); % actual from RIR (samples)
prop_delay_smp = peak_idx - 1;              % samples (0-indexed)

fprintf('Output length: %d samples (%.2f s)\n', length(y_mvdr), length(y_mvdr)/fs);
fprintf('Propagation delay: %.2f ms (%d samples) — compensating for SI-SDR\n\n', ...
    prop_delay_smp/fs*1000, prop_delay_smp);

% Shift reverberant/processed signals LEFT by prop_delay_smp,
% or equivalently, shift dry reference RIGHT (pad zeros at start, trim end).
% Easier: trim the first prop_delay_smp samples of reverb signals, and
%         the last prop_delay_smp samples of the dry reference.
if prop_delay_smp > 0 && prop_delay_smp < length(clean_dry) - 1000
    N_align = length(clean_dry) - prop_delay_smp;
    clean_dry_aligned = clean_dry(1:N_align);
    clean_rev_aligned = clean_rev(prop_delay_smp+1 : prop_delay_smp+N_align);
    x_mic_aligned     = x_mic(prop_delay_smp+1 : prop_delay_smp+N_align);
    y_mvdr_aligned    = y_mvdr(prop_delay_smp+1 : prop_delay_smp+N_align);
    y_pf_aligned      = y_pf(prop_delay_smp+1 : prop_delay_smp+N_align);
    has_alignment = true;
else
    clean_dry_aligned = clean_dry;
    clean_rev_aligned = clean_rev;
    x_mic_aligned     = x_mic;
    y_mvdr_aligned    = y_mvdr;
    y_pf_aligned      = y_pf;
    has_alignment = false;
    warning('Propagation delay (%d samples) out of range — skipping alignment', prop_delay_smp);
end

%% ===================== REVERB FLOOR DIAGNOSTIC =====================
fprintf('============================================\n');
fprintf('  REVERB FLOOR DIAGNOSTIC\n');
fprintf('============================================\n');

% --- SI-SDR reverb floor: with and without delay compensation ---
sisdr_floor_raw     = si_sdr(clean_rev, clean_dry);
sisdr_floor_aligned = si_sdr(clean_rev_aligned, clean_dry_aligned);

fprintf('\n  SI-SDR(reverb vs dry) WITHOUT delay comp:  %+.2f dB\n', sisdr_floor_raw);
fprintf('  SI-SDR(reverb vs dry) WITH    delay comp:  %+.2f dB\n', sisdr_floor_aligned);
fprintf('  → Delay compensation recovered %+.1f dB\n', sisdr_floor_aligned - sisdr_floor_raw);
fprintf('  → %+.2f dB is the physics floor (no beamformer can beat this)\n\n', sisdr_floor_aligned);
sisdr_reverb_floor = sisdr_floor_aligned;

% --- DRR from RIR (use direct path delay + small margin) ---
% DRR and C50 already computed in RIR section above
fprintf('  DRR (direct peak ± 1ms vs rest):  %+.1f dB\n', DRR);
fprintf('  C50 (early < 50ms vs late):        %+.1f dB\n', DRR_early);

try
    stoi_reverb_floor = stoi(clean_dry_aligned, clean_rev_aligned, fs);
    fprintf('  STOI(reverb, dry) aligned:         %.3f\n', stoi_reverb_floor);
catch
    stoi_reverb_floor = NaN;
end

try
    [vq_floor, ~, ~] = visqol(clean_rev_aligned, clean_dry_aligned, fs, Mode='speech');
    fprintf('  ViSQOL(reverb, dry) aligned:       %.2f MOS\n', vq_floor);
catch
    vq_floor = NaN;
end

fprintf('\n  The DNN after MVDR+PF must dereverberate + denoise.\n\n');

%% ===================== QUALITY METRICS =====================
fprintf('============================================\n');
fprintf('  QUALITY METRICS vs DRY CLEAN REFERENCE\n');
fprintf('============================================\n\n');

% --- SI-SDR (delay-aligned) ---
sisdr_mic  = si_sdr(x_mic_aligned,  clean_dry_aligned);
sisdr_mvdr = si_sdr(y_mvdr_aligned, clean_dry_aligned);
sisdr_pf   = si_sdr(y_pf_aligned,   clean_dry_aligned);

fprintf('--- SI-SDR vs DRY (delay-aligned, dB) — reverb floor = %+.1f dB ---\n', sisdr_reverb_floor);
fprintf('  Reverb floor:  %+7.2f dB   ← ceiling (clean reverb vs dry)\n', sisdr_reverb_floor);
fprintf('  Mic (ch1):     %+7.2f dB\n', sisdr_mic);
fprintf('  MVDR:          %+7.2f dB\n', sisdr_mvdr);
fprintf('  MVDR+PF:       %+7.2f dB\n', sisdr_pf);
fprintf('  MVDR gain:     %+7.2f dB  (over mic)\n', sisdr_mvdr - sisdr_mic);
fprintf('  PF gain:       %+7.2f dB  (over mic)\n\n', sisdr_pf - sisdr_mic);

% --- OSINR (Output Signal-to-Interference+Noise Ratio) ---
%  Competition submission requires OSINR.  OSINR = 10*log10(Ps/Pn)
%  where Ps = target power, Pn = residual (estimate − target) power.
resid_mic  = x_mic_aligned  - clean_rev_aligned;
resid_mvdr = y_mvdr_aligned - clean_rev_aligned;
resid_pf   = y_pf_aligned   - clean_rev_aligned;
osinr_mic  = 10*log10(mean(clean_rev_aligned.^2) / (mean(resid_mic.^2)  + 1e-9));
osinr_mvdr = 10*log10(mean(clean_rev_aligned.^2) / (mean(resid_mvdr.^2) + 1e-9));
osinr_pf   = 10*log10(mean(clean_rev_aligned.^2) / (mean(resid_pf.^2)   + 1e-9));

fprintf('--- OSINR (vs reverberant target, dB) ---\n');
fprintf('  Mic (ch1):     %+7.2f dB\n', osinr_mic);
fprintf('  MVDR:          %+7.2f dB\n', osinr_mvdr);
fprintf('  MVDR+PF:       %+7.2f dB\n', osinr_pf);
fprintf('  MVDR gain:     %+7.2f dB  (over mic)\n', osinr_mvdr - osinr_mic);
fprintf('  PF gain:       %+7.2f dB  (over mic)\n\n', osinr_pf - osinr_mic);

% --- SI-SDR vs reverberant reference (MVDR-specific metric) ---
sisdr_mic_r  = si_sdr(x_mic_aligned,  clean_rev_aligned);
sisdr_mvdr_r = si_sdr(y_mvdr_aligned, clean_rev_aligned);
sisdr_pf_r   = si_sdr(y_pf_aligned,   clean_rev_aligned);

fprintf('--- SI-SDR vs REVERBERANT ref (delay-aligned, dB) ---\n');
fprintf('  Mic (ch1):     %+7.2f dB\n', sisdr_mic_r);
fprintf('  MVDR:          %+7.2f dB\n', sisdr_mvdr_r);
fprintf('  MVDR+PF:       %+7.2f dB\n', sisdr_pf_r);
fprintf('  MVDR gain:     %+7.2f dB  (over mic)\n', sisdr_mvdr_r - sisdr_mic_r);
fprintf('  PF gain:       %+7.2f dB  (over mic)\n\n', sisdr_pf_r - sisdr_mic_r);

% --- STOI ---
try
    stoi_mic  = stoi(clean_dry_aligned, x_mic_aligned,  fs);
    stoi_mvdr = stoi(clean_dry_aligned, y_mvdr_aligned, fs);
    stoi_pf   = stoi(clean_dry_aligned, y_pf_aligned,   fs);

    fprintf('--- STOI (vs dry, delay-aligned) — reverb floor = %.3f ---\n', stoi_reverb_floor);
    fprintf('  Reverb floor:  %.3f  ← ceiling\n', stoi_reverb_floor);
    fprintf('  Mic (ch1):     %.3f\n', stoi_mic);
    fprintf('  MVDR:          %.3f\n', stoi_mvdr);
    fprintf('  MVDR+PF:       %.3f\n\n', stoi_pf);
catch ME
    fprintf('STOI unavailable: %s\n\n', ME.message);
    stoi_mic = NaN; stoi_mvdr = NaN; stoi_pf = NaN;
end

% --- ViSQOL ---
try
    [vq_mic, ~, ~]  = visqol(x_mic_aligned,  clean_dry_aligned, fs, Mode='speech');
    [vq_mvdr, ~, ~] = visqol(y_mvdr_aligned, clean_dry_aligned, fs, Mode='speech');
    [vq_pf, ~, ~]   = visqol(y_pf_aligned,   clean_dry_aligned, fs, Mode='speech');

    fprintf('--- ViSQOL MOS (vs dry, delay-aligned) ---\n');
    if ~isnan(vq_floor)
        fprintf('  Reverb floor:  %.2f  ← ceiling\n', vq_floor);
    end
    fprintf('  Mic (ch1):     %.2f\n', vq_mic);
    fprintf('  MVDR:          %.2f\n', vq_mvdr);
    fprintf('  MVDR+PF:       %.2f\n\n', vq_pf);
catch ME
    fprintf('ViSQOL unavailable: %s\n\n', ME.message);
    vq_mic = NaN; vq_mvdr = NaN; vq_pf = NaN;
end

% --- PESQ ---
try
    pesq_mic  = pesq(clean_dry_aligned, x_mic_aligned,  fs);
    pesq_mvdr = pesq(clean_dry_aligned, y_mvdr_aligned, fs);
    pesq_pf   = pesq(clean_dry_aligned, y_pf_aligned,   fs);

    fprintf('--- PESQ MOS (vs dry, delay-aligned) ---\n');
    fprintf('  Mic (ch1):     %.2f\n', pesq_mic);
    fprintf('  MVDR:          %.2f\n', pesq_mvdr);
    fprintf('  MVDR+PF:       %.2f\n\n', pesq_pf);
catch ME
    fprintf('PESQ unavailable: %s\n\n', ME.message);
    pesq_mic = NaN; pesq_mvdr = NaN; pesq_pf = NaN;
end

%% ===================== SAVE AUDIO =====================
fprintf('============================================\n');
fprintf('  SAVING AUDIO → %s\n', out_dir);
fprintf('============================================\n');

audiowrite(fullfile(out_dir, '1_clean_dry.wav'),              safe_norm(clean_dry), fs);
audiowrite(fullfile(out_dir, '2_clean_reverb_mic1.wav'),      safe_norm(clean_rev), fs);
audiowrite(fullfile(out_dir, '3_mixture_stereo.wav'),         safe_norm_stereo(mix_stereo), fs);
audiowrite(fullfile(out_dir, '4_mic_ch1.wav'),                safe_norm(x_mic), fs);
audiowrite(fullfile(out_dir, '5_mvdr_output.wav'),            y_mvdr, fs);
audiowrite(fullfile(out_dir, '6_mvdr_postfilter_output.wav'), y_pf, fs);

save(fullfile(out_dir, 'demo_results.mat'), ...
    'rir_target', 'rir_interf', 'rt60_vals', ...
    'absorption_final', 'scatter_final', 'refl_final', 'DRR', 'DRR_early', ...
    'sisdr_mic', 'sisdr_mvdr', 'sisdr_pf', 'sisdr_reverb_floor', ...
    'osinr_mic', 'osinr_mvdr', 'osinr_pf', ...
    'sir_actual', 'snr_actual', 'az_target', 'az_interf', ...
    'mic_pos_3d', 'source_target', 'source_interf', 'room_dim', ...
    'fs', 'd', 'RT60_target');

files_saved = {
    '1_clean_dry.wav              — original dry speech (competition reference)'
    '2_clean_reverb_mic1.wav      — reverberant target at mic1 (no interference)'
    '3_mixture_stereo.wav         — 2-ch reverberant mixture (what MVDR sees)'
    '4_mic_ch1.wav                — mic ch1 (reverb + interf + noise)'
    '5_mvdr_output.wav            — MVDR only'
    '6_mvdr_postfilter_output.wav — MVDR + Zelinski post-filter'
    'demo_results.mat             — RIRs, RT60, metrics'
};
for ii = 1:length(files_saved)
    fprintf('  %s\n', files_saved{ii});
end
fprintf('\n');

%% ===================== PLOTS =====================

% ---- Figure 1: RIR & Energy Decay Curves ----
figure('Name', 'RIR & RT60 Verification', 'Position', [50 100 1300 800]);

t_rir_tgt = (0:size(rir_target,2)-1) / fs;
t_rir_int = (0:size(rir_interf,2)-1) / fs;

subplot(2,2,1);
plot(t_rir_tgt, rir_target(1,:), 'b'); hold on;
plot(t_rir_tgt, rir_target(2,:), 'r', 'Color', [0.85 0.33 0.1]);
title('RIR: Target → Mics'); xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2'); grid on;

subplot(2,2,2);
plot(t_rir_int, rir_interf(1,:), 'b'); hold on;
plot(t_rir_int, rir_interf(2,:), 'r', 'Color', [0.85 0.33 0.1]);
title('RIR: Interference → Mics'); xlabel('Time (s)'); ylabel('Amplitude');
legend('Mic 1', 'Mic 2'); grid on;

% Energy Decay Curve — Target→Mic1
subplot(2,2,3);
ir1 = rir_target(1,:);
edc1 = cumsum(ir1(end:-1:1).^2);
edc1 = edc1(end:-1:1);
edc1_db = 10*log10(edc1 / max(edc1) + eps);
plot(t_rir_tgt, edc1_db, 'b', 'LineWidth', 1.5); hold on;
yline(-5, 'r--', '-5 dB');
yline(-35, 'r--', '-35 dB');
yline(-60, 'k:', 'RT60 = -60 dB');
title(sprintf('EDC: Target→Mic1   RT60 = %.3f s', rt60_vals(1)));
xlabel('Time (s)'); ylabel('Energy (dB)'); grid on; ylim([-80 5]);

% Energy Decay Curve — Interf→Mic1
subplot(2,2,4);
ir3 = rir_interf(1,:);
edc3 = cumsum(ir3(end:-1:1).^2);
edc3 = edc3(end:-1:1);
edc3_db = 10*log10(edc3 / max(edc3) + eps);
plot(t_rir_int, edc3_db, 'b', 'LineWidth', 1.5); hold on;
yline(-5, 'r--', '-5 dB');
yline(-35, 'r--', '-35 dB');
yline(-60, 'k:', 'RT60 = -60 dB');
title(sprintf('EDC: Interf→Mic1   RT60 = %.3f s', rt60_vals(3)));
xlabel('Time (s)'); ylabel('Energy (dB)'); grid on; ylim([-80 5]);

sgtitle(sprintf('Room Impulse Responses — Room [%.1f×%.1f×%.1f] m, RT60 = %.3f s', ...
    room_dim, mean(rt60_vals)), 'FontSize', 14, 'FontWeight', 'bold');

% ---- Figure 2: Waveforms ----
figure('Name', 'Waveforms', 'Position', [100 50 1400 900]);
t_wav = (0:length(clean_dry)-1) / fs;

subplot(5,1,1);
plot(t_wav, clean_dry, 'Color', [0.0 0.5 0.0]);
title('1. Clean (dry) — competition reference'); ylabel('Amp'); grid on;
xlim([0 t_wav(end)]);

subplot(5,1,2);
plot(t_wav, clean_rev, 'Color', [0.0 0.4 0.7]);
title('2. Reverberant target (mic1, no interference)'); ylabel('Amp'); grid on;
xlim([0 t_wav(end)]);

subplot(5,1,3);
plot(t_wav, x_mic, 'Color', [0.6 0.0 0.0]);
title('3. Mic ch1 (reverb + interference + noise)'); ylabel('Amp'); grid on;
xlim([0 t_wav(end)]);

subplot(5,1,4);
plot(t_wav, y_mvdr, 'Color', [0.85 0.33 0.1]);
title(sprintf('4. MVDR output   (SI-SDR = %+.1f dB)', sisdr_mvdr)); ylabel('Amp'); grid on;
xlim([0 t_wav(end)]);

subplot(5,1,5);
plot(t_wav, y_pf, 'Color', [0.49 0.18 0.56]);
title(sprintf('5. MVDR + Post-Filter   (SI-SDR = %+.1f dB)', sisdr_pf));
ylabel('Amp'); xlabel('Time (s)'); grid on;
xlim([0 t_wav(end)]);

sgtitle('Task 2 Reverberant Demo — Waveforms', 'FontSize', 14, 'FontWeight', 'bold');

% ---- Figure 3: Spectrograms ----
figure('Name', 'Spectrograms', 'Position', [150 50 1400 900]);

subplot(3,2,1);
spectrogram(clean_dry, 512, 384, 512, fs, 'yaxis');
title('Clean (dry)'); colorbar off;

subplot(3,2,2);
spectrogram(clean_rev, 512, 384, 512, fs, 'yaxis');
title('Reverberant target (mic1)'); colorbar off;

subplot(3,2,3);
spectrogram(x_mic, 512, 384, 512, fs, 'yaxis');
title('Mic ch1 (reverb mixture)'); colorbar off;

subplot(3,2,4);
spectrogram(y_mvdr, 512, 384, 512, fs, 'yaxis');
title(sprintf('MVDR   (SI-SDR %+.1f dB)', sisdr_mvdr)); colorbar off;

subplot(3,2,5);
spectrogram(y_pf, 512, 384, 512, fs, 'yaxis');
title(sprintf('MVDR + PF   (SI-SDR %+.1f dB)', sisdr_pf)); colorbar off;

sgtitle('Task 2 Reverberant Demo — Spectrograms', 'FontSize', 14, 'FontWeight', 'bold');

%% ===================== SUMMARY =====================
fprintf('============================================\n');
fprintf('  SUMMARY\n');
fprintf('============================================\n');
fprintf('Room:           [%.1f × %.1f × %.1f] m\n', room_dim);
fprintf('RT60 (avg):     %.3f s  (target %.2f s)\n', mean(rt60_vals), RT60_target);
fprintf('SIR actual:     %.2f dB  (target %d dB)\n', sir_actual, SIR_dB);
fprintf('SNR actual:     %.2f dB  (target %d dB)\n', snr_actual, SNR_dB);
fprintf('\n');
fprintf('SI-SDR (vs dry):  Mic=%+.2f   MVDR=%+.2f   PF=%+.2f dB\n', ...
    sisdr_mic, sisdr_mvdr, sisdr_pf);
fprintf('SI-SDR (vs rev):  Mic=%+.2f   MVDR=%+.2f   PF=%+.2f dB\n', ...
    sisdr_mic_r, sisdr_mvdr_r, sisdr_pf_r);
if ~isnan(stoi_mic)
    fprintf('STOI:             Mic=%.3f   MVDR=%.3f   PF=%.3f\n', ...
        stoi_mic, stoi_mvdr, stoi_pf);
end
if exist('vq_mic','var') && ~isnan(vq_mic)
    fprintf('ViSQOL MOS:       Mic=%.2f    MVDR=%.2f    PF=%.2f\n', ...
        vq_mic, vq_mvdr, vq_pf);
end
if exist('pesq_mic','var') && ~isnan(pesq_mic)
    fprintf('PESQ MOS:         Mic=%.2f    MVDR=%.2f    PF=%.2f\n', ...
        pesq_mic, pesq_mvdr, pesq_pf);
end
fprintf('OSINR:            Mic=%+.2f   MVDR=%+.2f   PF=%+.2f dB\n', ...
    osinr_mic, osinr_mvdr, osinr_pf);
fprintf('\nAudio saved to: %s\n', out_dir);
fprintf('============================================\n');
fprintf('  Listen and verify, then proceed to full dataset generation.\n');
fprintf('============================================\n');


%% ################################################################
%  LOCAL FUNCTIONS
%  ################################################################

function sig = load_audio(filepath, fs, duration)
    target_samples = fs * duration;
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
end

function x = safe_norm(x)
    pk = max(abs(x(:)));
    if pk > 0, x = 0.99 * x / pk; end
end

function x = safe_norm_stereo(x)
    pk = max(abs(x(:)));
    if pk > 0, x = 0.99 * x / pk; end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

%% ==================== ISM SHOEBOX RIR ====================
% Image Source Method for shoebox rooms
% Vectorised Allen & Berkley with fractional-delay interpolation,
% per-surface absorption & scattering.
%
%  ir = ismShoeboxRIR(roomDim, tx, rx, fs, c, order, absorption6, scattering6)
%
%  roomDim     : [Lx Ly Lz] room dimensions (m)
%  tx          : [1×3] source position
%  rx          : [N×3] microphone positions
%  fs          : sample rate
%  c           : speed of sound
%  order       : ISM image order
%  absorption6 : [6×1] absorption per surface (floor, front, back, left, right, ceiling)
%  scattering6 : [6×1] scattering per surface
%
%  Returns ir : [N × L] impulse responses

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

    % Image source positions
    Ximg = ((-1).^Nx) .* tx(1) + 2 .* Nx .* L(1);
    Yimg = ((-1).^Ny) .* tx(2) + 2 .* Ny .* L(2);
    Zimg = ((-1).^Nz) .* tx(3) + 2 .* Nz .* L(3);

    ax = abs(Nx); ay = abs(Ny); az = abs(Nz);

    % Per-surface reflection counts
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

    % Surface order: floor(1), front(2), back(3), left(4), right(5), ceiling(6)
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
% Schroeder backward integration with T30 → T20 → T10 fallback

function rt60 = estimate_rt60(ir, fs)
    ir = ir(:);
    energy = cumsum(ir(end:-1:1).^2);
    energy = energy(end:-1:1);
    energy = energy / max(energy + eps);
    energy_db = 10*log10(energy + eps);

    idx_5db = find(energy_db <= -5, 1, 'first');
    if isempty(idx_5db), rt60 = NaN; return; end

    % T30: -5 dB to -35 dB → ×2
    idx_35db = find(energy_db <= -35, 1, 'first');
    if ~isempty(idx_35db)
        rt60 = 2 * ((idx_35db - idx_5db) / fs);
        return;
    end

    % T20: -5 dB to -25 dB → ×3
    idx_25db = find(energy_db <= -25, 1, 'first');
    if ~isempty(idx_25db)
        rt60 = 3 * ((idx_25db - idx_5db) / fs);
        return;
    end

    % T10: -5 dB to -15 dB → ×6
    idx_15db = find(energy_db <= -15, 1, 'first');
    if ~isempty(idx_15db)
        rt60 = 6 * ((idx_15db - idx_5db) / fs);
        return;
    end

    rt60 = NaN;
end
