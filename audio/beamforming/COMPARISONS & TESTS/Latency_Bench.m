%% ############ MVDR LATENCY BENCHMARK — Real-Time Smartphone Feasibility ############
%  Measures per-frame and full-pipeline latency of the MVDR beamformer.
%  Reports Real-Time Factor (RTF) and estimated smartphone performance.
%
%  WORKFLOW:
%    1. Set K_selected below (MVDR weight-update interval)
%    2. Run this script
%    3. Copy the "PASTE INTO NOTEBOOK" block from the output
%    4. Paste into the Python notebook cell → run → get final combined report
%
%  Smartphone estimation: desktop MATLAB timing × slowdown factor (3–5×)
%  to approximate ARM CPU (e.g. Snapdragon / Apple A-series).
%
clear; clc;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation');

%% ================= USER SETTING — SET YOUR K HERE =================
K_selected = 4;   % ← CHANGE THIS: Update MVDR weights every K frames
                   %   K=1  → every frame (max adaptivity, highest cost)
                   %   K=4  → every 4 frames (recommended sweet spot)
                   %   K=8  → every 8 frames (lower cost, slower tracking)
                   %   K=16 → every 16 frames (minimal cost, slow tracking)

%% ================= CONFIGURATION =================
fs   = 16000;
c    = 340;

d    = 0.08;
mic_pos = [-d/2; d/2];
numMics = 2;

theta_target = 0;   % degrees
alpha = 0.99;       % covariance smoothing
delta = 1e-6;       % diagonal loading

% STFT parameters (must match your MVDR pipeline)
N      = 512;
hop    = 128;
nfft   = 512;
window = sqrt(hann(N, 'periodic'));

numFreqs = nfft/2 + 1;   % 257
freqs    = (0:numFreqs-1)' * fs / nfft;

% Benchmark settings
test_durations_sec = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0];
n_warmup           = 3;       % warmup runs (excluded from timing)
n_runs             = 20;      % timed runs per duration
smartphone_factor  = 3.5;     % estimated ARM vs desktop MATLAB slowdown

hop_duration_ms = (hop / fs) * 1000;   % 8.0 ms at 16 kHz / hop=128

fprintf('==========================================================================\n');
fprintf('  MVDR Beamformer — Latency Benchmark  (K=%d)\n', K_selected);
fprintf('==========================================================================\n');
fprintf('  fs=%d Hz | N=%d | hop=%d | nfft=%d | mics=%d\n', fs, N, hop, nfft, numMics);
fprintf('  Weight update interval K = %d  (every %d frames = %.1f ms)\n', ...
    K_selected, K_selected, K_selected * hop_duration_ms);
fprintf('  Frame duration (real-time budget per hop): %.1f ms\n', hop_duration_ms);
fprintf('  Warmup iters: %d | Timed iters: %d\n', n_warmup, n_runs);
fprintf('  Smartphone slowdown factor: %.1fx\n', smartphone_factor);
fprintf('==========================================================================\n\n');

%% ================= PRE-COMPUTE FIXED QUANTITIES =================
% Steering vector is fixed for a given look direction — compute once
dvec = compute_steering_vector(theta_target, freqs, mic_pos, c);

%% ================= COMPONENT-LEVEL BENCHMARK =================
% Use a 3-second signal for component profiling
comp_dur   = 3.0;
comp_len   = round(comp_dur * fs);
x_comp     = randn(comp_len, numMics) * 0.01;
numFrames_comp = floor((comp_len - N) / hop) + 1;

fprintf('--- Component-Level Profiling (%.1fs / %d frames) ---\n', comp_dur, numFrames_comp);

% 1) Full STFT
t_stft = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    X_tmp = stft_multichannel(x_comp, window, hop, nfft);
    elapsed = toc;
    if r > n_warmup
        t_stft(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  stft_multichannel        : %7.2f ± %.2f ms\n', mean(t_stft), std(t_stft));

% Get STFT for subsequent steps
X_bench = stft_multichannel(x_comp, window, hop, nfft);

% 2) Covariance estimation (full loop)
t_cov = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    Rxx_tmp = init_covariance(numFreqs, numMics);
    tic;
    for frm = 1:numFrames_comp
        X_frame = squeeze(X_bench(:, frm, :));
        Rxx_tmp = update_covariance(Rxx_tmp, X_frame, alpha);
    end
    elapsed = toc;
    if r > n_warmup
        t_cov(r - n_warmup) = elapsed * 1000;
    end
end
Rxx_bench = Rxx_tmp;  % save for next steps
fprintf('  Covariance (all frames)  : %7.2f ± %.2f ms\n', mean(t_cov), std(t_cov));
fprintf('  Covariance per frame     : %7.4f ms\n', mean(t_cov) / numFrames_comp);

% 3) Single covariance update (per-frame cost)
X_frame_single = squeeze(X_bench(:, 1, :));
Rxx_single     = init_covariance(numFreqs, numMics);
t_cov1 = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    Rxx_single = update_covariance(Rxx_single, X_frame_single, alpha);
    elapsed = toc;
    if r > n_warmup
        t_cov1(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  Single cov update        : %7.4f ± %.4f ms\n', mean(t_cov1), std(t_cov1));

% 4) Steering vector (usually precomputed, but measure anyway)
t_steer = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    dvec_tmp = compute_steering_vector(theta_target, freqs, mic_pos, c);
    elapsed = toc;
    if r > n_warmup
        t_steer(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  Steering vector          : %7.4f ± %.4f ms\n', mean(t_steer), std(t_steer));

% 5) MVDR weight computation
t_wt = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    W_tmp = compute_mvdr_weights(Rxx_bench, dvec, delta);
    elapsed = toc;
    if r > n_warmup
        t_wt(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  MVDR weights             : %7.4f ± %.4f ms\n', mean(t_wt), std(t_wt));

% 6) Apply MVDR (beamform all frames)
W_bench = compute_mvdr_weights(Rxx_bench, dvec, delta);
t_apply = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    Y_tmp = apply_mvdr(X_bench, W_bench);
    elapsed = toc;
    if r > n_warmup
        t_apply(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  apply_mvdr (all frames)  : %7.2f ± %.2f ms\n', mean(t_apply), std(t_apply));
fprintf('  apply_mvdr per frame     : %7.4f ms\n', mean(t_apply) / numFrames_comp);

% 7) iSTFT
Y_bench = apply_mvdr(X_bench, W_bench);
t_istft = zeros(n_runs, 1);
for r = 1:(n_warmup + n_runs)
    tic;
    y_tmp = istft_single_channel(Y_bench, window, hop, nfft, comp_len);
    elapsed = toc;
    if r > n_warmup
        t_istft(r - n_warmup) = elapsed * 1000;
    end
end
fprintf('  istft_single_channel     : %7.2f ± %.2f ms\n', mean(t_istft), std(t_istft));

fprintf('\n');

%% ================= FULL-PIPELINE BENCHMARK (varying durations) =================
fprintf('==========================================================================\n');
fprintf('  Full-Pipeline Latency vs Audio Duration\n');
fprintf('==========================================================================\n');
fprintf('  %8s  %10s  %10s  %8s  %10s  %10s\n', ...
    'Duration', 'Full(ms)', 'Std(ms)', 'RTF-CPU', 'RTF-Phone', 'Status');
fprintf('  %s  %s  %s  %s  %s  %s\n', ...
    repmat('-',1,8), repmat('-',1,10), repmat('-',1,10), ...
    repmat('-',1,8), repmat('-',1,10), repmat('-',1,10));

results = struct();

for di = 1:length(test_durations_sec)
    dur = test_durations_sec(di);
    sig_len = round(dur * fs);
    x_test  = randn(sig_len, numMics) * 0.01;
    
    full_times = zeros(n_runs, 1);
    
    for r = 1:(n_warmup + n_runs)
        tic;
        
        % === Full MVDR pipeline (matches your code exactly) ===
        X_t = stft_multichannel(x_test, window, hop, nfft);
        [nF, nT, nM] = size(X_t);
        
        Rxx_t = init_covariance(nF, nM);
        for frm = 1:nT
            X_frame_t = squeeze(X_t(:, frm, :));
            Rxx_t = update_covariance(Rxx_t, X_frame_t, alpha);
        end
        
        % Steering vector (precomputed in real app, but include for fairness)
        dvec_t = compute_steering_vector(theta_target, freqs(1:nF), mic_pos, c);
        W_t    = compute_mvdr_weights(Rxx_t, dvec_t, delta);
        Y_t    = apply_mvdr(X_t, W_t);
        y_t    = istft_single_channel(Y_t, window, hop, nfft, sig_len);
        
        elapsed = toc;
        if r > n_warmup
            full_times(r - n_warmup) = elapsed * 1000;
        end
    end
    
    avg_ms  = mean(full_times);
    std_ms  = std(full_times);
    audio_ms = dur * 1000;
    
    rtf_cpu   = avg_ms / audio_ms;
    rtf_phone = rtf_cpu * smartphone_factor;
    
    if rtf_phone < 1.0
        status = 'OK';
    else
        status = 'SLOW';
    end
    
    fprintf('  %7.1fs  %9.2f  %9.2f  %8.4f  %9.4f  %s\n', ...
        dur, avg_ms, std_ms, rtf_cpu, rtf_phone, status);
    
    results(di).duration_s    = dur;
    results(di).full_mean_ms  = avg_ms;
    results(di).full_std_ms   = std_ms;
    results(di).rtf_cpu       = rtf_cpu;
    results(di).rtf_phone     = rtf_phone;
    results(di).realtime_ok   = rtf_phone < 1.0;
end

%% ================= FRAME-BY-FRAME STREAMING SIMULATION =================
% This simulates actual real-time operation: process one hop at a time.
fprintf('\n==========================================================================\n');
fprintf('  Frame-by-Frame Streaming Simulation (3s audio)\n');
fprintf('==========================================================================\n');

stream_dur = 3.0;
stream_len = round(stream_dur * fs);
x_stream   = randn(stream_len, numMics) * 0.01;
numFrames_stream = floor((stream_len - N) / hop) + 1;

% Pre-compute steering vector (done once at init)
dvec_stream = compute_steering_vector(theta_target, freqs, mic_pos, c);

% Simulate streaming: process frame-by-frame
frame_times = zeros(numFrames_stream, 1);
Rxx_stream  = init_covariance(numFreqs, numMics);

% Buffer for windowed frames
for frm = 1:numFrames_stream
    tic;
    
    % 1) Extract frame & apply window
    idx_start = (frm - 1) * hop + 1;
    idx_end   = idx_start + N - 1;
    x_frame   = x_stream(idx_start:idx_end, :);          % [N x numMics]
    x_win     = x_frame .* window;                         % apply window
    
    % 2) Per-frame FFT (single frame STFT)
    X_frame_fft = fft(x_win, nfft, 1);                    % [nfft x numMics]
    X_frame_fft = X_frame_fft(1:numFreqs, :);             % [numFreqs x numMics]
    
    % 3) Update covariance
    Rxx_stream = update_covariance(Rxx_stream, X_frame_fft, alpha);
    
    % 4) Compute MVDR weights (in real-time, may update less frequently)

    %    W is [numMics x numFreqs], transpose to [numFreqs x numMics]
    W_stream = compute_mvdr_weights(Rxx_stream, dvec_stream, delta);
    
    % 5) Apply beamforming to this frame
    %    W_stream.' is [numFreqs x numMics], X_frame_fft is [numFreqs x numMics]
    Y_frame = sum(conj(W_stream.') .* X_frame_fft, 2);   % [numFreqs x 1]
    
    % 6) Per-frame iFFT
    Y_full        = zeros(nfft, 1);
    Y_full(1:numFreqs) = Y_frame;
    Y_full(numFreqs+1:end) = conj(Y_frame(end-1:-1:2));
    y_frame_time  = real(ifft(Y_full, nfft));
    y_frame_time  = y_frame_time(1:N) .* window;          % apply synthesis window
    
    elapsed = toc;
    frame_times(frm) = elapsed * 1000;   % ms
end

% Stats
avg_frame    = mean(frame_times);
std_frame    = std(frame_times);
p50_frame    = prctile(frame_times, 50);
p95_frame    = prctile(frame_times, 95);
p99_frame    = prctile(frame_times, 99);
max_frame    = max(frame_times);

avg_phone    = avg_frame * smartphone_factor;
p95_phone    = p95_frame * smartphone_factor;
p99_phone    = p99_frame * smartphone_factor;
max_phone    = max_frame * smartphone_factor;

fprintf('  Frames processed: %d\n', numFrames_stream);
fprintf('  Frame budget    : %.2f ms (hop=%d @ %d Hz)\n\n', hop_duration_ms, hop, fs);

fprintf('  Desktop CPU:\n');
fprintf('    Mean   : %.4f ms/frame\n', avg_frame);
fprintf('    Median : %.4f ms/frame\n', p50_frame);
fprintf('    p95    : %.4f ms/frame\n', p95_frame);
fprintf('    p99    : %.4f ms/frame\n', p99_frame);
fprintf('    Max    : %.4f ms/frame\n', max_frame);

fprintf('\n  Estimated Smartphone (%.1fx slowdown):\n', smartphone_factor);
fprintf('    Mean   : %.4f ms/frame  [budget: %.1f ms]', avg_phone, hop_duration_ms);
if avg_phone < hop_duration_ms
    fprintf('  <<<  OK\n');
else
    fprintf('  <<<  TOO SLOW\n');
end
fprintf('    p95    : %.4f ms/frame', p95_phone);
if p95_phone < hop_duration_ms
    fprintf('  <<<  OK\n');
else
    fprintf('  <<<  TOO SLOW\n');
end
fprintf('    p99    : %.4f ms/frame', p99_phone);
if p99_phone < hop_duration_ms
    fprintf('  <<<  OK\n');
else
    fprintf('  <<<  TOO SLOW\n');
end
fprintf('    Max    : %.4f ms/frame', max_phone);
if max_phone < hop_duration_ms
    fprintf('  <<<  OK\n');
else
    fprintf('  <<<  SPIKE\n');
end

%% ================= BENCHMARK FOR SELECTED K =================
% Runs the streaming simulation with K_selected weight-update interval.
fprintf('\n==========================================================================\n');
fprintf('  Benchmark: K=%d (update MVDR weights every %d frames)\n', K_selected, K_selected);
fprintf('==========================================================================\n');

ft_sel = zeros(numFrames_stream, 1);
Rxx_sel = init_covariance(numFreqs, numMics);
W_sel   = ones(numMics, numFreqs) / numMics;  % init uniform [numMics x numFreqs]

for frm = 1:numFrames_stream
    tic;
    
    idx_start = (frm - 1) * hop + 1;
    idx_end   = idx_start + N - 1;
    x_f = x_stream(idx_start:idx_end, :) .* window;
    
    X_f = fft(x_f, nfft, 1);
    X_f = X_f(1:numFreqs, :);
    
    Rxx_sel = update_covariance(Rxx_sel, X_f, alpha);
    
    % Only recompute weights every K_selected frames
    if mod(frm - 1, K_selected) == 0
        W_sel = compute_mvdr_weights(Rxx_sel, dvec_stream, delta);
    end
    
    Y_f = sum(conj(W_sel.') .* X_f, 2);  % W_sel.' [numFreqs x numMics]
    
    Y_full_sel = zeros(nfft, 1);
    Y_full_sel(1:numFreqs) = Y_f;
    Y_full_sel(numFreqs+1:end) = conj(Y_f(end-1:-1:2));
    y_f = real(ifft(Y_full_sel, nfft));
    y_f = y_f(1:N) .* window;
    
    elapsed = toc;
    ft_sel(frm) = elapsed * 1000;
end

avg_sel    = mean(ft_sel);
std_sel    = std(ft_sel);
p95_sel    = prctile(ft_sel, 95);
p99_sel    = prctile(ft_sel, 99);
max_sel    = max(ft_sel);
phone_sel  = avg_sel * smartphone_factor;
p95_phone_sel = p95_sel * smartphone_factor;

fprintf('  Desktop CPU:\n');
fprintf('    Mean   : %.4f ms/frame\n', avg_sel);
fprintf('    Std    : %.4f ms\n', std_sel);
fprintf('    p95    : %.4f ms/frame\n', p95_sel);
fprintf('    p99    : %.4f ms/frame\n', p99_sel);
fprintf('    Max    : %.4f ms/frame\n', max_sel);

fprintf('\n  Estimated Smartphone (%.1fx):\n', smartphone_factor);
fprintf('    Mean   : %.4f ms/frame  [budget: %.1f ms]', phone_sel, hop_duration_ms);
if phone_sel < hop_duration_ms
    fprintf('  <<< OK\n');
else
    fprintf('  <<< TOO SLOW\n');
end
fprintf('    p95    : %.4f ms/frame', p95_phone_sel);
if p95_phone_sel < hop_duration_ms
    fprintf('  <<< OK\n');
else
    fprintf('  <<< TOO SLOW\n');
end

%% ================= PASTE-READY OUTPUT FOR PYTHON NOTEBOOK =================
fprintf('\n==========================================================================\n');
fprintf('  ╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('  ║  PASTE THE FOLLOWING INTO THE PYTHON NOTEBOOK CELL:            ║\n');
fprintf('  ╚══════════════════════════════════════════════════════════════════╝\n');
fprintf('==========================================================================\n');
fprintf('\n');
fprintf('mvdr_K = %d\n', K_selected);
fprintf('mvdr_ms_per_frame = %.4f\n', avg_sel);
fprintf('\n');
fprintf('==========================================================================\n');

%% ================= FINAL VERDICT =================
fprintf('\n==========================================================================\n');
fprintf('  FINAL VERDICT\n');
fprintf('==========================================================================\n');
fprintf('  MVDR config: N=%d, hop=%d, nfft=%d, %d mics, K=%d\n', N, hop, nfft, numMics, K_selected);
fprintf('  Frame budget: %.1f ms\n', hop_duration_ms);
fprintf('\n');

if phone_sel < hop_duration_ms
    headroom = ((hop_duration_ms - phone_sel) / hop_duration_ms) * 100;
    fprintf('  ✅ MVDR beamformer (K=%d): REAL-TIME CAPABLE on smartphone\n', K_selected);
    fprintf('  Average per-frame (phone est.): %.4f ms  (%.0f%% headroom)\n', ...
        phone_sel, headroom);
else
    overshoot = ((phone_sel - hop_duration_ms) / hop_duration_ms) * 100;
    fprintf('  ❌ MVDR beamformer (K=%d): NOT real-time capable on smartphone\n', K_selected);
    fprintf('  Average per-frame (phone est.): %.4f ms  (%.0f%% over budget)\n', ...
        phone_sel, overshoot);
    fprintf('\n  Recommendations:\n');
    fprintf('    1. Increase K (update weights less often)\n');
    fprintf('    2. Increase hop size (e.g., 256 → 16 ms budget)\n');
    fprintf('    3. Implement in C/C++ with NEON SIMD for ARM\n');
end

fprintf('\n  Next step: paste mvdr_K and mvdr_ms_per_frame into the notebook\n');
fprintf('  and run the combined pipeline analysis.\n');
fprintf('==========================================================================\n');
