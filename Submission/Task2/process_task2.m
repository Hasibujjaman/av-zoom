%% process_task2.m
% Task 2: Reverberant MVDR beamforming + RealNet DNN enhancement
% SP Cup 2026 Phase 2
%
% Loads mixture data from Task2_Reverberant_5dB.mat, applies MPDR beamformer
% with Zelinski post-filter, then enhances via RealNet V2 DNN.
% Applies propagation delay compensation before computing metrics.
%
% Required files (same folder):
%   Task2_Reverberant_5dB.mat, mixture_signal{1..3}.wav,
%   target_signal.flac, interference_signal{1..3}.flac,
%   model_task2.pt, dnn_inference.py
%
% Outputs:
%   processed_signal{1..3}.wav, amplitude/spectrogram PNGs

clear; clc; close all;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

mat_path       = fullfile(script_dir, 'Task2_Reverberant_5dB.mat');
dnn_checkpoint = fullfile(script_dir, 'model_task2.pt');
dnn_script     = fullfile(script_dir, 'dnn_inference.py');
python_cmd     = 'python3';

%% Load data
assert(isfile(mat_path), 'Task2_Reverberant_5dB.mat not found. Run Task2_Reverberant_5dB.m first.');
data = load(mat_path);
params   = data.params;
examples = data.examples;

fs           = params.fs;
duration     = params.duration_s;
L            = fs * duration;
c            = params.speed_of_sound;
d_mic        = params.mic_spacing_m;
mic_pos      = params.mic_positions;
theta_target = params.theta_target_deg;
theta_interf = params.theta_interf_deg;
sir_db       = params.sir_db;
snr_db       = params.snr_db;
room_dim     = params.room_dim;
RT60_target  = params.RT60;

% RT60 achieved and propagation delay
if isfield(data, 'rir_data') && isfield(data.rir_data, 'RT60_s')
    rt60_mid = data.rir_data.RT60_s;
else
    rt60_mid = RT60_target;
end
if isfield(data, 'rir_data') && isfield(data.rir_data, 'prop_delay_smp')
    prop_delay_smp = data.rir_data.prop_delay_smp;
else
    if isfield(data, 'rir_data') && isfield(data.rir_data, 'target_rir')
        [~, pk_idx] = max(abs(data.rir_data.target_rir(1,:)));
        prop_delay_smp = pk_idx - 1;
    else
        prop_delay_smp = round(1.0 / c * fs);
    end
end
fprintf('Propagation delay: %d smp (%.2f ms)\n', prop_delay_smp, prop_delay_smp/fs*1000);

interf_labels = {'Female speech', 'Music', 'Noise'};
n_examples    = length(examples);

%% STFT parameters
nfft  = 512;
hop   = 128;
win   = sqrt(hann(nfft, 'periodic'));
delta = 1e-3;

%% Steering vector
freqs = (0:(nfft/2))' * fs / nfft;
dvec  = compute_steering_vec(theta_target, freqs, mic_pos, c);

fprintf('Task 2 — Reverberant beamforming + DNN\n');
fprintf('Room: [%.1fx%.1fx%.1f] m, RT60=%.2f s\n', room_dim, rt60_mid);
fprintf('SIR=%d dB  SNR=%d dB  theta_t=%d  theta_i=%d\n\n', ...
    sir_db, snr_db, theta_target, theta_interf);

%% Load clean target
target_flac = fullfile(script_dir, 'target_signal.flac');
if isfile(target_flac)
    clean_target_full = load_audio(target_flac, fs, duration);
else
    clean_target_full = data.target_signal;
    if length(clean_target_full) > L
        clean_target_full = clean_target_full(1:L);
    end
end

%% Process each example
for ex = 1:n_examples
    fprintf('--- Example %d/%d (%s) ---\n', ex, n_examples, interf_labels{ex});

    % load mixture
    mix_wav = fullfile(script_dir, sprintf('mixture_signal%d.wav', ex));
    if isfile(mix_wav)
        mixture = audioread(mix_wav);
    else
        mixture = examples(ex).mixture_signal;
    end

    % load interference
    interf_flac = fullfile(script_dir, sprintf('interference_signal%d.flac', ex));
    if isfile(interf_flac)
        interf_signal = load_audio(interf_flac, fs, duration);
    else
        interf_signal = examples(ex).interference_signal;
    end

    % STFT
    X = stft_mc(mixture, win, hop, nfft);
    [numFreqs, numFrames, numMics] = size(X);

    % batch covariance
    Rxx = zeros(numMics, numMics, numFreqs);
    for n = 1:numFrames
        for k = 1:numFreqs
            xk = squeeze(X(k, n, :));
            Rxx(:,:,k) = Rxx(:,:,k) + (xk * xk');
        end
    end
    Rxx = Rxx / numFrames;

    % MPDR weights
    W = compute_mvdr_weights(Rxx, dvec, delta);

    % Zelinski post-filter
    Y_pf = mvdr_postfilter(X, W, dvec);

    % iSTFT
    y_mvdr_pf = real(istft_sc(Y_pf, win, hop, nfft, L));
    y_mvdr_pf = y_mvdr_pf(1:L);

    % trim 20 ms edges
    trim = round(0.020 * fs);
    idx  = (trim+1):(L-trim);
    y_mvdr_pf   = y_mvdr_pf(idx);
    clean_dry   = clean_target_full(idx);
    mix_trimmed = mixture(idx, :);

    y_mvdr_pf   = peak_norm(y_mvdr_pf);
    clean_dry   = peak_norm(clean_dry);
    mix_trimmed = peak_norm_stereo(mix_trimmed);

    % DNN enhancement
    tmp_in  = fullfile(script_dir, sprintf('_tmp_mvdr_%d.wav', ex));
    tmp_out = fullfile(script_dir, sprintf('_tmp_dnn_%d.wav', ex));
    audiowrite(tmp_in, y_mvdr_pf, fs);

    dnn_cmd = sprintf('%s "%s" --input "%s" --output "%s" --checkpoint "%s"', ...
        python_cmd, dnn_script, tmp_in, tmp_out, dnn_checkpoint);

    fprintf('  Running DNN...\n');
    [status, result] = system(dnn_cmd);

    if status == 0 && isfile(tmp_out)
        processed_signal = audioread(tmp_out);
        processed_signal = processed_signal(1:min(end, length(clean_dry)));
        processed_signal = peak_norm(processed_signal);
        fprintf('  DNN done.\n');
    else
        fprintf('  DNN failed (status=%d), using MVDR+PF output.\n', status);
        if ~isempty(result), fprintf('  %s\n', result); end
        processed_signal = y_mvdr_pf;
    end

    if isfile(tmp_in),  delete(tmp_in);  end
    if isfile(tmp_out), delete(tmp_out); end

    % match lengths
    minL = min([length(clean_dry), length(processed_signal), length(y_mvdr_pf)]);
    clean_dry        = clean_dry(1:minL);
    processed_signal = processed_signal(1:minL);
    y_mvdr_pf        = y_mvdr_pf(1:minL);
    interf_signal_trimmed = peak_norm(interf_signal(idx(1:minL)));

    % propagation delay compensation for metrics
    % The RIR introduces a delay between the dry reference and the
    % processed signals. Align them before computing SI-SDR etc.
    if prop_delay_smp > 0
        N_align = minL - prop_delay_smp;
        clean_dry_comp        = clean_dry(1:N_align);
        y_mvdr_pf_comp        = y_mvdr_pf(prop_delay_smp+1 : prop_delay_smp+N_align);
        processed_signal_comp = processed_signal(prop_delay_smp+1 : prop_delay_smp+N_align);
        mix_comp              = mix_trimmed(prop_delay_smp+1 : prop_delay_smp+N_align, :);
        interf_comp           = interf_signal_trimmed(1:N_align);
    else
        N_align               = minL;
        clean_dry_comp        = clean_dry;
        y_mvdr_pf_comp        = y_mvdr_pf;
        processed_signal_comp = processed_signal;
        mix_comp              = mix_trimmed(1:minL, :);
        interf_comp           = interf_signal_trimmed;
    end

    clean_dry_comp        = peak_norm(clean_dry_comp);
    y_mvdr_pf_comp        = peak_norm(y_mvdr_pf_comp);
    processed_signal_comp = peak_norm(processed_signal_comp);
    mix_comp              = peak_norm_stereo(mix_comp);
    interf_comp           = peak_norm(interf_comp);

    % metrics (on delay-compensated signals)
    metrics_ex = struct();

    metrics_ex.si_sdr_mixture   = si_sdr(mix_comp(:,1), clean_dry_comp);
    metrics_ex.si_sdr_mvdr_pf   = si_sdr(y_mvdr_pf_comp, clean_dry_comp);
    metrics_ex.si_sdr_processed = si_sdr(processed_signal_comp, clean_dry_comp);

    try
        metrics_ex.stoi_mixture   = stoi(clean_dry_comp, mix_comp(:,1), fs);
        metrics_ex.stoi_mvdr_pf   = stoi(clean_dry_comp, y_mvdr_pf_comp, fs);
        metrics_ex.stoi_processed = stoi(clean_dry_comp, processed_signal_comp, fs);
    catch
        metrics_ex.stoi_mixture   = NaN;
        metrics_ex.stoi_mvdr_pf   = NaN;
        metrics_ex.stoi_processed = NaN;
        fprintf('  STOI unavailable.\n');
    end

    try
        metrics_ex.visqol_mixture   = visqol(clean_dry_comp, mix_comp(:,1), fs);
        metrics_ex.visqol_mvdr_pf   = visqol(clean_dry_comp, y_mvdr_pf_comp, fs);
        metrics_ex.visqol_processed = visqol(clean_dry_comp, processed_signal_comp, fs);
    catch
        metrics_ex.visqol_mixture   = NaN;
        metrics_ex.visqol_mvdr_pf   = NaN;
        metrics_ex.visqol_processed = NaN;
        fprintf('  ViSQOL unavailable.\n');
    end

    fprintf('  SI-SDR: Mix=%.2f  MVDR+PF=%.2f  DNN=%.2f dB\n', ...
        metrics_ex.si_sdr_mixture, metrics_ex.si_sdr_mvdr_pf, metrics_ex.si_sdr_processed);
    fprintf('  STOI:   Mix=%.4f  MVDR+PF=%.4f  DNN=%.4f\n', ...
        metrics_ex.stoi_mixture, metrics_ex.stoi_mvdr_pf, metrics_ex.stoi_processed);
    fprintf('  ViSQOL: Mix=%.3f  MVDR+PF=%.3f  DNN=%.3f\n\n', ...
        metrics_ex.visqol_mixture, metrics_ex.visqol_mvdr_pf, metrics_ex.visqol_processed);

    % save processed audio (unmodified output, no delay compensation)
    audiowrite(fullfile(script_dir, sprintf('processed_signal%d.wav', ex)), ...
        processed_signal, fs);

    % store compensated versions for plots & metrics
    examples(ex).processed_signal      = processed_signal_comp;
    examples(ex).mvdr_pf_signal        = y_mvdr_pf_comp;
    examples(ex).clean_ref             = clean_dry_comp;
    examples(ex).interference_trimmed  = interf_comp;
    examples(ex).mixture_trimmed       = mix_comp;
    examples(ex).metrics               = metrics_ex;
    examples(ex).processed_signal_full = processed_signal;
end

%% Results summary
fprintf('\n========== TASK 2 RESULTS ==========\n\n');
fprintf('Config: SIR=%d dB, SNR=%d dB, RT60=%.2f s, Reverberant\n', sir_db, snr_db, rt60_mid);
fprintf('        Room=[%.1fx%.1fx%.1f] m, theta_t=%d, theta_i=%d, d=%.2f m\n\n', ...
    room_dim, theta_target, theta_interf, d_mic);

for ex = 1:n_examples
    m = examples(ex).metrics;
    fprintf('Example %d (%s):\n', ex, examples(ex).interference_type);
    fprintf('  SI-SDR (dB):  Mix=%+.2f  MVDR+PF=%+.2f  DNN=%+.2f\n', ...
        m.si_sdr_mixture, m.si_sdr_mvdr_pf, m.si_sdr_processed);
    fprintf('  STOI:         Mix=%.4f  MVDR+PF=%.4f  DNN=%.4f\n', ...
        m.stoi_mixture, m.stoi_mvdr_pf, m.stoi_processed);
    fprintf('  ViSQOL:       Mix=%.3f  MVDR+PF=%.3f  DNN=%.3f\n\n', ...
        m.visqol_mixture, m.visqol_mvdr_pf, m.visqol_processed);
end

%% Plots
for ex = 1:n_examples
    clean_plt  = examples(ex).clean_ref;
    interf_plt = examples(ex).interference_trimmed;
    mix_plt    = examples(ex).mixture_trimmed(:,1);
    proc_plt   = examples(ex).processed_signal;
    n_samp     = length(clean_plt);
    t_ax       = (0:n_samp-1) / fs;

    % waveforms
    fig1 = figure('Position', [50 50 1200 800], 'Color', 'w');
    sgtitle(sprintf('Task 2 — Ex %d: %s — Waveforms', ...
        ex, examples(ex).interference_type), 'FontWeight', 'bold');

    subplot(4,1,1);
    plot(t_ax, clean_plt, 'Color', [0.20 0.60 0.20]);
    ylabel('Amp'); title('Clean Target'); xlim([0 t_ax(end)]); grid on;

    subplot(4,1,2);
    plot(t_ax, interf_plt, 'Color', [0.85 0.33 0.10]);
    ylabel('Amp'); title(sprintf('%s (Interference)', examples(ex).interference_type));
    xlim([0 t_ax(end)]); grid on;

    subplot(4,1,3);
    plot(t_ax, mix_plt, 'Color', [0.00 0.45 0.74]);
    ylabel('Amp'); title('Mixture (ch1)'); xlim([0 t_ax(end)]); grid on;

    subplot(4,1,4);
    plot(t_ax, proc_plt, 'Color', [0.49 0.18 0.56]);
    ylabel('Amp'); xlabel('Time (s)'); title('Enhanced Output');
    xlim([0 t_ax(end)]); grid on;

    saveas(fig1, fullfile(script_dir, sprintf('amplitude_example%d.png', ex)));

    % spectrograms
    fig2 = figure('Position', [80 80 1200 800], 'Color', 'w');
    sgtitle(sprintf('Task 2 — Ex %d: %s — Spectrograms', ...
        ex, examples(ex).interference_type), 'FontWeight', 'bold');

    sw = hann(512, 'periodic');  sov = 384;

    subplot(4,1,1);
    spectrogram(clean_plt, sw, sov, 512, fs, 'yaxis');
    title('Clean Target'); colorbar;

    subplot(4,1,2);
    spectrogram(interf_plt, sw, sov, 512, fs, 'yaxis');
    title(sprintf('%s (Interference)', examples(ex).interference_type)); colorbar;

    subplot(4,1,3);
    spectrogram(mix_plt, sw, sov, 512, fs, 'yaxis');
    title('Mixture (ch1)'); colorbar;

    subplot(4,1,4);
    spectrogram(proc_plt, sw, sov, 512, fs, 'yaxis');
    title('Enhanced Output'); colorbar;

    saveas(fig2, fullfile(script_dir, sprintf('spectrogram_example%d.png', ex)));
end

fprintf('Task 2 processing complete.\n');
fprintf('Output folder: %s\n', script_dir);


%% ==== Local functions ====

function sig = load_audio(filepath, fs, duration)
    target_len = fs * duration;
    [sig, file_fs] = audioread(filepath);
    if file_fs ~= fs, sig = resample(sig, fs, file_fs); end
    if size(sig, 2) > 1, sig = mean(sig, 2); end
    if length(sig) > target_len
        sig = sig(1:target_len);
    elseif length(sig) < target_len
        sig = [sig; zeros(target_len - length(sig), 1)];
    end
    sig = sig - mean(sig);
end

function x = peak_norm(x)
    pk = max(abs(x(:)));
    if pk > 0, x = 0.99 * x / pk; end
end

function x = peak_norm_stereo(x)
    pk = max(abs(x(:)));
    if pk > 0, x = 0.99 * x / pk; end
end

function X = stft_mc(x, window, hop, nfft)
    [T, M] = size(x);
    N = length(window);
    numFrames = floor((T - N) / hop) + 1;
    numFreqs  = nfft/2 + 1;
    X = zeros(numFreqs, numFrames, M);
    for m = 1:M
        for n = 1:numFrames
            seg = x((n-1)*hop + (1:N), m) .* window;
            Xf  = fft(seg, nfft);
            X(:, n, m) = Xf(1:numFreqs);
        end
    end
end

function y = istft_sc(Y, window, hop, nfft, outLen)
    [~, numFrames] = size(Y);
    N = length(window);
    y = zeros(outLen, 1);
    wnorm = zeros(outLen, 1);
    for n = 1:numFrames
        pos = (n-1)*hop + (1:N);
        Yfull = [Y(:,n); conj(Y(end-1:-1:2,n))];
        frame = real(ifft(Yfull, nfft));
        y(pos)     = y(pos) + frame(1:N) .* window;
        wnorm(pos) = wnorm(pos) + window.^2;
    end
    nz = wnorm > eps;
    y(nz) = y(nz) ./ wnorm(nz);
end

function d = compute_steering_vec(az_deg, freqs, micPos, c)
    az = deg2rad(az_deg);
    numFreqs = length(freqs);
    numMics  = length(micPos);
    d = zeros(numMics, numFreqs);
    for k = 1:numFreqs
        tau = micPos * cos(az) / c;
        d(:,k) = exp(-1j * 2*pi*freqs(k) * tau);
    end
end

function W = compute_mvdr_weights(Rxx, d, delta)
    [numMics, ~, numFreqs] = size(Rxx);
    W = zeros(numMics, numFreqs);
    for k = 1:numFreqs
        R  = Rxx(:,:,k) + delta * trace(Rxx(:,:,k))/numMics * eye(numMics);
        dk = d(:,k);
        W(:,k) = R \ dk / (dk' * (R \ dk));
    end
end

function Y_pf = mvdr_postfilter(X, W, ~)
    [numFreqs, numFrames, ~] = size(X);

    Y_mvdr = zeros(numFreqs, numFrames);
    for n = 1:numFrames
        for k = 1:numFreqs
            Y_mvdr(k,n) = W(:,k)' * squeeze(X(k,n,:));
        end
    end

    alpha = 0.92;
    beta  = 0.02;
    Pxx   = zeros(numFreqs, 1);
    Px12  = zeros(numFreqs, 1);
    Y_pf  = zeros(numFreqs, numFrames);

    for n = 1:numFrames
        x1 = squeeze(X(:,n,1));
        x2 = squeeze(X(:,n,2));

        Pxx  = alpha*Pxx  + (1-alpha)*0.5*(abs(x1).^2 + abs(x2).^2);
        Px12 = alpha*Px12 + (1-alpha)*real(x1 .* conj(x2));

        G = max(min(Px12 ./ (Pxx + eps), 1), beta);
        Y_pf(:,n) = G .* Y_mvdr(:,n);
    end
end

function val = si_sdr(y, s)
    y = y(:); s = s(:);
    s = s - mean(s);
    y = y - mean(y);
    a = (s'*y) / (s'*s);
    s_target = a * s;
    e_noise  = y - s_target;
    val = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + 1e-9));
end
