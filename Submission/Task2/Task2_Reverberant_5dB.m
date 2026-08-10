%% Task2_Reverberant_5dB.m
% Generate reverberant 2-mic mixture signals for Task 2 (SP Cup 2026 Phase 2).
%
% Generates ISM shoebox RIRs with binary-search absorption tuning
% (RT60 = 0.5 s), convolves source audio, and creates 2-mic mixtures
% at SIR = 0 dB, SNR = 5 dB.
%
% Room: 4.9 x 4.9 x 4.9 m, RT60 = 0.5 s
% Array: 2-mic ULA, d = 0.08 m, room centre, h = 1.5 m
% Target: broadside (90 deg), 1 m     Interference: 40 deg, 1 m
%
% Input files  (same folder):
%   target_signal.wav, interference_signal{1,2,3}.wav
%
% Output files:
%   mixture_signal{1,2,3}.wav, Task2_Reverberant_5dB.mat

clear; clc; close all;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

target_path = fullfile(script_dir, 'target_signal.wav');
interf_paths = {
    fullfile(script_dir, 'interference_signal1.wav');
    fullfile(script_dir, 'interference_signal2.wav');
    fullfile(script_dir, 'interference_signal3.wav');
};
interf_labels = {'Female speech', 'Music', 'Noise'};

%% Parameters
fs       = 16000;
duration = 3;
L        = fs * duration;

room_dim    = [4.9, 4.9, 4.9];
RT60_target = 0.50;

c     = 340;
d_mic = 0.08;
mic_pos_1d   = [-d_mic/2; d_mic/2];
array_centre = [2.45, 2.45, 1.50];
mic_pos_3d   = [array_centre(1)-d_mic/2, array_centre(2), array_centre(3);
                array_centre(1)+d_mic/2, array_centre(2), array_centre(3)];

source_target = [2.45, 3.45, 1.50];
dist_interf   = 1.0;

theta_target = 90;
theta_interf = 40;
sir_db = 0;
snr_db = 5;

ism_order_init = 15;
rng(2026);

fprintf('Task 2 — Reverberant mixture generation\n');
fprintf('Room: [%.1fx%.1fx%.1f] m, RT60=%.2f s\n', room_dim, RT60_target);
fprintf('SIR=%d dB, SNR=%d dB, theta_t=%d, theta_i=%d\n\n', ...
    sir_db, snr_db, theta_target, theta_interf);

%% Load target
target_signal = load_audio(target_path, fs, duration);
fprintf('Target: %s\n', target_path);

%% Pre-compute ISM RIRs
fprintf('\nComputing RIRs (ISM)...\n');
t_pre = tic;

V = prod(room_dim);
S = 2*(room_dim(1)*room_dim(2) + room_dim(1)*room_dim(3) + room_dim(2)*room_dim(3));
alpha_ey = 1 - exp(-0.161 * V / (S * RT60_target));
refl_ey  = sqrt(1 - alpha_ey);
bounces  = ceil(3.5 / (-log10(refl_ey + eps)));
ism_order = max(bounces + 5, 30);
scatter_final = 0.0;

% binary search for absorption
alpha_lo = 0.05;
alpha_hi = 0.80;

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
fprintf('  alpha=%.4f, RT60=%.3f s (target %.3f s)\n', absorption_final, rt60_mid, RT60_target);

% target RIR
rir_target = ismShoeboxRIR(room_dim, source_target, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));
[~, peak_idx] = max(abs(rir_target(1,:)));
prop_delay_smp = peak_idx - 1;
fprintf('  Target RIR: %d smp, prop delay=%d smp (%.2f ms)\n', ...
    size(rir_target,2), prop_delay_smp, prop_delay_smp/fs*1000);

% interference RIR
az_rad = deg2rad(theta_interf);
source_interf = array_centre + dist_interf * [cos(az_rad), sin(az_rad), 0];
rir_interf = ismShoeboxRIR(room_dim, source_interf, mic_pos_3d, ...
    fs, c, ism_order, absorption_final*ones(6,1), scatter_final*ones(6,1));

fprintf('  RIR pre-computation: %.1f s\n\n', toc(t_pre));

%% Generate mixtures
n_examples = 3;
all_mixtures    = cell(n_examples, 1);
all_target_mc   = cell(n_examples, 1);
all_interf_mc   = cell(n_examples, 1);
all_noise_mc    = cell(n_examples, 1);
all_interf_mono = cell(n_examples, 1);

for ex = 1:n_examples
    fprintf('Example %d/%d (%s)\n', ex, n_examples, interf_labels{ex});

    interf_signal = load_audio(interf_paths{ex}, fs, duration);

    [mixture, target_mc, interf_mc, noise_mc] = create_reverb_mixture( ...
        target_signal, interf_signal, rir_target, rir_interf, L, sir_db, snr_db);

    all_mixtures{ex}    = mixture;
    all_target_mc{ex}   = target_mc;
    all_interf_mc{ex}   = interf_mc;
    all_noise_mc{ex}    = noise_mc;
    all_interf_mono{ex} = interf_signal;

    mix_path = fullfile(script_dir, sprintf('mixture_signal%d.wav', ex));
    audiowrite(mix_path, mixture, fs);
    fprintf('  Saved: %s\n', mix_path);

    int_path = fullfile(script_dir, sprintf('interference_signal%d.wav', ex));
    audiowrite(int_path, peak_norm(interf_signal), fs);
end

audiowrite(fullfile(script_dir, 'target_signal.wav'), peak_norm(target_signal), fs);

%% Save .mat
rir_data = struct();
rir_data.type              = 'reverberant';
rir_data.description       = sprintf('ISM shoebox, RT60=%.3fs, room=[%.1fx%.1fx%.1f]m', rt60_mid, room_dim);
rir_data.room_dim_m        = room_dim;
rir_data.RT60_s            = rt60_mid;
rir_data.RT60_target_s     = RT60_target;
rir_data.absorption        = absorption_final;
rir_data.scatter           = scatter_final;
rir_data.ism_order         = ism_order;
rir_data.target_rir        = rir_target;
rir_data.interf_rir        = rir_interf;
rir_data.source_target     = source_target;
rir_data.source_interf     = source_interf;
rir_data.array_centre      = array_centre;
rir_data.mic_positions_3d  = mic_pos_3d;
rir_data.prop_delay_smp    = prop_delay_smp;

params = struct();
params.fs               = fs;
params.duration_s       = duration;
params.n_mics           = 2;
params.mic_spacing_m    = d_mic;
params.mic_positions    = mic_pos_1d;
params.speed_of_sound   = c;
params.theta_target_deg = theta_target;
params.theta_interf_deg = theta_interf;
params.sir_db           = sir_db;
params.snr_db           = snr_db;
params.condition        = 'Reverberant';
params.task             = 'Task 2';
params.room_dim         = room_dim;
params.RT60             = RT60_target;
params.RT60_achieved    = rt60_mid;
params.date             = datestr(now);               
params.interf_labels    = {interf_labels{:}};         

for ex = 1:n_examples
    examples(ex).interference_type   = interf_labels{ex};    
    examples(ex).interference_signal = all_interf_mono{ex};
    examples(ex).mixture_signal      = all_mixtures{ex};
    examples(ex).target_mc           = all_target_mc{ex};
    examples(ex).interf_mc           = all_interf_mc{ex};
    examples(ex).noise_mc            = all_noise_mc{ex};
end

target_signal_out   = peak_norm(target_signal);       
mixture_signal      = all_mixtures{1};                
interference_signal = all_interf_mono{1};             

mat_path = fullfile(script_dir, 'Task2_Reverberant_5dB.mat');
save(mat_path, 'target_signal_out', 'interference_signal', 'mixture_signal', ...
    'rir_data', 'params', 'examples', '-v7.3');

target_signal = target_signal_out;                    
save(mat_path, 'target_signal', '-append');

fprintf('\nSaved: %s\n', mat_path);
fprintf('Done. Run process_task2.m next.\n');


%% ---- Local functions ----

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


function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_reverb_mixture(target_sig, interf_sig, rir_target, rir_interf, L, sir_db, snr_db)

    target_mc = zeros(L, 2);
    interf_mc = zeros(L, 2);
    for ch = 1:2
        t_conv = fftfilt(rir_target(ch,:)', target_sig);
        target_mc(:,ch) = t_conv(1:L);
        i_conv = fftfilt(rir_interf(ch,:)', interf_sig);
        interf_mc(:,ch) = i_conv(1:L);
    end

    Pt = mean(target_mc(:).^2) + eps;
    Pi = mean(interf_mc(:).^2) + eps;
    interf_mc = interf_mc * sqrt(Pt / (Pi * 10^(sir_db/10)));

    mixture_clean = target_mc + interf_mc;
    Ps = mean(mixture_clean(:).^2) + eps;
    noise_mc = sqrt(Ps / 10^(snr_db/10)) * randn(size(mixture_clean));
    mixture = mixture_clean + noise_mc;

    pk = max(abs(mixture(:)));
    if pk > 0
        sc = 0.99 / pk;
        mixture   = mixture   * sc;
        target_mc = target_mc * sc;
        interf_mc = interf_mc * sc;
        noise_mc  = noise_mc  * sc;
    end
end


function x = peak_norm(x)
    pk = max(abs(x(:)));
    if pk > 0, x = 0.99 * x / pk; end
end


function ir = ismShoeboxRIR(roomDim, tx, rx, fs, c, order, absorption6, scattering6)
    Lo = double(roomDim(:)).';
    tx = double(tx);
    rx = double(rx);

    alpha = absorption6(:);
    scat  = scattering6(:);
    refl  = sqrt(max(0, 1 - alpha)) .* sqrt(max(0, 1 - scat));

    n = -order:order;
    [Nx, Ny, Nz] = ndgrid(n, n, n);
    keep = (abs(Nx) + abs(Ny) + abs(Nz)) <= order;
    Nx = Nx(keep); Ny = Ny(keep); Nz = Nz(keep);

    Ximg = ((-1).^Nx) .* tx(1) + 2 .* Nx .* Lo(1);
    Yimg = ((-1).^Ny) .* tx(2) + 2 .* Ny .* Lo(2);
    Zimg = ((-1).^Nz) .* tx(3) + 2 .* Nz .* Lo(3);

    ax = abs(Nx); ay = abs(Ny); az = abs(Nz);

    nLeft = zeros(size(Nx)); nRight = nLeft;
    nFront = nLeft; nBack = nLeft;
    nFloor = nLeft; nCeil = nLeft;

    pos = Nx >= 0;
    nRight(pos)  = ceil(ax(pos)  ./ 2);
    nLeft(pos)   = floor(ax(pos) ./ 2);
    nLeft(~pos)  = ceil(ax(~pos) ./ 2);
    nRight(~pos) = floor(ax(~pos)./ 2);

    pos = Ny >= 0;
    nFront(pos)  = ceil(ay(pos)  ./ 2);
    nBack(pos)   = floor(ay(pos) ./ 2);
    nBack(~pos)  = ceil(ay(~pos) ./ 2);
    nFront(~pos) = floor(ay(~pos)./ 2);

    pos = Nz >= 0;
    nCeil(pos)   = ceil(az(pos)  ./ 2);
    nFloor(pos)  = floor(az(pos) ./ 2);
    nFloor(~pos) = ceil(az(~pos) ./ 2);
    nCeil(~pos)  = floor(az(~pos)./ 2);

    nReceivers = size(rx, 1);
    maxDist = sqrt(sum((max(abs([Ximg(:), Yimg(:), Zimg(:)]), [], 1) - rx(1,:)).^2));
    irLen = ceil(maxDist / c * fs) + 100;
    ir = zeros(nReceivers, irLen);

    for rr = 1:nReceivers
        dx = Ximg - rx(rr,1);
        dy = Yimg - rx(rr,2);
        dz = Zimg - rx(rr,3);
        dist = sqrt(dx.^2 + dy.^2 + dz.^2);

        reflCoef = (refl(1).^nLeft) .* (refl(2).^nRight) .* ...
                   (refl(3).^nFront) .* (refl(4).^nBack) .* ...
                   (refl(5).^nFloor) .* (refl(6).^nCeil);

        amp = reflCoef ./ (4*pi*dist + eps);
        delay_smp = dist / c * fs;
        sidx = round(delay_smp) + 1;

        valid = sidx >= 1 & sidx <= irLen;
        for jj = find(valid(:))'
            ir(rr, sidx(jj)) = ir(rr, sidx(jj)) + amp(jj);
        end
    end
end


function rt60 = estimate_rt60(ir, fs)
    ir = ir(:);
    edc = cumsum(ir(end:-1:1).^2);
    edc = edc(end:-1:1);
    edc_db = 10 * log10(edc / max(edc) + eps);

    i5  = find(edc_db <= -5, 1, 'first');
    i25 = find(edc_db <= -25, 1, 'first');

    if isempty(i5) || isempty(i25) || i25 <= i5
        rt60 = NaN;
        return;
    end

    t = (0:length(edc_db)-1)' / fs;
    p = polyfit(t(i5:i25), edc_db(i5:i25), 1);

    if p(1) >= 0
        rt60 = NaN;
        return;
    end
    rt60 = -60 / p(1);
end
