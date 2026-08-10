%% Task1_Anechoic_5dB.m
% Generate anechoic 2-mic mixture signals for Task 1 (SP Cup 2026 Phase 2).
%
% Reads source .wav files from the script folder, spatializes via
% fractional delay for a 2-mic ULA, and mixes at SIR=0 dB, SNR=5 dB.
%
% Input files  (same folder as this script):
%   target_signal.wav
%   interference_signal{1,2,3}.wav
%
% Output files:
%   mixture_signal{1,2,3}.wav   — 2-channel stereo mixtures
%   Task1_Anechoic_5dB.mat      — all signals, parameters, metadata

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
duration = 3;           % seconds
L        = fs * duration;

c     = 340;            % speed of sound (m/s)
d_mic = 0.08;           % mic spacing (m)
mic_pos = [-d_mic/2; d_mic/2];

theta_target = 90;      % broadside
theta_interf = 40;      % off-axis
sir_db = 0;
snr_db = 5;

rng(2026);

fprintf('Task 1 — Anechoic mixture generation\n');
fprintf('SIR=%d dB, SNR=%d dB, theta_t=%d, theta_i=%d\n\n', ...
    sir_db, snr_db, theta_target, theta_interf);

%% Load target
target_signal = load_audio(target_path, fs, duration);
fprintf('Target: %s\n', target_path);

%% Generate mixtures
n_examples = 3;
all_target_mc   = cell(n_examples, 1);
all_interf_mc   = cell(n_examples, 1);
all_noise_mc    = cell(n_examples, 1);
all_mixtures    = cell(n_examples, 1);
all_interf_mono = cell(n_examples, 1);

for ex = 1:n_examples
    fprintf('\nExample %d/%d (%s)\n', ex, n_examples, interf_labels{ex});

    interf_signal = load_audio(interf_paths{ex}, fs, duration);

    [mixture, target_mc, interf_mc, noise_mc] = create_anechoic_mixture( ...
        target_signal, interf_signal, theta_target, theta_interf, ...
        fs, sir_db, snr_db, c, d_mic);

    all_mixtures{ex}    = mixture;
    all_target_mc{ex}   = target_mc;
    all_interf_mc{ex}   = interf_mc;
    all_noise_mc{ex}    = noise_mc;
    all_interf_mono{ex} = interf_signal;

    % save mixture wav (2-ch)
    mix_path = fullfile(script_dir, sprintf('mixture_signal%d.wav', ex));
    audiowrite(mix_path, mixture, fs);
    fprintf('  Saved: %s\n', mix_path);

    % save interference wav (mono)
    int_path = fullfile(script_dir, sprintf('interference_signal%d.wav', ex));
    audiowrite(int_path, peak_norm(interf_signal), fs);
end

% save target wav (mono)
audiowrite(fullfile(script_dir, 'target_signal.wav'), peak_norm(target_signal), fs);

%% Save .mat
rir_data = struct();
rir_data.type = 'anechoic';
rir_data.description = 'Free-field, fractional delay spatialization';
rir_data.target_delays_samples = mic_pos * cosd(theta_target) / c * fs;
rir_data.interf_delays_samples = mic_pos * cosd(theta_interf) / c * fs;

params = struct();
params.fs               = fs;
params.duration_s       = duration;
params.n_mics           = 2;
params.mic_spacing_m    = d_mic;
params.mic_positions    = mic_pos;
params.speed_of_sound   = c;
params.theta_target_deg = theta_target;
params.theta_interf_deg = theta_interf;
params.sir_db           = sir_db;
params.snr_db           = snr_db;
params.condition        = 'Anechoic';
params.task             = 'Task 1';
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

mat_path = fullfile(script_dir, 'Task1_Anechoic_5dB.mat');
save(mat_path, 'target_signal_out', 'interference_signal', 'mixture_signal', ...
    'rir_data', 'params', 'examples', '-v7.3');

target_signal = target_signal_out;                    
save(mat_path, 'target_signal', '-append');

fprintf('\nSaved: %s\n', mat_path);
fprintf('Done. Run process_task1.m next.\n');


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
    create_anechoic_mixture(target, interf, theta_t, theta_i, fs, sir_db, snr_db, c, d)

    micPos = [-d/2; d/2];
    Lsig = min(length(target), length(interf));
    target = target(1:Lsig);
    interf = interf(1:Lsig);

    % fractional-delay spatialization
    frac_delay = @(x, tau) interp1((0:length(x)-1)', x, ...
        (0:length(x)-1)' - tau*fs, 'linear', 0);
    make_mc = @(sig, az) [frac_delay(sig, micPos(1)*cosd(az)/c), ...
                          frac_delay(sig, micPos(2)*cosd(az)/c)];

    target_mc = make_mc(target, theta_t);
    interf_mc = make_mc(interf, theta_i);

    % SIR scaling
    Pt = mean(target_mc(:).^2) + eps;
    Pi = mean(interf_mc(:).^2) + eps;
    interf_mc = interf_mc * sqrt(Pt / (Pi * 10^(sir_db/10)));

    % add AWGN at target SNR
    mixture_clean = target_mc + interf_mc;
    Ps = mean(mixture_clean(:).^2) + eps;
    noise_mc = sqrt(Ps / 10^(snr_db/10)) * randn(size(mixture_clean));
    mixture = mixture_clean + noise_mc;

    % peak-normalize (preserves relative levels)
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
