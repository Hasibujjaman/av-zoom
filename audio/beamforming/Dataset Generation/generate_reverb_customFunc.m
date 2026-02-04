%% Room Impulse Response Synthesis for Microphone Array
% Room Setup
clear; clc; close all;

SNR_dB = 5;  % 5 dB SNR as specified
% Room dimensions (shoebox)
roomDimensions = [4.9, 4.9, 4.9]; % [Lx, Ly, Lz] in meters

% Microphone positions (2-microphone linear array)
mic1 = [2.41, 2.45, 1.5];  % Mic 1
mic2 = [2.49, 2.45, 1.5];  % Mic 2
rx = [mic1; mic2];         % Receiver array

% Source positions
source1 = [2.45, 3.45, 1.5];  % Target (0° azimuth)
source2 = [3.22, 3.06, 1.5];  % Interference (40° azimuth)

%% Step 1: Calculate absorption coefficient for RT60 ≈ 0.5 s
% Using Sabine's formula: RT60 = 0.161 * V / (A * S)
% Where: V = volume, S = total surface area, A = average absorption coefficient

V = prod(roomDimensions);  % Room volume
S = 2*(roomDimensions(1)*roomDimensions(2) + ...
       roomDimensions(1)*roomDimensions(3) + ...
       roomDimensions(2)*roomDimensions(3)); % Total surface area

RT60_target = 0.5;  % Target reverberation time
alpha_sabine = 0.161 * V / (RT60_target * S); % Required absorption coefficient

% Convert to absorption coefficient (simplified approach)
% Note: For more accuracy, we might need frequency-dependent coefficients
% We'll use the same absorption for all surfaces initially
% Using Sabine formula: alpha ≈ 0.263 for RT60 = 0.5s in this room
absorption_coefficient = alpha_sabine;  % Use calculated value (~0.263)

% Compute reflection coefficient from absorption coefficient
% reflection_coeff = sqrt(1 - alpha)
reflection_coefficient = sqrt(1 - absorption_coefficient);
fprintf('Calculated absorption coefficient: %.4f\n', absorption_coefficient);
fprintf('Calculated reflection coefficient: %.4f\n', reflection_coefficient);

%% Step 2: Generate Room Impulse Responses using acousticRoomResponse
% Common parameters
fs = 16000;  % Sampling rate
sound_speed = 340;  % m/s
algorithm = "image-source";  % Using image source method
% Image-source order controls how late the longest reflections can be.
% To realize RT60_target, the RIR must extend beyond RT60_target.
% With a shoebox ISM, a rough upper bound on max path length is ~2*order*min(roomDimensions).
% So choose a minimum order that can reach (RT60_target * c).
image_source_order = 15; % you can still set this manually
min_order_for_rt60 = ceil((RT60_target * sound_speed) / (2 * min(roomDimensions))) + 2; % +2 margin
if image_source_order < min_order_for_rt60
    fprintf('Increasing ImageSourceOrder from %d to %d to support RT60=%.2f s...\n', image_source_order, min_order_for_rt60, RT60_target);
    image_source_order = min_order_for_rt60;
end
ms = 0.05;  % Lower scattering for more specular reflections

% The Sabine-based absorption is a good starting point, but the finite image-source order
% can make the measured RT60 (via Schroeder integration) differ from the target.
% Tune a uniform absorption coefficient (all 6 surfaces) to match the measured RT60.
fprintf('\nTuning absorption coefficient for RT60 target %.2f s (ImageSourceOrder=%d)...\n', RT60_target, image_source_order);
% We search smaller absorptions first. In this simplified ISM, we also allow
% scattering=0 because we treat scattering as an additional energy-loss lever.
absorption_candidates = [0.0 0.0005 0.001 0.002 0.005 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.12 0.14 0.16 0.20 0.24 0.28];
scattering_candidates = [0.0 0.01 0.05 0.2 0.5];

% Prefer (but do not require) an IR somewhat longer than the target RT60.
% (Gives the Schroeder decay curve enough tail to estimate RT60 robustly.)
min_ir_duration_s = 1.1 * RT60_target;

best_alpha = absorption_coefficient;
best_rt60 = NaN;
best_err = Inf;
best_ms = ms;
best_ir_dur = 0;

for si = 1:numel(scattering_candidates)
    s = scattering_candidates(si);
    for ai = 1:numel(absorption_candidates)
        a = absorption_candidates(ai);
        ir_test = acousticRoomResponse(roomDimensions, source1, rx(1,:), ...
            'SampleRate', fs, ...
            'SoundSpeed', sound_speed, ...
            'Algorithm', algorithm, ...
            'ImageSourceOrder', image_source_order, ...
            'MaterialAbsorption', a * ones(6,1), ...
            'MaterialScattering', s * ones(6,1));

        ir_dur = size(ir_test,2) / fs;
        rt = estimate_rt60(ir_test(1,:), fs);
        err = abs(rt - RT60_target);

        % Prefer solutions that produce a sufficiently long IR.
        if ir_dur >= min_ir_duration_s
            if ~isnan(rt) && err < best_err
                best_err = err;
                best_alpha = a;
                best_ms = s;
                best_rt60 = rt;
                best_ir_dur = ir_dur;
            end
        else
            % If nothing meets the duration requirement, keep the longest IR we find.
            if best_ir_dur < min_ir_duration_s && ir_dur > best_ir_dur
                best_alpha = a;
                best_ms = s;
                best_rt60 = rt;
                best_ir_dur = ir_dur;
                best_err = err;
            end
        end

        % Early exit if we're close enough and IR is long enough.
        if best_ir_dur >= min_ir_duration_s && best_err <= 0.03
            break;
        end
    end
    if best_ir_dur >= min_ir_duration_s && best_err <= 0.03
        break;
    end
end

% Fine search around the best absorption found (keep best scattering).
% This helps dial in RT60 without bumping ImageSourceOrder again.
if ~isnan(best_rt60)
    fine_span = 0.03;
    fine_step = 0.0025;
    fine_a = max(0, best_alpha - fine_span) : fine_step : min(0.95, best_alpha + fine_span);

    for ai = 1:numel(fine_a)
        a = fine_a(ai);
        ir_test = acousticRoomResponse(roomDimensions, source1, rx(1,:), ...
            'SampleRate', fs, ...
            'SoundSpeed', sound_speed, ...
            'Algorithm', algorithm, ...
            'ImageSourceOrder', image_source_order, ...
            'MaterialAbsorption', a * ones(6,1), ...
            'MaterialScattering', best_ms * ones(6,1));

        ir_dur = size(ir_test,2) / fs;
        rt = estimate_rt60(ir_test(1,:), fs);
        err = abs(rt - RT60_target);

        if ir_dur >= min_ir_duration_s
            if ~isnan(rt) && err < best_err
                best_err = err;
                best_alpha = a;
                best_rt60 = rt;
                best_ir_dur = ir_dur;
            end
        end

        if best_err <= 0.01
            break;
        end
    end
end

absorption_coefficient = best_alpha;
reflection_coefficient = sqrt(1 - absorption_coefficient);
ms = best_ms;
fprintf('Tuned absorption coefficient: %.4f\n', absorption_coefficient);
fprintf('Tuned reflection coefficient: %.4f\n', reflection_coefficient);
fprintf('Tuned scattering coefficient: %.4f\n', ms);
if ~isnan(best_rt60)
    fprintf('Estimated RT60 after tuning (test path): %.3f s\n\n', best_rt60);
else
    fprintf('Estimated RT60 after tuning (test path): NaN (IR did not decay enough within generated length)\n\n');
end

% Generate RIR for Source 1 (Target)
fprintf('Generating RIR for Source 1 (Target)...\n');
ir_source1 = acousticRoomResponse(roomDimensions, source1, rx, ...
    'SampleRate', fs, ...
    'SoundSpeed', sound_speed, ...
    'Algorithm', algorithm, ...
    'ImageSourceOrder', image_source_order, ...
    'MaterialAbsorption', absorption_coefficient * ones(6,1), ...
    'MaterialScattering', ms * ones(6,1)); % Floor, front, back, left, right, ceiling

% Generate RIR for Source 2 (Interference)
fprintf('Generating RIR for Source 2 (Interference)...\n');
ir_source2 = acousticRoomResponse(roomDimensions, source2, rx, ...
    'SampleRate', fs, ...
    'SoundSpeed', sound_speed, ...
    'Algorithm', algorithm, ...
    'ImageSourceOrder', image_source_order, ...
    'MaterialAbsorption', absorption_coefficient * ones(6,1), ...
    'MaterialScattering', ms * ones(6,1));

fprintf('RIR length: %d samples (%.3f s)\n', size(ir_source1,2), size(ir_source1,2)/fs);
if (size(ir_source1,2)/fs) < RT60_target
    fprintf(['WARNING: RIR duration (%.3f s) is shorter than RT60 target (%.3f s).\n' ...
             'With ImageSourceOrder=%d, the image-source model may not generate\n' ...
             'late reflections long enough to realize RT60=%.3f s.\n'], ...
        size(ir_source1,2)/fs, RT60_target, image_source_order, RT60_target);

    fprintf(['Note: To physically realize RT60=%.3f s, the simulated RIR must extend beyond that time.\n' ...
             'Increase ImageSourceOrder (most direct), or accept an extrapolated RT60 estimate.\n\n'], ...
        RT60_target, image_source_order, size(ir_source1,2)/fs);
end
% 
% %% Step 3: Plot Impulse Responses
% t = (0:size(ir_source1,2)-1)/fs;
% 
% figure('Position', [100, 100, 1200, 800]);
% 
% % Source 1 RIRs
% subplot(2,2,1);
% plot(t, ir_source1(1,:));
% title('Source 1 - Mic 1 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,2,2);
% plot(t, ir_source1(2,:));
% title('Source 1 - Mic 2 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% % Source 2 RIRs
% subplot(2,2,3);
% plot(t, ir_source2(1,:));
% title('Source 2 - Mic 1 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,2,4);
% plot(t, ir_source2(2,:));
% title('Source 2 - Mic 2 Impulse Response');
% xlabel('Time (s)'); ylabel('Amplitude');
% grid on;
% 
% sgtitle('Room Impulse Responses (RT60 ≈ 0.5 s)');

%% Step 4: Load and Process Audio Signals
% Load source signals
[target_signal, fs_target] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav');
[interference_signal, fs_interf] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/female_piano_14s.wav');
% Ensure resampling to 16 kHz if needed
if fs_target ~= fs
    target_signal = resample(target_signal, fs, fs_target);
end
if fs_interf ~= fs
    interference_signal = resample(interference_signal, fs, fs_interf);
end

%% Step 5: Apply RIRs to signals (Convolution)
% Initialize received signals
mic1_received = zeros(1, length(target_signal) + size(ir_source1,2) - 1);
mic2_received = zeros(1, length(target_signal) + size(ir_source1,2) - 1);

% Convolve target signal with RIRs
target_mic1 = conv(target_signal, ir_source1(1,:));
target_mic2 = conv(target_signal, ir_source1(2,:));

% Convolve interference signal with RIRs
interf_mic1 = conv(interference_signal, ir_source2(1,:));
interf_mic2 = conv(interference_signal, ir_source2(2,:));

% Trim to same length
min_len = min([length(target_mic1), length(interf_mic1), length(mic1_received)]);
target_mic1 = target_mic1(1:min_len);
target_mic2 = target_mic2(1:min_len);
interf_mic1 = interf_mic1(1:min_len);
interf_mic2 = interf_mic2(1:min_len);
mic1_received = mic1_received(1:min_len);
mic2_received = mic2_received(1:min_len);

% Adjust SIR (0 dB - equal power)
P_target_mic1 = mean(target_mic1.^2);
P_interf_mic1 = mean(interf_mic1.^2);
P_target_mic2 = mean(target_mic2.^2);
P_interf_mic2 = mean(interf_mic2.^2);

scale_mic1 = sqrt(P_target_mic1 / P_interf_mic1);
scale_mic2 = sqrt(P_target_mic2 / P_interf_mic2);

interf_mic1 = interf_mic1 * scale_mic1;
interf_mic2 = interf_mic2 * scale_mic2;

% Mix signals
mic1_received = target_mic1 + interf_mic1;
mic2_received = target_mic2 + interf_mic2;

%% Step 6: Add Sensor Noise (SNR = 5 dB)
% Calculate signal power
P_signal_mic1 = mean(mic1_received.^2);
P_signal_mic2 = mean(mic2_received.^2);

% Calculate required noise power for SNR = 5 dB
SNR_linear = 10^(SNR_dB/10);  % Convert dB to linear
P_noise_mic1 = P_signal_mic1 / SNR_linear;
P_noise_mic2 = P_signal_mic2 / SNR_linear;

% Generate white Gaussian noise
noise_mic1 = sqrt(P_noise_mic1) * randn(size(mic1_received));
noise_mic2 = sqrt(P_noise_mic2) * randn(size(mic2_received));

% Add noise to signals
mic1_noisy = mic1_received + noise_mic1;
mic2_noisy = mic2_received + noise_mic2;

%% Step 7: Verify Actual RT60 (Schroeder Integration)
function rt60 = estimate_rt60(ir, fs)
    ir = ir(:);

    % Schroeder backward integration (Energy Decay Curve)
    energy = cumsum(ir(end:-1:1).^2);
    energy = energy(end:-1:1);
    energy = energy / max(energy + eps);
    energy_db = 10*log10(energy + eps);

    idx_5db = find(energy_db <= -5, 1, 'first');
    if isempty(idx_5db)
        rt60 = NaN;
        return;
    end

    % Prefer RT30 (-5 to -35), fall back to RT20 (-5 to -25) or RT10 (-5 to -15)
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

function ir = acousticRoomResponse(room, tx, rx, varargin)
% Local fallback implementation of acousticRoomResponse for shoebox rooms.
% Implements a deterministic image-source method (ISM) with an order limit.
%
% NOTE:
% - This shadows MathWorks acousticRoomResponse *within this file only*.
% - Supports Algorithm="image-source" (only), plus a subset of Name=Value args.

    p = inputParser;
    p.FunctionName = 'acousticRoomResponse (local ISM)';

    addRequired(p, 'room', @(x) isnumeric(x) && isvector(x) && numel(x) == 3);
    addRequired(p, 'tx', @(x) isnumeric(x) && isequal(size(x), [1, 3]));
    addRequired(p, 'rx', @(x) isnumeric(x) && size(x,2) == 3);

    addParameter(p, 'SampleRate', 16000, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'SoundSpeed', 343, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Algorithm', "image-source", @(x) (isstring(x) || ischar(x)));
    addParameter(p, 'ImageSourceOrder', 3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'MaterialAbsorption', 0.2, @(x) isnumeric(x) && all(x(:) >= 0) && all(x(:) <= 1));
    addParameter(p, 'MaterialScattering', 0.0, @(x) isnumeric(x) && all(x(:) >= 0) && all(x(:) <= 1));

    parse(p, room, tx, rx, varargin{:});
    args = p.Results;

    alg = string(args.Algorithm);
    if lower(alg) ~= "image-source"
        error('This local implementation supports Algorithm="image-source" only.');
    end

    fs_local = args.SampleRate;
    c = args.SoundSpeed;
    order = round(args.ImageSourceOrder);

    L = double(args.room(:)).';
    tx = double(tx);
    rx = double(rx);

    % Surface order: floor, front, back, left, right, ceiling
    alpha = args.MaterialAbsorption;
    if isscalar(alpha)
        alpha = repmat(alpha, 6, 1);
    else
        alpha = alpha(:);
        if numel(alpha) ~= 6
            error('MaterialAbsorption must be a scalar or a 6-element vector for shoebox rooms.');
        end
    end

    scatter = args.MaterialScattering;
    if isscalar(scatter)
        scatter = repmat(scatter, 6, 1);
    else
        scatter = scatter(:);
        if numel(scatter) ~= 6
            error('MaterialScattering must be a scalar or a 6-element vector for shoebox rooms.');
        end
    end

    % Simple effective reflection model: specular reflection reduced by scattering.
    % (Scattering doesn't strictly remove energy, but this provides a practical tuning lever.)
    refl = sqrt(max(0, 1 - alpha)) .* sqrt(max(0, 1 - scatter));
    % refl order: [floor, front, back, left, right, ceiling]

    validateInRoom(tx, L, 'tx');
    for rxi = 1:size(rx,1)
        validateInRoom(rx(rxi,:), L, sprintf('rx(%d,:)', rxi));
    end

    % Vectorized standard shoebox ISM:
    % Use integer image indices nx,ny,nz with |nx|+|ny|+|nz| <= order.
    n = -order:order;
    [Nx, Ny, Nz] = ndgrid(n, n, n);
    keep = (abs(Nx) + abs(Ny) + abs(Nz)) <= order;
    Nx = Nx(keep);
    Ny = Ny(keep);
    Nz = Nz(keep);

    % Image positions (Allen & Berkley style)
    Ximg = ((-1) .^ Nx) .* tx(1) + 2 .* Nx .* L(1);
    Yimg = ((-1) .^ Ny) .* tx(2) + 2 .* Ny .* L(2);
    Zimg = ((-1) .^ Nz) .* tx(3) + 2 .* Nz .* L(3);

    % Reflection counts per surface derived from the image indices.
    % Surface order: floor(1), front(2), back(3), left(4), right(5), ceiling(6)
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

    reflGain = (refl(4) .^ nLeft) .* (refl(5) .^ nRight) .* ...
               (refl(2) .^ nFront) .* (refl(3) .^ nBack) .* ...
               (refl(1) .^ nFloor) .* (refl(6) .^ nCeil);

    % For each receiver, compute distances and accumulate delayed impulses.
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

        sFloat = (dist ./ c) .* fs_local;
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

% Estimate RT60 for each channel
rt60_estimates = zeros(4,1);
rt60_estimates(1) = estimate_rt60(ir_source1(1,:), fs);
rt60_estimates(2) = estimate_rt60(ir_source1(2,:), fs);
rt60_estimates(3) = estimate_rt60(ir_source2(1,:), fs);
rt60_estimates(4) = estimate_rt60(ir_source2(2,:), fs);

fprintf('\nRT60 Estimates:\n');
fprintf('Source1-Mic1: %.3f s\n', rt60_estimates(1));
fprintf('Source1-Mic2: %.3f s\n', rt60_estimates(2));
fprintf('Source2-Mic1: %.3f s\n', rt60_estimates(3));
fprintf('Source2-Mic2: %.3f s\n', rt60_estimates(4));
fprintf('Average: %.3f s (Target: 0.5 s)\n', mean(rt60_estimates));

% Normalize signals to prevent clipping (scale to [-1, 1] range)
max_amp = max([max(abs(mic1_noisy(:))), max(abs(mic2_noisy(:)))]);
if max_amp > 0
    mic1_normalized = mic1_noisy / max_amp * 0.95;  % Leave 5% headroom
    mic2_normalized = mic2_noisy / max_amp * 0.95;
else
    mic1_normalized = mic1_noisy;
    mic2_normalized = mic2_noisy;
end

stereo_output(:,1) = mic1_normalized(:);
stereo_output(:,2) = mic2_normalized(:);

% Save received audio (normalized to prevent clipping)
audiowrite('mic1_received.wav', mic1_normalized(:), fs);
audiowrite('mic2_received.wav', mic2_normalized(:), fs);
audiowrite('stereo_output.wav', stereo_output, fs);

% Save RIR data and configuration
save('room_impulse_responses.mat', ...
    'ir_source1', 'ir_source2', 'fs', 'roomDimensions', 'rx', 'source1', 'source2', ...
    'RT60_target', 'alpha_sabine', 'absorption_coefficient', 'reflection_coefficient', ...
    'ms', 'image_source_order', 'algorithm');

fprintf('\nMax amplitude before normalization: %.4f\n', max_amp);

fprintf('\nSimulation complete!\n');
fprintf('Files saved:\n');
fprintf('  - room_impulse_responses.mat (RIR data)\n');
fprintf('  - mic1_received.wav (Mic 1 audio)\n');
fprintf('  - mic2_received.wav (Mic 2 audio)\n');
fprintf('  - stereo_output.wav\n');