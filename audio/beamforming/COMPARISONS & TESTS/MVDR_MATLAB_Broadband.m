clear; clc;

addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Taki');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Metrics');
addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/');


%% 1. System Parameters
fs = 16000;       % Sample rate (Hz)
c = 343;          % Speed of sound (m/s)

%% 2. Load Speech Signal
% Replace this with your actual file: [speech, ~] = audioread('your_audio.wav');
% For this test, we generate a synthetic speech-like signal (filtered noise)
len = 3 * fs;     % 3 seconds
t = (0:len-1)'/fs;
src_speech = sign(sin(2*pi*150*t)) .* exp(-5*t); % Synthetic pulsed speech
src_speech = src_speech + 0.1*randn(size(src_speech)); % Add texture

%% 3. Define Microphone Array (X-Axis)
% We align the array along the X-axis.
micNum = 2;
% Spacing: < lambda/2 at highest freq (8kHz) to prevent aliasing
% lambda at 8kHz is ~4.3cm. Let's use 4cm spacing.
micSpacing = 0.08; 

array = phased.ULA('NumElements', micNum, ...
                   'ElementSpacing', micSpacing, ...
                   'ArrayAxis', 'x'); % CRITICAL: Array is on X-axis

%% 4. Simulate the Acoustic Environment
% We need to create the multi-channel "received" signal.
% Since the source is broadside (90 deg) to an X-axis array, 
% the signal actually arrives at all mics at the same time (Delay = 0).
% Interference at 30 deg will have delays.

% Target angle: +ve Y-axis is 90 degrees Azimuth in MATLAB convention
targetAngle = [90; 0]; 
interferenceAngle = [40; 0]; % Interference coming from the side

% Use a Wideband Collector to simulate speech delays correctly
collector = phased.WidebandCollector('Sensor', array, ...
    'PropagationSpeed', c, ...
    'SampleRate', fs, ...
    'ModulatedInput', false); % Speech is baseband, not modulated

% Generate Interference (White Noise)
noise_src = 0.5 * randn(size(src_speech));

% Collect signals (Simulate what the mics hear)
% Signal 1: Target Speech from 90 deg
% Signal 2: Noise from 30 deg
sig_mic = collector([src_speech, noise_src], [targetAngle, interferenceAngle]);




% Add a little thermal noise floor
sig_mic = sig_mic + 0.01 * randn(size(sig_mic));
[sig_mic,~] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/multichannel_maleClean_femalePiano.wav');

disp('Simulation created. Playing "Noisy" Mic 1...');
% soundsc(sig_mic(:,1), fs); % Uncomment to hear the noisy input




%% 5. Configure Subband MVDR Beamformer
% This processes the audio in frequency chunks (Subbands)
mvdrSubband = phased.SubbandMVDRBeamformer('SensorArray', array, ...
    'SampleRate', fs, ...
    'PropagationSpeed', c, ...
    'Direction', targetAngle, ... % Look at +Y (90 deg)
    'OperatingFrequency', fs, ... % Max freq
    'NumSubbands', 512, ...          % FFT size equivalent
    'DiagonalLoadingFactor', 1e-2); % Improves robustness

%% 6. Apply Beamformer
disp('Processing...');
y_out = mvdrSubband(sig_mic);

%% 7. Visualize and Compare
t_plot = (0:length(y_out)-1)/fs;
[src_speech, ~] = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav');
t_speech_plot = (0:length(src_speech)-1)/fs;

figure('Name', 'MVDR Speech Beamforming');

subplot(3,1,1);
plot(t_speech_plot, src_speech);
title('Original Clean Speech (Reference)');
ylim([-1.5 1.5]); grid on;

subplot(3,1,2);
plot(t_plot, sig_mic(:,1)); 
title('Microphone 1 Input (Speech + Interference)');
ylim([-1.5 1.5]); grid on;

subplot(3,1,3);
plot(t_plot, real(y_out));
title('MVDR Output (Beamformed)');
ylim([-1.5 1.5]); grid on;

disp('Done! The interference in the 3rd plot should be significantly reduced.');
% soundsc(real(y_out), fs); % Uncomment to hear the result



audiowrite('../Test_output/MVDR_MATLAB_Broadband_1.wav', real(y_out), fs);