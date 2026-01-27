function [x, fs_expected] = generate_multichannel_test_signal_v2(voicePath, noisePath, theta_voice, theta_noise, SNR_dB, outputFilename)
% CREATE_MULTICHANNEL_MIXTURE Simulates a 2-microphone array recording.
%
%   Usage:
%   [x, fs] = create_multichannel_mixture(voicePath, noisePath, theta_v, theta_n, snr, outFile)
%
%   Inputs:
%       voicePath     - String path to clean voice wav file
%       noisePath     - String path to noise wav file
%       theta_voice   - DOA of voice (degrees)
%       theta_noise   - DOA of noise (degrees)
%       SNR_dB        - Desired SNR at the reference microphone
%       outputFilename- (Optional) String path to save the output .wav file. 
%                       If empty or omitted, no file is saved.
%
%   Outputs:
%       x             - The resulting multichannel signal (Samples x Mics)
%       fs_expected   - The sampling rate used (16000 Hz)
%
%   Dependencies:
%       Phased Array System Toolbox (for 'delayseq')

    %% ================= ARRAY PARAMETERS =================
    % These can be moved to input arguments if you need variable spacing
    c = 343;             % speed of sound (m/s)
    micSpacing = 0.08;   % 8 cm spacing
    micPos = [0; micSpacing];
    numMics = length(micPos);
    fs_expected = 16000;

    %% ================= LOAD AUDIO =================
    [voice, fs_v] = audioread(voicePath);
    [noise, fs_n] = audioread(noisePath);

    % Check Sampling Rates
    if fs_v ~= fs_expected || fs_n ~= fs_expected
        error('Sampling rate mismatch. Both files must be %d Hz.', fs_expected);
    end

    % Force Mono
    if size(voice, 2) > 1
        voice = mean(voice, 2);
    end
    if size(noise, 2) > 1
        noise = mean(noise, 2);
    end

    %% ================= LENGTH ALIGNMENT =================
    % Truncate to the length of the shorter file
    L = min(length(voice), length(noise));
    voice = voice(1:L);
    noise = noise(1:L);

    %% ================= SNR SCALING =================
    % Normalize both to unit energy first to establish a baseline
    voice = voice / rms(voice);
    noise = noise / rms(noise);
    
    % Scale noise relative to voice to achieve desired SNR
    % SNR_dB = 20 * log10(rms_signal / rms_noise)
    noise = noise * 10^(-SNR_dB/20);

    %% ================= DELAY COMPUTATION =================
    % Calculate time delays based on Far-Field assumption
    % 
    tau_voice = micPos * sin(deg2rad(theta_voice)) / c;
    tau_noise = micPos * sin(deg2rad(theta_noise)) / c;

    %% ================= MULTICHANNEL MIXTURE =================
    x = zeros(L, numMics);
    
    % Apply fractional delays
    for m = 1:numMics
        % Check if delayseq exists (requires Phased Array Toolbox)
        if exist('delayseq', 'file') ~= 2
             error('Function "delayseq" not found. Please install the Phased Array System Toolbox.');
        end
        
        v_delayed = delayseq(voice, tau_voice(m), fs_expected);
        n_delayed = delayseq(noise, tau_noise(m), fs_expected);
        x(:,m) = v_delayed + n_delayed;
    end

    %% ================= NORMALIZATION =================
    % Normalize entire array to prevent clipping, preserving relative levels
    x = x / max(abs(x(:)));

    %% ================= SAVE OUTPUT (OPTIONAL) =================
    if nargin > 5 && ~isempty(outputFilename)
        % Create directory if it doesn't exist
        outputDir = fileparts(outputFilename);
        if ~isempty(outputDir) && ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        
        audiowrite(outputFilename, x, fs_expected);
        fprintf('Saved: %s\n', outputFilename);
    end
end