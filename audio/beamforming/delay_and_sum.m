function y_das = delay_and_sum(x, theta_deg, micPos, fs, c)
% Simple delay-and-sum beamformer
%
% Inputs:
%   x         : [samples x numMics] multichannel signal
%   theta_deg : steering angle (degrees)
%   micPos   : microphone positions [numMics x 1] (meters)
%   fs       : sampling frequency (Hz)
%   c        : speed of sound (m/s)
%
% Output:
%   y_das    : beamformed time-domain signal

numMics = size(x,2);
numSamples = size(x,1);

theta = deg2rad(theta_deg);

% Compute propagation delays
tau = micPos * sin(theta) / c;   % [seconds]

% Delay-compensated signals
x_aligned = zeros(numSamples, numMics);

for m = 1:numMics
    % Negative sign = compensate delay
    x_aligned(:,m) = delayseq(x(:,m), -tau(m), fs);
end

% Average across microphones
y_das = mean(x_aligned, 2);

end
