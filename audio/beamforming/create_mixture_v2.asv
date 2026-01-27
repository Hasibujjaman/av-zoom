function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_mixture(target, interf, theta_target, ...
                   theta_interf, fs, SNR_dB, c, d)

% =================================================
% Geometry: 2-mic linear array
% =================================================
micPos = [-d/2; d/2];     % meters (x-axis)

% =================================================
% Length match
% =================================================
L = min(length(target), length(interf));
target = target(1:L);
interf = interf(1:L);

% =================================================
% Fractional delay (plane wave)
% =================================================
apply_delay = @(x, tau) interp1( ...
    (0:length(x)-1)', x, ...
    (0:length(x)-1)' - tau*fs, ...
    'linear', 0);

make_multichannel = @(sig, theta) ...
    [ apply_delay(sig, micPos(1)*sin(deg2rad(theta))/c), ...
      apply_delay(sig, micPos(2)*sin(deg2rad(theta))/c) ];

% =================================================
% Multichannel signals
% =================================================
target_mc = make_multichannel(target, theta_target);
interf_mc = make_multichannel(interf, theta_interf);

% =================================================
% SIR CONTROL (0 dB) — USE BOTH CHANNELS
% =================================================
Pt = mean(target_mc(:).^2);
Pi = mean(interf_mc(:).^2);

interf_mc = interf_mc * sqrt(Pt / Pi);   % SIR = 0 dB

% =================================================
% Clean mixture
% =================================================
mixture_clean = target_mc + interf_mc;

% =================================================
% Add white Gaussian noise (SNR = SNR_dB)
% =================================================
mixture  = zeros(size(mixture_clean));
noise_mc = zeros(size(mixture_clean));

for m = 1:size(mixture_clean,2)
    mixture(:,m) = awgn(mixture_clean(:,m), ...
                        SNR_dB, 'measured');
    noise_mc(:,m) = mixture(:,m) - mixture_clean(:,m);
end

% =================================================
% FINAL NORMALIZATION (CRITICAL)
% Prevent loud hiss & audiowrite clipping
% =================================================
peak = max(abs(mixture(:)));
if peak > 0
    mixture  = 0.99 * mixture  / peak;
    target_mc = 0.99 * target_mc / peak;
    interf_mc = 0.99 * interf_mc / peak;
    noise_mc  = 0.99 * noise_mc  / peak;
end

end
