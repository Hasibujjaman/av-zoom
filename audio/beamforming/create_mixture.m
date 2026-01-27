% function mixture = create_mixture(target, interf, theta_target, ...
%                                     theta_interf, fs, SNR_dB, c, d)
% 
% micPos = [-d/2; d/2];             % x-axis (meters)
% 
% L = min(length(target), length(interf));
% target = target(1:L);
% interf = interf(1:L);
% 
% apply_delay = @(x, tau) interp1( ...
%     (0:length(x)-1)', x, ...
%     (0:length(x)-1)' - tau*fs, ...
%     'linear', 0);
% 
% % Plane-wave model
% make_multichannel = @(sig, theta) ...
%     [ apply_delay(sig, micPos(1)*sin(deg2rad(theta))/c), ...
%       apply_delay(sig, micPos(2)*sin(deg2rad(theta))/c) ];
% 
% target_mc = make_multichannel(target, theta_target);
% interf_mc = make_multichannel(interf, theta_interf);
% 
% mixture = target_mc + interf_mc;
% end



function [mixture, target_mc, interf_mc, noise_mc] = ...
    create_mixture(target, interf, theta_target, ...
                   theta_interf, fs, SNR_dB, c, d)

% ---------------- Geometry ----------------
micPos = [-d/2; d/2];     % x-axis (meters)

% ---------------- Length match ----------------
L = min(length(target), length(interf));
target = target(1:L);
interf = interf(1:L);

% ---------------- Fractional delay ----------------
apply_delay = @(x, tau) interp1( ...
    (0:length(x)-1)', x, ...
    (0:length(x)-1)' - tau*fs, ...
    'linear', 0);

% ---------------- Plane-wave model ----------------
make_multichannel = @(sig, theta) ...
    [ apply_delay(sig, micPos(1)*sin(deg2rad(theta))/c), ...
      apply_delay(sig, micPos(2)*sin(deg2rad(theta))/c) ];

target_mc = make_multichannel(target, theta_target);
interf_mc = make_multichannel(interf, theta_interf);

% =================================================
%            SIR CONTROL (0 dB)
% =================================================
Pt = mean(target_mc(:,1).^2);
Pi = mean(interf_mc(:,1).^2);

interf_mc = interf_mc * sqrt(Pt / Pi);   % SIR = 0 dB

% =================================================
%            CLEAN MIXTURE
% =================================================
mixture_clean = target_mc + interf_mc;

% =================================================
%            ADD WHITE GAUSSIAN NOISE
% =================================================
mixture = zeros(size(mixture_clean));
noise_mc = zeros(size(mixture_clean));

for m = 1:size(mixture_clean,2)
    mixture(:,m) = awgn(mixture_clean(:,m), SNR_dB, 'measured');
    noise_mc(:,m) = mixture(:,m) - mixture_clean(:,m);
end

end
