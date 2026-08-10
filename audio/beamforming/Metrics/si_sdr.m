function val = si_sdr(y, s)
% SI_SDR  Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al., 2019)
%
%   val = si_sdr(estimate, reference)
%
%   y : estimated (enhanced) signal
%   s : reference (clean) signal
%
%   val : SI-SDR in dB
%
% The projection is:  alpha = (s'*y)/(s'*s),  s_target = alpha*s
% SI-SDR = 10*log10( ||s_target||^2 / ||y - s_target||^2 )
%
% Note: SI-SDR is symmetric in the sense that swapping y,s gives the
%       same numerical value (the projection ratio is the same), but
%       by convention y = estimate, s = reference.

y = y(:);
s = s(:);

s = s - mean(s);
y = y - mean(y);

alpha = (s' * y) / (s' * s);
s_target = alpha * s;
e_noise  = y - s_target;

val = 10 * log10( sum(s_target.^2) / (sum(e_noise.^2) + 1e-9) );
end
