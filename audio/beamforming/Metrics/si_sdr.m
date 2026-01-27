function val = si_sdr(y, s)
% Scale-Invariant SDR (BSS Eval v4 style)

y = y(:);
s = s(:);

s = s - mean(s);
y = y - mean(y);

alpha = (s' * y) / (s' * s);
s_target = alpha * s;
e_noise  = y - s_target;

val = 10 * log10( sum(s_target.^2) / (sum(e_noise.^2) + 1e-9) );
end
