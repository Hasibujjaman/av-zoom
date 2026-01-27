function osinr_db = compute_osinr(s, v)
% Output Signal-to-Interference-plus-Noise Ratio

s = s(:);
v = v(:);

Ps = mean(s.^2);
Pv = mean(v.^2) + 1e-9;

osinr_db = 10 * log10(Ps / Pv);
end
