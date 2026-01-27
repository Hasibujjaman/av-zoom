function [Xf, params] = multichannel_stft_taki(x, fs)
% x : [T x M] multichannel time-domain signal
% Xf: [F x T_frames x M]

[T, M] = size(x);

for m = 1:M
    [Xf(:,:,m), params] = compute_stft(x(:,m), fs);
end

end
