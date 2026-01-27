function y = reconstruct_taki(Y, params, sigLen, padLen)
% Y       : [F x T]
% params  : from compute_stft
% sigLen  : original signal length
% padLen  : padding used before STFT

y_full = compute_istft(Y, params);

% Remove padding
y = real(y_full(padLen+1 : padLen+sigLen));

end
