function x = compute_istft(X, params)

    x = istft(X, ...
        'Window', params.window, ...
        'OverlapLength', params.frameLen - params.hopLen, ...
        'FFTLength', params.nfft);

    x = real(x);              % Numerical safety
    x = x - mean(x);          % Remove DC
    x = x / (max(abs(x)) + 1e-6);   % Peak normalize

end
