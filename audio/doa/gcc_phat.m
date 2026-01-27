function tau = gcc_phat(X1, X2, fs)

    R = X2 .* conj(X1);
    R = R ./ (abs(R) + eps);

    cc = real(ifft(R));

    % Shift zero lag to center
    cc = fftshift(cc);

    % Time lag axis
    N = length(cc);
    lags = (-N/2:N/2-1) / fs;

    % Peak location
    [~, idx] = max(cc);
    tau = lags(idx);
end
