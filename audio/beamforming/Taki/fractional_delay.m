function y = fractional_delay(x, delay_sec, fs)
% Applies fractional delay using frequency-domain phase shift

    N = length(x);
    X = fft(x);

    freqs = (0:N-1).' * fs / N;
    phase_shift = exp(-1j * 2*pi * freqs * delay_sec);

    y = real(ifft(X .* phase_shift));
end
