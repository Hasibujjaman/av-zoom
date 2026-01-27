function y = mix_clean_and_noise(clean, noise, snr_db)

    clean = clean(:);
    noise = noise(:);

    % Match lengths
    if length(noise) < length(clean)
        noise = repmat(noise, ceil(length(clean)/length(noise)), 1);
    end
    noise = noise(1:length(clean));

    % Remove DC
    clean = clean - mean(clean);
    noise = noise - mean(noise);

    % Scale noise to desired SNR (KEEP clean untouched)
    clean_rms = rms(clean);
    noise = noise / rms(noise);
    noise = noise * (clean_rms / (10^(snr_db/20)));

    y = clean + noise;

    % Safety limiter
    y = y / max(abs(y) + 1e-6);
end
