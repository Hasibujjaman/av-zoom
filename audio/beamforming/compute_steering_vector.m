 function d = compute_steering_vector(azimuth_deg, freqs, micPos, c)
 % azimuth_deg : DOA azimuth in degrees (MATLAB convention: 0°=endfire, 90°=broadside)
 % freqs       : frequency vector (Hz)
 % micPos      : mic positions [numMics x 1] (meters), along array axis
 % c           : speed of sound
 
 az = deg2rad(azimuth_deg);
 numFreqs = length(freqs);
 numMics = length(micPos);
 
 d = zeros(numMics, numFreqs);
 
 for k = 1:numFreqs
     omega = 2*pi*freqs(k);
     tau = micPos * cos(az) / c;
     d(:,k) = exp(-1j * omega * tau);
 end
 
 end


