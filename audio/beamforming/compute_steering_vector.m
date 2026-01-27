 function d = compute_steering_vector(theta_deg, freqs, micPos, c)
 % theta_deg : DOA in degrees (broadside = 0)
 % freqs     : frequency vector (Hz)
 % micPos    : mic positions [numMics x 1] (meters)
 % c         : speed of sound
 
 theta = deg2rad(theta_deg);
 numFreqs = length(freqs);
 numMics = length(micPos);
 
 d = zeros(numMics, numFreqs);
 
 for k = 1:numFreqs
     omega = 2*pi*freqs(k);
    tau = micPos * sin(theta) / c;
     d(:,k) = exp(-1j * omega * tau);
 end
 
 end


