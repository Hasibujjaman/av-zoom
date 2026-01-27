function Rxx = update_covariance(Rxx, X_frame, alpha)
% Recursive EMA update of spatial covariance
%
% Inputs:
%   Rxx     : [numMics x numMics x numFreqs]
%   X_frame : [numFreqs x numMics]  (one STFT frame)
%   alpha   : smoothing factor (0.95–0.99)
%
% Output:
%   Rxx     : updated covariance

numFreqs = size(X_frame,1);

for k = 1:numFreqs
    xk = X_frame(k,:).';          % [numMics x 1]
    Rxx(:,:,k) = alpha * Rxx(:,:,k) ...
               + (1-alpha) * (xk * xk');
end

end
