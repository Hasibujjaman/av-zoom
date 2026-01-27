function Rxx = init_covariance(numFreqs, numMics)
% Initialize spatial covariance matrices
% Rxx: [numMics x numMics x numFreqs]

Rxx = zeros(numMics, numMics, numFreqs);

for k = 1:numFreqs
    Rxx(:,:,k) = eye(numMics);
end

end
