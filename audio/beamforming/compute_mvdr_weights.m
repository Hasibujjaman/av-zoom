function W = compute_mvdr_weights(Rxx, d, delta)
% Rxx   : [numMics x numMics x numFreqs]
% d     : [numMics x numFreqs]
% delta : diagonal loading factor (e.g., 1e-3)

[numMics, ~, numFreqs] = size(Rxx);
W = zeros(numMics, numFreqs);

for k = 1:numFreqs
    R = Rxx(:,:,k);
    R = R + delta * trace(R)/numMics * eye(numMics);  % loading
    dk = d(:,k);

    W(:,k) = R \ dk / (dk' * (R \ dk));
end

end
