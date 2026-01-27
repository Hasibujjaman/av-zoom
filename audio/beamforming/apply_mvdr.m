function Y = apply_mvdr(X, W)
% X : [numFreqs x numFrames x numMics]
% W : [numMics x numFreqs]

[numFreqs, numFrames, ~] = size(X);
Y = zeros(numFreqs, numFrames);

for n = 1:numFrames
    for k = 1:numFreqs
        xk = squeeze(X(k,n,:));
        Y(k,n) = W(:,k)' * xk;
    end
end

end
