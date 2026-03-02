function [X, Y] = DS_Split(features, target, win)
% features: [numFeatures × T]
% target:   [1 × T]
% win: window length

T = size(features, 2);
numSamples = T - win;

X = cell(numSamples,1);
Y = zeros(numSamples,1);

for i = 1:numSamples
    X{i} = features(:, i:i+win-1);
    Y(i) = target(i+win);
end
end