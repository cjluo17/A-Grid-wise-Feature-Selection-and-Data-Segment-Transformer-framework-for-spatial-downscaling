function bestW = find_best_window(features, target, winList, trainRatio, valRatio)

rng(42);

T = size(features,2);

trainEnd = floor(trainRatio*T);
valEnd   = floor((trainRatio+valRatio)*T);

bestRMSE = inf;
bestW = winList(1);

for w = winList
    
    if T - w < 10
        continue;
    end

    [X_all, Y_all] = DS_Split(features, target, w);

    totalSamples = length(Y_all);
    trainEnd_s = floor(trainRatio * totalSamples);
    valEnd_s   = floor((trainRatio+valRatio)*totalSamples);

    X_train = X_all(1:trainEnd_s);
    Y_train = Y_all(1:trainEnd_s);

    X_val = X_all(trainEnd_s+1:valEnd_s);
    Y_val = Y_all(trainEnd_s+1:valEnd_s);

    net = transformer(X_train, Y_train, size(features,1));

    pred = predict(net, X_val);
    rmse = sqrt(mean((pred - Y_val).^2));

    if rmse < bestRMSE
        bestRMSE = rmse;
        bestW = w;
    end
end
end