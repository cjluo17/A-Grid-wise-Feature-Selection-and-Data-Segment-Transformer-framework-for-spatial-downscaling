function [TransHighResEWH, TransLowResEWH] = Modeling( ...
    LRF, HRF, LREWH, LowResCoords, HighResCoords, Region)

rng(42);


winList = [4 8 16 32 64 128 256];
trainRatio = 0.7;
valRatio   = 0.15;
zero_idx   = 4;

[NormLowResFeatures, ~, ~]  = normalize_data(LRF);
[NormHighResFeatures, ~, ~] = normalize_data(HRF);
[NormLowResFusionEWH, mu, sigma] = normalize_data(LREWH);

month        = size(LRF,2);
NumLowGrid   = size(NormLowResFeatures,1);
NumHighGrid  = size(NormHighResFeatures,1);

TransModel       = cell(NumLowGrid,1);
BestWindowRecord = zeros(NumLowGrid,1);

TransPred        = zeros(NumHighGrid,month);
TransLowResPred  = zeros(NumLowGrid,month);

parfor g = 1:NumLowGrid
    lowFeat = squeeze(NormLowResFeatures(g,:,:))';

    selectedIdx = gfs_select_features(Region(g,:), zero_idx);
	lowFeatSel = lowFeat(selectedIdx, :);

    NumFeatures = size(lowFeatSel,1);

    bestW = find_best_window( ...
        lowFeatSel, ...
        NormLowResFusionEWH(g,:), ...
        winList, ...
        trainRatio, ...
        valRatio);

    BestWindowRecord(g) = bestW;

    [X_train, Y_train] = DS_Split( lowFeatSel, NormLowResFusionEWH(g,:), bestW);

    TransModel{g} = transformer( ...
        X_train, Y_train, NumFeatures);

    if mod(g,50)==0 || g==NumLowGrid
        fprintf('Low-res modeling: %d / %d\n', g, NumLowGrid);
    end
end

nearestLowResIdx_all = knnsearch(LowResCoords, HighResCoords);

parfor i = 1:NumHighGrid

    lowIdx = nearestLowResIdx_all(i);
    bestW  = BestWindowRecord(lowIdx);

    highFeat = squeeze(NormHighResFeatures(i,:,:))';

    selectedIdx = gfs_select_features(Region(lowIdx,:), zero_idx);
	highFeatSel = highFeat(selectedIdx, :);

    [X_pred, ~] = DS_Split(highFeatSel, zeros(1,size(highFeatSel,2)), bestW);

    pred = predict(TransModel{lowIdx}, X_pred);

    padLen = month - length(pred);
    TransPred(i,:) = [nan(1,padLen), pred'];

    if mod(i,200)==0 || i==NumHighGrid
        fprintf('High-res prediction: %d / %d\n', i, NumHighGrid);
    end
end


parfor i = 1:NumLowGrid

    bestW = BestWindowRecord(i);

    lowFeat = squeeze(NormLowResFeatures(i,:,:))';

    selectedIdx = gfs_select_features(Region(i,:), zero_idx);
	lowFeatSel  = lowFeat(selectedIdx, :);

    [X_pred, ~] = DS_Split(lowFeatSel, zeros(1,size(lowFeatSel,2)), bestW);

    pred = predict(TransModel{i}, X_pred);

    padLen = month - length(pred);
    TransLowResPred(i,:) = [nan(1,padLen), pred'];

    if mod(i,50)==0 || i==NumLowGrid
        fprintf('Low-res prediction: %d / %d\n', i, NumLowGrid);
    end
end

TransPred       = TransPred * sigma + mu;
TransLowResPred = TransLowResPred * sigma + mu;

TransHighResEWH = zeros(NumHighGrid,3,month);
TransHighResEWH(:,1:2,:) = repmat(HighResCoords,[1 1 month]);
TransHighResEWH(:,3,:)   = reshape(TransPred,[NumHighGrid,1,month]);

TransLowResEWH = zeros(NumLowGrid,3,month);
TransLowResEWH(:,1:2,:) = repmat(LowResCoords,[1 1 month]);
TransLowResEWH(:,3,:)   = reshape(TransLowResPred,[NumLowGrid,1,month]);

end