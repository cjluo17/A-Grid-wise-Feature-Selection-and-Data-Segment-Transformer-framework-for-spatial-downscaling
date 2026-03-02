function FI = compute_rf_importance(data, nTrees)
% data: [samples × variables × grids]
% last variable is target
% FI output: [grids × (variables-1)]

rng(42); 

[numSamples, numVars, numGrids] = size(data);
numFeatures = numVars - 1;

FI = zeros(numGrids, numFeatures);

    parfor i = 1:numGrids
        X = data(:,1:numFeatures,i);
        Y = data(:,numVars,i);

        rfModel = TreeBagger( ...
            nTrees, X, Y, ...
            'Method','regression', ...
            'OOBPrediction','on', ...
            'OOBPredictorImportance','on');

        imp = rfModel.OOBPermutedPredictorDeltaError;

        if sum(imp) > 0
            imp = imp / sum(imp);
        end

        FI(i,:) = imp;
    end
end
