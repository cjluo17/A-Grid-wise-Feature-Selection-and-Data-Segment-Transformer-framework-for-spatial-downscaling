function net = transformer(Lp_train, t_train, L)

rng(42);
gpurng(42);

layers = [ 
    sequenceInputLayer(L,Name="input")
    positionEmbeddingLayer(L,128,Name="pos-emb")
    additionLayer(2, Name="add")
    selfAttentionLayer(2,64,'AttentionMask','causal')
    selfAttentionLayer(2,64)
    indexing1dLayer("last")
    fullyConnectedLayer(1)
    regressionLayer];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,"input","add/in2");

options = trainingOptions('adam', ...   
    'MaxEpochs', 400, ...                
    'InitialLearnRate', 5e-4, ...        
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropFactor', 0.2, ...    
    'LearnRateDropPeriod', 50, ...   
    'Shuffle', 'every-epoch', ...   
    'MiniBatchSize',128,...
    'L2Regularization',0.001,...
    'Verbose', false);

net = trainNetwork(Lp_train, t_train, lgraph, options);
end