function selectedIdx = gfs_select_features(FI_row, topK)

% GFS feature selector based on RF importance ranking

if topK > length(FI_row)
    error('topK exceeds number of available features.');
end

[~, rankIdx] = sort(FI_row, 'descend');


selectedIdx = rankIdx(1:topK);

end