function outputdata = ResidualCorrect(inputlowresdata, inputhighresdata, EWH)

CountM = size(inputlowresdata,3);

[nHigh, ~, ~] = size(inputhighresdata);
outputdata = zeros(nHigh,3,CountM);
Interpo = zeros(size(EWH,1),3,CountM);

for i = 1:CountM
    Interpo(:,3,i) = EWH(:,3,i) - inputlowresdata(:,3,i);
end
Interpo(:,1:2,:) = EWH(:,1:2,1:CountM);

for i = 1:CountM
    
    xy = Interpo(:,1:3,i);
    x = xy(:,1);
    y = xy(:,2);
    residual_low = xy(:,3);

    xyxy = inputhighresdata(:,1:3,i);
    xi = xyxy(:,1);
    yi = xyxy(:,2);
    pred_high = xyxy(:,3);

    InterpoFactor = griddata(x, y, residual_low, xi, yi, 'nearest');

    corrected = pred_high + InterpoFactor;

    outputdata(:,3,i) = corrected;
    outputdata(:,1:2,i) = xyxy(:,1:2);
end

end