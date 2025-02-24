%% NASA Turbofan Predictive Maintenance System
% High-End Model with Fixed Input Dimensions & Optimized Performance
clear; close all; clc;

%% 1. Configuration
datasetPath = '/Users/jhanvikasundra/Desktop/PredictiveMaintenance/CMAPSSData';
subset = 'FD002';
maxRUL = 145;
sensorCols = 6:26;

%% 2. Data Loading with Validation
try
    fprintf('Loading FD002 training data...\n');
    trainData = readmatrix(fullfile(datasetPath, 'train_FD002.txt'));
    fprintf('Training data dimensions: %dx%d\n', size(trainData));
    fprintf('Loading FD002 test data...\n');
    testData = readmatrix(fullfile(datasetPath, 'test_FD002.txt'));
    fprintf('Test data dimensions: %dx%d\n', size(testData));
    rulData = readmatrix(fullfile(datasetPath, 'RUL_FD002.txt'));
catch ME
    error('Data loading error: %s', ME.message);
end

%% 3. Enhanced Preprocessing
sensorData = trainData(:, sensorCols);
sensorVars = var(sensorData);
validSensors = sensorVars > 0.1;
selectedSensors = sensorCols(validSensors);
fprintf('\nSelected %d/%d sensors with significant variance\n', sum(validSensors), length(validSensors));
[normalizedTrainX, mu, sigma] = zscore(trainData(:, selectedSensors));
normalizedTestX = (testData(:, selectedSensors) - mu) ./ sigma;

unitNumbers = trainData(:, 1);
RUL = zeros(size(trainData, 1), 1);
uniqueUnits = unique(unitNumbers);
for i = 1:length(uniqueUnits)
    unitID = uniqueUnits(i);
    idx = (unitNumbers == unitID);
    maxCycle = max(trainData(idx, 2));
    RUL(idx) = maxCycle - trainData(idx, 2);
end
RUL(RUL > maxRUL) = maxRUL;

%% 4. Sequence Generation
sequenceLength = 30;
sequenceStride = 1;
[XSequences, YSequences] = createSequences(normalizedTrainX, RUL, sequenceLength, sequenceStride);
if any(isnan(YSequences))
    disp('Warning: YSequences contains NaN values');
    validIdx = ~isnan(YSequences);
    XSequences = XSequences(validIdx);
    YSequences = YSequences(validIdx);
end

%% 5. Split Data into Training & Validation Sets
numSamples = numel(YSequences);
idx = randperm(numSamples);
numTrain = floor(0.9 * numSamples);
XTrain = XSequences(idx(1:numTrain));
YTrain = YSequences(idx(1:numTrain));
XVal = XSequences(idx(numTrain+1:end));
YVal = YSequences(idx(numTrain+1:end));

if any(isnan(YTrain))
    validIdx = ~isnan(YTrain);
    XTrain = XTrain(validIdx);
    YTrain = YTrain(validIdx);
end

YTrain = double(YTrain);
fprintf('Training Sequences: %d, Validation Sequences: %d\n', numel(XTrain), numel(XVal));

%% 6. Optimized LSTM Architecture with Overfitting Prevention
numFeatures = size(XTrain{1}, 1);
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(256, 'OutputMode', 'sequence') % Increased units for deeper learning
    batchNormalizationLayer % Improve convergence
    dropoutLayer(0.4) % Increased dropout to prevent overfitting
    lstmLayer(128, 'OutputMode', 'last') % Second LSTM layer with reduced units
    batchNormalizationLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

%% 7. Training Configuration with Dynamic Learning Rate & Early Stopping
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ... % Increased epochs for deeper learning
    'InitialLearnRate', 0.0003, ... % Tuned learning rate
    'MiniBatchSize', 128, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 15, ... % Increased patience to avoid early stopping
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 30, ...
    'L2Regularization', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress', ...
    'Verbose', 1);

%% 8. Model Training
fprintf('\n=== Training Initialization (LSTM) ===\n');
if any(isnan(YTrain))
    error('YTrain contains NaN values after cleaning.');
end
lstmNet = trainNetwork(XTrain, YTrain, layers, options);

%% 9. Model Evaluation on Validation Data
trainPred = predict(lstmNet, XTrain);
valPred = predict(lstmNet, XVal);
figure;
subplot(1,2,1);
plot(YTrain, 'LineWidth', 1.5);
hold on;
plot(trainPred, 'LineWidth', 1);
title('Training Predictions');
legend('Actual RUL', 'Predicted RUL');
xlabel('Sample');
ylabel('RUL');
grid on;
subplot(1,2,2);
scatter(YVal, valPred, 20, 'filled');
hold on;
plot([0 maxRUL], [0 maxRUL], 'r--', 'LineWidth', 1.5);
title(sprintf('Validation Predictions (R = %.2f)', corr(YVal, valPred)));
xlabel('Actual RUL');
ylabel('Predicted RUL');
grid on;
axis equal;

%% 10. Test Data Evaluation
[XTest, YTest] = createSequences(normalizedTestX, rulData, sequenceLength, sequenceStride);
testPred = predict(lstmNet, XTest);
rmseTest = sqrt(mean((testPred - YTest).^2));
maeTest = mean(abs(testPred - YTest));
SSres = sum((YTest - testPred).^2);
SStot = sum((YTest - mean(YTest)).^2);
rSquared = 1 - (SSres / SStot);
fprintf('Test RMSE: %.2f\nTest MAE: %.2f\nTest R-squared: %.2f\n', rmseTest, maeTest, rSquared);

figure;
scatter(YTest, testPred, 20, 'filled');
hold on;
plot([0 maxRUL], [0 maxRUL], 'r--', 'LineWidth', 1.5);
xlabel('Actual RUL');
ylabel('Predicted RUL');
title('Actual vs. Predicted RUL on Test Data');
grid on;
axis equal;

%% 11. Save Model
save(fullfile(datasetPath, 'turbofanModels_FD002_final.mat'), ...
    'lstmNet', 'mu', 'sigma', 'selectedSensors', '-v7.3');
fprintf('\n=== Model Successfully Saved ===\n');

%% Supporting Function
function [XSeq, YSeq] = createSequences(data, targets, seqLength, stride)
    numSamples = size(data, 1);
    numSequences = floor((numSamples - seqLength) / stride) + 1;
    XSeq = cell(numSequences, 1);
    YSeq = zeros(numSequences, 1);
    for i = 1:numSequences
        startIdx = (i - 1) * stride + 1;
        endIdx = startIdx + seqLength - 1;
        if endIdx > numSamples, break; end
        XSeq{i} = data(startIdx:endIdx, :)';
        if endIdx <= length(targets)
            YSeq(i) = targets(endIdx);
        end
    end
end

