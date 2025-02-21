% -------------------------------------------------------------------------
% Simple Linear Regression Decoder for (x, y) from Neural Spikes (Iterative)
% -------------------------------------------------------------------------

clear; clc; close all;

%% 1) Load the provided dataset
load('monkeydata_training.mat');  % Loads 'trial' into the workspace
% 'trial' is a [nTrials x nAngles] struct with fields:
%   .spikes  -> [nNeurons x T]
%   .handPos -> [3 x T]
%   .trialId (unused here)

% Basic info
[nTrials, nAngles] = size(trial);
nNeurons = size(trial(1,1).spikes, 1);

%% 2) Define a time window for summing spikes
windowSize = 300;  % in ms (you can adjust this)

% We will build a design matrix X and a target matrix Y.
%   X_all: [TotalSamples x (1 + nNeurons)]
%   Y_all: [TotalSamples x 2] (for x and y)
X_all = [];
Y_all = [];

%% 3) Build the design (training) matrix
% We loop over all angles, all trials, and all time bins:
for angle_i = 1:nAngles
    for trial_i = 1:nTrials
        
        % Extract spikes and handPos for this trial
        spikes  = trial(trial_i, angle_i).spikes;   % [nNeurons x T]
        handPos = trial(trial_i, angle_i).handPos;  % [3 x T]
        T = size(spikes, 2);
        t=1;
        % For each time t, sum spikes in the last 'windowSize' ms
        while t<T
            tStart = max(1, t - windowSize + 1);
            recentSpikes = sum(spikes(:, tStart:t), 2);  % [nNeurons x 1]
            
            % Construct feature vector with a 1 for bias
            X_row = [1; recentSpikes];  % dimension = (1 + nNeurons) x 1
            
            % Target: current x(t), y(t)
            xTrue = handPos(1, t);  % X position
            yTrue = handPos(2, t);  % Y position
            
            % Append to full design matrix
            X_all = [X_all, X_row];
            Y_all = [Y_all, [xTrue; yTrue]];
            t=t+5;
        end
    end
    disp(angle_i);
end

% Convert to standard row-wise format
X_all = X_all';   % [Nsamples x (1 + nNeurons)]
Y_all = Y_all';   % [Nsamples x 2]

%% 4) Fit the linear regression weights (ITERATIVE GRADIENT DESCENT)
%
% Instead of using:
%   B = pinv(X_all) * Y_all;
% we do an iterative approach.

% Basic dimensions
[N, d] = size(X_all);  % Nsamples x nFeatures
% Y_all is Nsamples x 2

% Initialize B as zeros (size: [d x 2])
B = zeros(d, 2);

% Hyperparameters for gradient descent
learning_rate = 1e-6;  % adjust if it diverges or is too slow
num_iters     = 1000;

for k = 1:num_iters
    % 1) Predict: Y_pred is [N x 2]
    Y_pred = X_all * B;

    % 2) Error: [N x 2]
    Err = Y_pred - Y_all;

    % 3) Gradient wrt B: [d x 2]
    % MSE gradient factor is (2/N)*X_all'*(X_all*B - Y_all)
    grad_B = (2/N) * (X_all' * Err);

    % 4) Update step
    B = B - learning_rate * grad_B;

    % (Optional) Track MSE every 100 iterations
    if mod(k, 100) == 0
        mse_val = mean(sum(Err.^2, 2));  % sum across x,y -> [N x 1], then average
        fprintf('Iter %d: MSE=%.4f\n', k, mse_val);
    end
end

% Now, B is [d x 2]. The first column is for x, the second for y
beta_x = B(:,1);
beta_y = B(:,2);

%% 5) Test the model on the training set (for demonstration)
% We'll do a quick check by re-predicting some positions on the same data
% (Typically you'd do cross-validation or use a held-out test set.)

% Let's pick just the first trial of angle 1 to see how well we do.
testSpikes  = trial(1, 1).spikes;   % [nNeurons x Ttest]
testHandPos = trial(1, 1).handPos;  % [3 x Ttest]
Ttest       = size(testSpikes, 2);

x_pred = zeros(1, Ttest);
y_pred = zeros(1, Ttest);

for t = 1:Ttest
    tStart = max(1, t - windowSize + 1);
    recentSpikes = sum(testSpikes(:, tStart:t), 2);  % sum over last windowSize
    X_row        = [1; recentSpikes];  % same feature construction
    
    x_pred(t) = X_row' * beta_x;  % dot product
    y_pred(t) = X_row' * beta_y;
end

% Ground truth
x_true = testHandPos(1, :);
y_true = testHandPos(2, :);

% Quick RMSE
rmse_x = sqrt(mean((x_pred - x_true).^2));
rmse_y = sqrt(mean((y_pred - y_true).^2));
fprintf('\nDemo: On trial(1,1), RMSE_x = %.2f, RMSE_y = %.2f\n', rmse_x, rmse_y);


figure; hold on;
plot(x_true, y_true, 'k', 'LineWidth', 2);
plot(x_pred, y_pred, 'r--', 'LineWidth', 2);
legend('True Trajectory', 'Predicted Trajectory');
xlabel('X (mm)');
ylabel('Y (mm)');
title('Decoded Hand Trajectory (Simple Linear Regression)');
axis equal;
grid on;
