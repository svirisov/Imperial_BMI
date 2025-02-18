% -------------------------------------------------------------------------
% Simple Linear Regression Decoder for (x, y) from Neural Spikes
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
X_all = [];  % will become [TotalSamples x (1 + nNeurons)]
Y_all = [];  % will become [TotalSamples x 2], since we want x & y

%% 3) Build the design (training) matrix
% We loop over all angles, all trials, and all time bins:
for angle_i = 1:nAngles
    for trial_i = 1:nTrials
        
        % Extract spikes and handPos for this trial
        spikes  = trial(trial_i, angle_i).spikes;   % [nNeurons x T]
        handPos = trial(trial_i, angle_i).handPos;  % [3 x T]
        T = size(spikes, 2);
        
        % For each time t, sum spikes in the last 'windowSize' ms
        t=1;
        while t < T
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

%% 4) Fit the linear regression weights
% We solve Y = X * B, where B = [ (1+nNeurons) x 2 ]
% Column 1 -> weights for predicting X
% Column 2 -> weights for predicting Y

% B = inv(X_all' * X_all) * X_all' * Y_all;  % Classic formula, but
% "pinv" is often safer numerically:
B = pinv(X_all) * Y_all;  % yields a (1+nNeurons) x 2 matrix

% You can split them if you want:
beta_x = B(:,1);  % weights to get x
beta_y = B(:,2);  % weights to get y

%% 5) Test the model on the training set (for demonstration)
% We'll do a quick check by re-predicting some positions on the same data
% (Typically you'd do cross-validation or use a held-out test set.)

% Let's pick just the first trial of angle 1 to see how well we do.
testSpikes  = trial(1, 1).spikes;   % [nNeurons x Ttest]
testHandPos = trial(1, 1).handPos;  % [3 x Ttest]
Ttest = size(testSpikes, 2);

x_pred = zeros(1, Ttest);
y_pred = zeros(1, Ttest);

for t = 1:Ttest
    tStart = max(1, t - windowSize + 1);
    recentSpikes = sum(testSpikes(:, tStart:t), 2);  % sum over last windowSize
    X_row = [1; recentSpikes];  % same feature construction
    
    x_pred(t) = X_row' * beta_x;  % dot product
    y_pred(t) = X_row' * beta_y;
end

% Ground truth
x_true = testHandPos(1, :);
y_true = testHandPos(2, :);

% Quick RMSE
rmse_x = sqrt(mean((x_pred - x_true).^2));
rmse_y = sqrt(mean((y_pred - y_true).^2));
fprintf('Demo: On trial(1,1), RMSE_x = %.2f, RMSE_y = %.2f\n', rmse_x, rmse_y);

%% 6) Plot predicted vs actual trajectory (for the example trial)
figure; hold on;
plot(x_true, y_true, 'k', 'LineWidth', 2);
plot(x_pred, y_pred, 'r--', 'LineWidth', 2);
legend('True Trajectory', 'Predicted Trajectory');
xlabel('X (mm)');
ylabel('Y (mm)');
title('Decoded Hand Trajectory (Simple Linear Regression)');
axis equal;
grid on;
