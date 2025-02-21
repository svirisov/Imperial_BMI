% -------------------------------------------------------------------------
% Linear Regression Decoder: One Model per Angle using RMSE-Minimizing Gradient Descent
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1) Load the dataset
load('monkeydata_training.mat');  % Loads variable 'trial'
[nTrials, nAngles] = size(trial);
nNeurons = size(trial(1,1).spikes, 1);  % e.g., 98 neurons
windowSize = 400;  % window length in ms

% Create a cell array to store the models for each angle.
% Each cell will hold the regression weights (B) for that angle.
models = cell(nAngles, 1);

%% 2) Build and train a model for each angle separately using iterative gradient descent
for angle_i = 1:nAngles
    % Count total samples for this angle
    totalSamplesAngle = 0;
    for trial_i = 1:nTrials
        totalSamplesAngle = totalSamplesAngle + size(trial(trial_i, angle_i).spikes, 2);
    end

    % Preallocate design (X_angle) and target (Y_angle) matrices for this angle
    X_angle = zeros(totalSamplesAngle, nNeurons + 1);  % +1 for bias
    Y_angle = zeros(totalSamplesAngle, 2);             % for x and y positions
    sampleIdx = 1;
    
    % Loop over trials for this angle
    for trial_i = 1:nTrials
        spikes  = trial(trial_i, angle_i).spikes;   % [nNeurons x T]
        handPos = trial(trial_i, angle_i).handPos;     % [3 x T]
        T_trial = size(spikes, 2);
        
        % Compute the moving sum over the last 'windowSize' ms
        movingSum = filter(ones(1, windowSize), 1, spikes, [], 2);  % [nNeurons x T_trial]
        
        % Build the design matrix for this trial: one row per time step.
        trialX = [ones(T_trial, 1), movingSum'];  % [T_trial x (1+nNeurons)]
        trialY = handPos(1:2, :)';                % [T_trial x 2]
        
        % Insert the trial data into the overall matrices for this angle.
        X_angle(sampleIdx:sampleIdx+T_trial-1, :) = trialX;
        Y_angle(sampleIdx:sampleIdx+T_trial-1, :) = trialY;
        sampleIdx = sampleIdx + T_trial;
    end
    
    % Train the model for this angle using iterative gradient descent minimizing RMSE
    [N, d] = size(X_angle);  % N samples, d features
    B_angle = zeros(d, 2);   % Initialize weights
    
    % Hyperparameters for gradient descent
    learning_rate = 1e-4;  % Adjust as needed
    num_iters     = 5000;
    
    for k = 1:num_iters
        % 1) Predict: Y_pred is [N x 2]
        Y_pred = X_angle * B_angle;
        
        % 2) Compute error: [N x 2]
        Err = Y_pred - Y_angle;
        
        % 3) Compute RMSE cost over all samples
        rmse_val = sqrt(mean(sum(Err.^2, 2)));
        
        % Avoid division by zero in the gradient computation
        if rmse_val == 0
            break;
        end
        
        % 4) Compute gradient of RMSE with respect to B:
        %    grad_B = (1/(N * rmse_val)) * X_angle' * (X_angle*B_angle - Y_angle)
        grad_B = (1/(N * rmse_val)) * (X_angle' * Err);
        
        % 5) Update weights
        B_angle = B_angle - learning_rate * grad_B;
        
        % (Optional) Print progress every 100 iterations
        if mod(k, 100) == 0
            fprintf('Angle %d, Iter %d: RMSE = %.4f\n', angle_i, k, rmse_val);
        end
    end
    
    % Store the trained model for this angle
    models{angle_i} = B_angle;
    fprintf('Model for angle %d trained.\n', angle_i);
end

%% 3) Test and plot each model on the first trial of its corresponding angle
for angle_i = 1:nAngles
    % Use the first trial for testing for this angle.
    testSpikes  = trial(1, angle_i).spikes;   % [nNeurons x Ttest]
    testHandPos = trial(1, angle_i).handPos;    % [3 x Ttest]
    Ttest = size(testSpikes, 2);
    
    % Compute the moving sum for the test trial
    movingSumTest = filter(ones(1, windowSize), 1, testSpikes, [], 2);  % [nNeurons x Ttest]
    X_test = [ones(Ttest, 1), movingSumTest'];  % [Ttest x (1+nNeurons)]
    
    % Retrieve the trained model for this angle
    B_angle = models{angle_i};
    
    % Predict the positions
    pred = X_test * B_angle;  % [Ttest x 2]
    x_pred = pred(:, 1);
    y_pred = pred(:, 2);
    
    % True positions (for x and y)
    x_true = testHandPos(1, :);
    y_true = testHandPos(2, :);
    
    % Compute RMSE for this angle
    rmse_x = sqrt(mean((x_pred' - x_true).^2));
    rmse_y = sqrt(mean((y_pred' - y_true).^2));
    fprintf('Angle %d: RMSE_x = %.2f, RMSE_y = %.2f\n', angle_i, rmse_x, rmse_y);
    
    % Plot the predicted versus actual trajectory
    figure;
    hold on;
    plot(x_true, y_true, 'k', 'LineWidth', 2);
    plot(x_pred, y_pred, 'r--', 'LineWidth', 2);
    legend('True Trajectory', 'Predicted Trajectory');
    xlabel('X (mm)');
    ylabel('Y (mm)');
    title(sprintf('Angle %d: True vs Predicted Trajectory', angle_i));
    axis equal;
    grid on;
end
