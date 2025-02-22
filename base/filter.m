clear all; %close all;

data = load('base/monkeydata_training.mat');
trials = data.trial; % main data format

% initialize vars
nNeurons = 98; % implicit in var dimensions
nTrials = 100;
tTotal = 613; % obsolete
tPreMovement = 300;
tPostMovement = 100;
lambda = 1; % regularization param, currently unused

% noise covariance matrices
Q = eye(4) * 1e-2; % process noise
R = eye(nNeurons) * 1e-2; % measurement noise (uncertainty in decoding)

% define state transition model (constant velocity assumption)
% further work required to implement acceleration/jerk
dt = 1; % time step
A = [1 0 dt 0; 
     0 1 0 dt; 
     0 0 1  0; 
     0 0 0  1]; 

% set 10% aside for validation via pseudo-random trial separation 
% currently unused, intended to be utilised as a local test env 
trialCutoff = 90;

for trial=1:nTrials
    for angle=1:8
        % select trial from dataset and extract critical information
        dataset = trials(trial,angle);
        spikes = dataset.spikes;
        trajectory = dataset.handPos(1:2,:);
        
        % initialize state
        state = zeros(4,1); % x, y, xdot, ydot; potential to estimate (x,y) from (xdot,ydot)
        P = eye(4); % state estimate uncertainty, initially high
        
        % populate states with true values
        velocity = diff(trajectory, 1, 2);
        velocity = [[0;0],velocity]; % restore dimensions to prevent errors
        
        path = [trajectory;velocity]; % known values
        predictedPath = zeros(size(path));
        state(:,1) = path(:,1); % set start of prediction to known start location
        
        H = spikes*path'/(path*path'); % pseudoinverse
        
        % get num of bins, 
        % intended for averaging across window (likely 20 samples)
        nBins = size(path,2); %floor(size(path,2)/20);
                
        for t=1:nBins
            if t <= 300
                % skip updates for the first 300 samples (pre-movement)
                % no prediction during pre-movement phase
                predictedPath(:, t) = state;  % repeat initial state
                continue;  % skip prediction calcs
            end
    
            z = spikes(:,t); % state meas input
            state = A * state; % update state est
            P = A * P * A' + Q; % update uncertainty
            
            % partial calcs for Kalman gain
            S = H*P*H' + R;
            Kk = P * H' / S;
        
            % update state estimates
            state = state + Kk * (z - H*state);
        
            % update covariance
            P = (eye(4) - Kk * H) * P;
        
            % Store history
            predictedPath(:,t) = state;
        end
    
    % publish last 8 angles as examples of learning (?)
    if trial==100
        name = strcat('AnglePath', num2str(angle));
        figure(1); clf;
        plot(trajectory(1, :), trajectory(2, :), 'Color', '#DE4D86', 'LineWidth', 2); hold on;
        plot(predictedPath(1, :), predictedPath(2, :), 'Color', '#2D5D7B', 'LineWidth', 2);
        legend('True Trajectory', 'Decoded Trajectory');
        xlabel('X Coordinate'); ylabel('Y Coordinate');
        title('Decoded Hand Trajectory');
        grid on;
        filename = strcat('imgs/', name, '.png');
        exportgraphics(gcf,filename,'Resolution',300);
    end
    end
end
