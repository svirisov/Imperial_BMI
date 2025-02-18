clear;
clc;

load("monkeydata_training.mat");

angles = [30, 70, 110, 150, 190, 230, 310, 350];

figure;
hold on;
colors = lines(8);

for k = 1:8
    for n = 1:100
        handPos = trial(n, k).handPos; 
        x = handPos(1, :);
        y = handPos(2, :);
        
        plot(x, y, 'Color', colors(k, :), 'LineWidth', 0.8, 'HandleVisibility', 'off');
    end
    plot(NaN, NaN, 'Color', colors(k, :), 'LineWidth', 1.5, 'DisplayName', sprintf('%d°', angles(k)));
end

xlabel('X');
ylabel('Y');
legend show;
grid on;
hold off;

%plotting single trajectories
figure;
plot(trial(1,1).handPos(1,:), trial(1,1).handPos(2,:));