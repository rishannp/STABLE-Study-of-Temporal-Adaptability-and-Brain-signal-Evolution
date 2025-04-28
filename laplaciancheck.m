% Load the EEGLab location file
load('EEGLabLocsMPICap.mat');  % should contain 'Chanlocs'

% Extract coordinates and labels
X = [Chanlocs.X];
Y = [Chanlocs.Y];
Z = [Chanlocs.Z];
labels = string({Chanlocs.labels});

% Pick the reference electrode
targetLabel = 'Pz';
targetIdx = find(labels == targetLabel);

if isempty(targetIdx)
    error('Electrode %s not found.', targetLabel);
end

% Compute distances to all others
targetCoord = [X(targetIdx), Y(targetIdx), Z(targetIdx)];
coords = [X(:), Y(:), Z(:)];
dists = sqrt(sum((coords - targetCoord).^2, 2));

% Set radius threshold (e.g. 20 mm)
radius = 40;
neighborIdx = find((dists < radius) & (dists > 0));

% Print results
fprintf('\nNeighbors of %s within %.1f mm:\n', targetLabel, radius);
for i = 1:length(neighborIdx)
    fprintf('  %-5s  —  %.2f mm\n', labels(neighborIdx(i)), dists(neighborIdx(i)));
end

% Plot 2D projection for visualization
figure;
scatter(X, Y, 50, 'k'); hold on;
scatter(X(targetIdx), Y(targetIdx), 100, 'r', 'filled');
scatter(X(neighborIdx), Y(neighborIdx), 80, 'g', 'filled');
text(X+1, Y, labels, 'FontSize', 7);
title(sprintf('Neighbors of %s within %.1f mm', targetLabel, radius));
xlabel('X (mm)'); ylabel('Y (mm)');
axis equal; grid on;
