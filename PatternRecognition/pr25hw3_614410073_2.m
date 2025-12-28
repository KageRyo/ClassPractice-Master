% Homework3, Question2
% 614410073 張健勳
clear; clc; close all;

% x1 points: black = R1, red = R2
X = [-4 -3 -2 -1  1  2  3  4];          % 1xN
T = [ 0  0  0  1  0  1  1  1];          % 1xN  (R1->0, R2->1)
N = numel(X);

% (a) Activation function
% Choose logistic sigmoid: f(a) = 1/(1+exp(-a)), output range (0, 1)
% (b) Target values: R1=0, R2=1
% (c) Criterion function: Cross entropy
%     E = -[ t*log(z) + (1-t)*log(1-z) ]

% Network size
d  = 1;          % input dimension
nH = 3;          % # hidden units
c  = 1;          % # output units

% Hyperparameters
eta      = 0.1;          % learning rate
maxEpoch = 50000;
epsLog   = 1e-12;        % avoid log(0)

% W1: input->hidden, size (nH x (d+1))  (include bias x0=1)
% W2: hidden->output, size (c  x (nH+1)) (include bias y0=1)
rng(1, 'twister');
W1 = 0.5 * randn(nH, d+1);
W2 = 0.5 * randn(c , nH+1);

% Training (Batch Backpropagation)
J_hist  = zeros(maxEpoch, 1);
ER_hist = zeros(maxEpoch, 1);

for epoch = 1:maxEpoch

    dW1 = zeros(size(W1));
    dW2 = zeros(size(W2));

    J = 0;
    correct = 0;

    for m = 1:N
        % Forward
        x = [1; X(m)];                 % (d+1)x1, bias included
        net_h = W1 * x;                % nH x 1
        y = sigmoid(net_h);            % nH x 1
        yb = [1; y];                   % (nH+1)x1, bias included

        net_o = W2 * yb;               % c x 1
        z = sigmoid(net_o);            % c x 1

        t = T(m);

        % Cross entropy cost (single output)
        J = J - ( t*log(z + epsLog) + (1-t)*log(1 - z + epsLog) );

        % Training accuracy
        pred = (z >= 0.5);
        correct = correct + (pred == t);

        % Backward
        % For sigmoid + cross entropy: delta_out = (t - z)
        delta_o = (t - z);                                         % c x 1

        % Accumulate batch weight updates (PR_ch6 Algorithm 2, line 06)
        dW2 = dW2 + eta * (delta_o * yb');                          % c x (nH+1)

        % Hidden layer delta:
        % delta_h = y*(1-y) .* (W2_no_bias' * delta_o)
        W2_nb = W2(:, 2:end)';                                      % nH x c
        delta_h = (y .* (1 - y)) .* (W2_nb * delta_o);              % nH x 1

        dW1 = dW1 + eta * (delta_h * x');                           % nH x (d+1)
    end

    % Update weights once per epoch
    W2 = W2 + dW2;
    W1 = W1 + dW1;

    ER = 1 - correct / N;
    J_hist(epoch)  = J;
    ER_hist(epoch) = ER;

    % Stop when training error rate reaches 0
    if ER == 0
        fprintf('Converged at epoch = %d\n', epoch);
        break;
    end
end

% Trim history arrays
J_hist  = J_hist(1:epoch);
ER_hist = ER_hist(1:epoch);

% (d) Minimum training error rate
fprintf('\n(d) Minimum training error rate (on training set) = %.2f %%\n', 100*min(ER_hist));

% (e) Weight values that give the minimum training error rate
fprintf('\n(e) Weights (W1: input->hidden, W2: hidden->output)\n');
fprintf('W1 =\n'); disp(W1);
fprintf('W2 =\n'); disp(W2);

% Show outputs for each sample
fprintf('\nOutputs z for each x:\n');
for m = 1:N
    x  = [1; X(m)];
    y  = sigmoid(W1 * x);
    yb = [1; y];
    z  = sigmoid(W2 * yb);
    fprintf('x = %+2d, t = %d, z = %.6f, pred = %d\n', X(m), T(m), z, (z>=0.5));
end

% Plot learning curve
figure;
plot(J_hist, 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Cross Entropy J');
title('Batch Backpropagation Training Curve');

figure;
plot(ER_hist, 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Training Error Rate');
title('Training Error Rate');

% Helper function
function y = sigmoid(a)
    y = 1 ./ (1 + exp(-a));
end
