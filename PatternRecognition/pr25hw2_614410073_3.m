% Homework2, Question3
% 614410073 張健勳
clear; clc; close all;

% Table 1 (3 categories, 10 samples each)
% [x1, x2, x3]
W1 = [ 0.42  -0.087  0.58;
      -0.2   -3.3   -3.4;
       1.3   -0.32   1.7;
       0.39   0.71   0.23;
      -1.6   -5.3   -0.15;
      -0.029  0.89  -4.7;
      -0.23   1.9    2.2;
       0.27  -0.3   -0.87;
      -1.9    0.76  -2.1;
       0.87  -1.0   -2.6 ];

W2 = [-0.4    0.58   0.089;
      -0.31   0.27  -0.04;
       0.38   0.055 -0.035;
      -0.15   0.53   0.011;
      -0.35   0.47   0.034;
       0.17   0.69   0.1;
      -0.011  0.55  -0.18;
      -0.27   0.61   0.12;
      -0.065  0.49   0.0012;
      -0.12   0.054 -0.063 ];

W3 = [ 0.83   1.6   -0.014;
       1.1    1.6    0.48;
      -0.44  -0.41   0.32;
       0.047 -0.45   1.4;
       0.28   0.35   3.1;
      -0.39  -0.48   0.11;
       0.34  -0.079  0.14;
      -0.3   -0.22   2.2;
       1.1    1.2   -0.46;
       0.18  -0.11  -0.49 ];

% 轉置成 3xN 格式 (Feature x Sample)
X1 = W1'; 
X2 = W2'; 
X3 = W3';
X_all = [X1, X2, X3]; % 3 x 30 matrix
[d, n_total] = size(X_all);


% (a) Determine the scatter matrix
% sample mean m
m = mean(X_all, 2); 

% Scatter matrix S = sum((x-m)(x-m)')
S = zeros(d, d);
for k = 1:n_total
    diff = X_all(:, k) - m;
    S = S + (diff * diff');
end

% 輸出在 Console
fprintf('(a) Scatter Matrix S:\n');
disp(S);


% (b) Two largest eigenvalues and corresponding eigenvectors
[V, D] = eig(S);
eigenvalues = diag(D);

% 排序特徵值 (由大到小)
[lam, idx] = sort(eigenvalues, 'descend');
V_sorted = V(:, idx);

% 取前兩個最大的特徵值與對應向量
lam_2 = lam(1:2);
E_2 = V_sorted(:, 1:2); % 3x2 matrix

% 輸出在 Console
fprintf('(b) Two largest eigenvalues:\n');
fprintf('    lambda_1 = %.4f\n', lam_2(1));
fprintf('    lambda_2 = %.4f\n', lam_2(2));
fprintf('    Corresponding eigenvectors (columns):\n');
disp(E_2);


% (c) Plot the projected data points
% 投影公式: y = E' * (x - m)
Y1 = E_2' * (X1 - m);
Y2 = E_2' * (X2 - m);
Y3 = E_2' * (X3 - m);

% 繪圖
figure; hold on; box on; grid on;
plot(Y1(1,:), Y1(2,:), 'ro', 'LineWidth', 1.5, 'DisplayName', '\omega_1');
plot(Y2(1,:), Y2(2,:), 'bx', 'LineWidth', 1.5, 'DisplayName', '\omega_2');
plot(Y3(1,:), Y3(2,:), 'g^', 'LineWidth', 1.5, 'DisplayName', '\omega_3');
legend('Location', 'best');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('614410073 - 3(c) PCA Projection');


% (d) Misclassification rate in 2D subspace
% 題目要求: Prior equal, Normal density
% 題目給定 Sigma 計算公式為除以 N (1/10)，非 N-1
priors = [1/3, 1/3, 1/3];

% 計算投影後資料 (2D) 的 Mean 與 Covariance
[mu1_2d, S1_2d] = muSigma_biased(Y1);
[mu2_2d, S2_2d] = muSigma_biased(Y2);
[mu3_2d, S3_2d] = muSigma_biased(Y3);

% 進行分類並計算錯誤率
params = {mu1_2d, S1_2d; mu2_2d, S2_2d; mu3_2d, S3_2d};
Y_set = {Y1, Y2, Y3};
[CM, err] = bayesEval(Y_set, params, priors);

% 輸出在 Console
fprintf('(d) Misclassification Rate:\n');
fprintf('    Confusion Matrix:\n');
disp(CM);
fprintf('    Error Rate = %.2f %%\n', 100 * err);


% 輔助函數 (Local Functions)
% 計算 Mean 與 Biased Covariance (除以 N)
% 依據題目 Source 20: Sigma = 1/10 * sum((x-u)(x-u)')
function [mu, S] = muSigma_biased(X)
    n = size(X, 2);
    mu = mean(X, 2);
    Xm = X - mu;
    S = (Xm * Xm.') / n; % biased estimator
end

% 貝式分類器評估
function [CM, err] = bayesEval(Xset, params, priors)
    C = numel(Xset);    % 類別數
    CM = zeros(C);      % 混淆矩陣
    Nall = 0; 
    hit = 0;
    
    for true_c = 1:C
        Data = Xset{true_c};
        n_samples = size(Data, 2);
        
        for k = 1:n_samples
            x = Data(:, k);
            g = zeros(C, 1); % 判別函數值
            
            % 計算每個類別的 discriminant function g_i(x)
            for i = 1:C
                mu = params{i, 1};
                S  = params{i, 2};
                % g_i(x) = -0.5*(x-mu)'*inv(S)*(x-mu) - 0.5*log(det(S)) + log(P(w_i))
                term1 = -0.5 * (x - mu)' * (S \ (x - mu)); % 使用 S\ 替代 inv(S) 提高數值穩定性
                term2 = -0.5 * log(det(S));
                term3 = log(priors(i));
                g(i) = term1 + term2 + term3;
            end
            
            % 判斷最大值
            [~, pred_c] = max(g);
            
            % 紀錄結果
            CM(true_c, pred_c) = CM(true_c, pred_c) + 1;
            hit = hit + (pred_c == true_c);
            Nall = Nall + 1;
        end
    end
    err = 1 - hit / Nall;
end