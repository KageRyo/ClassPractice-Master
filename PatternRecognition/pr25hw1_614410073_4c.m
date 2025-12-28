% Homework1, Question4 (c)
% 614410073 張健勳

% 30 samples from three categories
W1 = [ -5.01 -8.12 -3.68; -5.43 -3.48 -3.54;  1.08 -5.52  1.66; ...
        0.86 -3.78 -4.11; -2.67  0.63  7.39;  4.94  3.29  2.08; ...
       -2.51  2.09 -2.59; -2.25 -2.13 -6.94;  5.56  2.86 -2.26; ...
        1.03 -3.33  4.33 ];
W2 = [ -0.91 -0.18 -0.05;  1.30 -2.06 -3.53; -7.75 -4.54 -0.95; ...
       -5.47  0.50  3.92;  6.14  5.72 -4.85;  3.60  1.26  4.36; ...
        5.37 -4.63 -3.65;  7.18  1.46 -6.66; -7.39  1.17  6.30; ...
       -7.50 -6.32 -0.31 ];
W3 = [  5.35  2.26  8.13;  5.12  3.22 -2.66; -1.34 -5.31 -9.87; ...
        4.48  3.42  5.19;  7.11  2.39  9.21;  7.17  4.33 -0.98; ...
        5.75  3.97  6.65;  0.77  0.27  2.41;  0.90 -0.43 -8.71; ...
        3.52 -0.36  6.43 ];

% 用 x1, x2, x3 當特徵值
X1 = W1';    % 3 x 10
X2 = W2';    % 3 x 10
X3 = W3';    % 3 x 10

% Means & Covariances(μ, Σ)
[mu1,S1] = muSigma(X1);
[mu2,S2] = muSigma(X2);
[mu3,S3] = muSigma(X3);

% 輸出數值(在Console)
pval('w1', mu1, S1);
pval('w2', mu2, S2);
pval('w3', mu3, S3);

% Calculate the percentage of misclassified samples
P = [0.3, 0.5, 0.2];
[CM, err] = bayesEval({X1,X2,X3}, {mu1,S1; mu2,S2; mu3,S3}, P);

% 輸出數值、混淆矩陣、誤判率(Console)
disp(CM);
fprintf('錯誤率 = %.2f %%\n', 100*err);

% 母體估計
function [mu,S] = muSigma(X)
    n  = size(X,2);
    mu = mean(X,2);
    Xm = X - mu;
    S  = (Xm*Xm.')/n;   % 母體共變異(除以 n)
end

% 高斯判別式
function [CM, err] = bayesEval(Xset, params, priors)
    C   = numel(Xset);
    CM  = zeros(C);
    Nall = 0; hit = 0;
    for c = 1:C
        X = Xset{c};
        for k = 1:size(X,2)
            x = X(:,k);
            g = zeros(C,1);
            for i = 1:C
                mu = params{i,1};  S = params{i,2};
                g(i) = -0.5*log(det(S)) - 0.5*(x-mu).'*(S\(x-mu)) + log(priors(i));
            end
            [~, pred] = max(g);
            CM(c,pred) = CM(c,pred)+1;
            hit  = hit + (pred==c);
            Nall = Nall + 1;
        end
    end
    err = 1 - hit/Nall;
end

% 輸出
function pval(tag, mu, S)
    fprintf('%s: mu = [%.3f; %.3f; %.3f]\n', tag, mu(1), mu(2), mu(3));
    fprintf('%s: Sigma =\n', tag); 
    disp(S);
end