% Homework1, Question2
% 614410073 張健勳

% u1, u2, 標準差
mu1 = -2; 
mu2 = 2; 
sigma = 1;

% 門檻公式定義
theta = @(pi) 0.25*log(pi./(1-pi));

% CDF
Phi   = @(z) 0.5*(1+erf(z/sqrt(2)));

% 風險公式定義
risk = @(pi,th) pi.*(1-Phi((th-mu1)/sigma)) + ...
                (1-pi).*Phi((th-mu2)/sigma);

% (a) bayes risk changes
pi = linspace(0.01,0.99,199);
th = theta(pi);
R_bayes = risk(pi, th);

% (b) fixed decision threshold
pi0 = 0.3; 
th0 = theta(pi0);
R_fixed = risk(pi, th0);

% (c) minimax
th_mm = (mu1+mu2)/2;
R_mm  = Phi((th_mm-mu2)/sigma);

% 輸出數值(在Console)
fprintf('(a) pi=0.3 -> theta=%.6f, R=%.6f\n', th0, risk(pi0, th0));
fprintf('(c) theta_mm=%.2f, R_mm=%.6f\n', th_mm, R_mm);

% 輸出圖表
figure; hold on; box on;
plot(pi, R_bayes, 'LineWidth', 1.5);
plot(pi, R_fixed, 'LineWidth', 1.5);
yline(R_mm, 'LineStyle','--');
legend('Bayes risk R(\pi)','Fixed-\theta risk','Minimax R_{mm}','Location','best');
xlabel('\pi = P(\omega_1)'); 
ylabel('Risk (error rate)'); 
title('614410073 - 2. Minimax Criterion');
