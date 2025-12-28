% Homework1, Question3
% 614410073 張健勳

% mu, Sigma
mu    = [1; 2];
Sigma = [1.0, -0.7; -0.7, 2.0];

% eigendecomposition for Sigma = U*D*U'
[U, D] = eig(Sigma);
[lam, idx] = sort(diag(D), 'descend');  % 由大到小（主軸先）
U = U(:, idx); 
D = diag(lam);

% parameterization: x(t) = mu + L*[r*cos t; r*sin t], with L = U*sqrt(D)
L = U * sqrt(D);
PI = 4*atan(1);
t = linspace(0, 2*PI, 400);
C = [cos(t); sin(t)];

% r = 1, 2 等距曲線
X1 = mu + L*(1*C);      % r = 1
X2 = mu + L*(2*C);      % r = 2

% 輸出數值(在Console)
a = sqrt(lam(1));                      % semi-major (r=1)
b = sqrt(lam(2));                      % semi-minor (r=1)
ang = atan2d(U(2,1), U(1,1));          % major-axis angle (deg)
fprintf('r=1 -> semi-axes a=%.4f, b=%.4f, angle=%.3f°\n', a, b, ang);
fprintf('r=2 -> semi-axes a=%.4f, b=%.4f, angle=%.3f°\n', 2*a, 2*b, ang);

% 輸出圖表
figure; hold on; box on; axis equal;
plot(X1(1,:), X1(2,:), 'LineWidth', 1.5);      % r = 1
plot(X2(1,:), X2(2,:), 'LineWidth', 1.5);      % r = 2
plot(mu(1), mu(2), 'k+','MarkerSize',8,'LineWidth',1.5);  % center
legend('r = 1','r = 2','\mu','Location','best');
xlabel('x_1'); ylabel('x_2');
title('614410073 - 3. Mahalanobis Distance');