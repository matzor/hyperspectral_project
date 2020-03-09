clc; clear;

load('data/task_3_case_1.mat') % Case 1
% load('data/task_3_case_2.mat') % Case 2

[H,W,L] = size(I_original);
X          = reshape(I_noisy, [H*W,L])';
X_original = reshape(I_original, [H*W,L])';
sigma      = cov(X');

% Todo: Compute the MNF reconstruction of X
X_hat_mnf = zeros(size(X));

% Todo: Compute the PCA reconstruction of X
X_hat_pca = zeros(size(X));

% Compute average pixel error, between 0 and 1
error_mnf = error(X_hat_mnf, X_original);
error_pca = error(X_hat_pca, X_original);

subplot(221);
show_image(X_original, H,W,L);
title('Original');
subplot(222);
show_image(X, H,W,L);
title('Noisy');
subplot(223);
show_image(X_hat_mnf, H,W,L);
title(sprintf('MNF, error = %.1f%%', 100*error_mnf));
subplot(224);
show_image(X_hat_pca, H,W,L);
title(sprintf('PCA, error = %.1f%%', 100*error_pca));

function e = error(X, X_ref)
    e = mean(vecnorm(X - X_ref, 2, 1)) / mean(vecnorm(X_ref, 2, 1));
end

function show_image(X, H, W, L)
    imshow(reshape(X', [H,W,L]));
end
