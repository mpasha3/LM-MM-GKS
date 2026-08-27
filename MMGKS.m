function [x, muHist, info] = MMGKS(A, b, L, q, epsilon, iter, tol, x_true)
% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% MMGKS  Standard Majorization-Minimization Generalized Krylov Subspace method.
% =====================================================================
% Solves the edge-preserving regularized inverse problem:
%   min_x  (1/2)||Ax - d||_2^2 + (lambda/q)||L x||_q^q
%
% using MM-GKS (Algorithm 2.1 in the paper). The search space grows
% by one vector per iteration (no compression). The regularization
% parameter lambda is selected automatically via GCV on the small
% projected problem at each iteration.
%
% Usage:
%   [x, muHist, info] = MMGKS(A, b, L, q, epsilon, iter, tol)
%   [x, muHist, info] = MMGKS(A, b, L, q, epsilon, iter, tol, x_true)
%
% Inputs:
%   A       - forward operator (matrix or @Aclass)
%   b       - measured data vector
%   L       - regularization operator (e.g., discrete gradient)
%   q       - norm parameter (use q=1 for l_1)
%   epsilon - smoothing parameter for the l_q approximation
%   iter    - maximum number of iterations
%   tol     - convergence tolerance: stop when ||x^{k}-x^{k-1}||/||x^{k-1}|| < tol
%   x_true  - (optional) true solution for computing RRE; use [] or omit
%
% Outputs:
%   x      - reconstructed solution
%   muHist - history of regularization parameters (lambda at each iter)
%   info   - struct with fields:
%              .saveX     - solution at each iteration (n x iter)
%              .Rerr      - relative reconstruction error history
%              .iter      - final iteration count
%              .stop_flag - 'converged' or 'max_iter'
%              .timesaved - CPU time per iteration
%   Algorithm 2.1 (MM-GKS).
% =====================================================================

% Handle optional x_true
if nargin < 8 || isempty(x_true)
    x_true = [];
end

% Initializations
b = b(:);
x = A' * b;
muHist = [];
stop_flag = 'max_iter';

% Build initial Krylov subspace via Golub-Kahan bidiagonalization (paper eq 2.7)
ll = 15;
[~, V, ~, ~] = GKB(A, b, ll);

% Precompute A*V and L*V
for j = 1:ll
    AV(:, j) = A * V(:, j);
    LV(:, j) = L * V(:, j);
end

% Initialize error tracking
Rerr = zeros(iter+1, 1);
if ~isempty(x_true)
    Rerr(1) = norm(x - x_true(:)) / norm(x_true(:));
end

u = L * x;

% Begin MM-GKS iterations (paper Algorithm 2.1)
for k = 1:iter
    x_old = x;
    t1 = cputime;

    % Compute weights (paper eqs 2.1-2.3)
    wr = (u.^2 + epsilon^2).^(q/2 - 1);

    % QR factorizations (paper eq 2.8)
    LL = bsxfun(@times, LV, wr.^(1/2));
    [QA, RA] = qr(AV, 0);
    [~, RL] = qr(LL, 0);

    % Select lambda via GCV on projected problem (paper Section 2.3)
    [~, mu] = solveProjTikhonovGCV(RA * inv(RL), QA' * b);
    muHist = [muHist mu];

    % Solve projected system (paper eq 2.9)
    y = [RA; sqrt(mu)*RL] \ [QA'*b; zeros(size(RL,1), 1)];
    x = V * y;

    % Store solution and RRE
    saveX(:, k) = x;
    if ~isempty(x_true)
        Rerr(k+1) = norm(x - x_true(:)) / norm(x_true(:));
    end

    % Update u = L*x via projected computation
    u = LV * y;

    % Compute residual of regularized normal equations (paper eq 2.10)
    ra = A' * (AV*y - b);
    rb = L' * (wr .* (LV*y));
    r = ra + mu * rb;

    % Reorthogonalize (twice for numerical stability)
    r = r - V * (V' * r);
    r = r - V * (V' * r);

    % Check convergence
    if norm(x - x_old) / norm(x_old) < tol
        stop_flag = 'converged';
        break;
    end

    % Enlarge subspace (paper eq 2.11)
    vn = r / norm(r);
    V = [V, vn];
    AV = [AV, A * vn];
    LV = [LV, L * vn];

    t2 = cputime;
    savetime(k) = t2 - t1;
end

% Assemble output info
info.saveX = saveX;
info.Rerr = Rerr(1:k+1);
info.iter = k;
info.stop_flag = stop_flag;
if exist('savetime', 'var')
    info.timesaved = savetime;
end
end
