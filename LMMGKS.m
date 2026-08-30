function [x, muHist, info, saveinfo_inner] = LMMGKS(A, b, L, q, e, iter, kiter, r, tol, compress, x_true, V0)
% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% LMMGKS  Limited Memory MM-GKS solver with selectable compression.
% =====================================================================
% Unified solver combining all four compression approaches:
%   'TSVD' - Truncated SVD (paper Section 5.1)
%   'RBD'  - Reduced Basis Decomposition (paper Section 5.2)
%   'SOC'  - Solution-Oriented Compression (paper Section 5.3)
%   'SEC'  - Sparsity-Enforcing Compression (paper Section 5.4)
%
% Usage:
%   [x, muHist, info, si] = LMMGKS(A, b, L, q, e, iter, kiter, r, tol, compress)
%   [x, muHist, info, si] = LMMGKS(A, b, L, q, e, iter, kiter, r, tol, compress, x_true)
%   [x, muHist, info, si] = LMMGKS(A, b, L, q, e, iter, kiter, r, tol, compress, x_true, V0)
%
% Inputs:
%   A        - forward operator (matrix or @Aclass)
%   b        - measured data vector
%   L        - regularization operator (e.g., discrete gradient)
%   q        - norm parameter (use q=1 for l_1)
%   e        - smoothing parameter epsilon
%   iter     - number of outer expand-compress cycles (i_max)
%   kiter    - number of inner expansion steps per cycle (s = kmax - kmin)
%   r        - compressed subspace dimension (k_min)
%   tol      - convergence tolerance for ||x^{i+1}-x^{i}||/||x^{i}||
%   compress - compression method: 'TSVD', 'RBD', 'SOC', or 'SEC'
%   x_true   - (optional) true solution for computing RRE; use [] or omit
%   V0       - (optional) initial subspace for recycling (s-LM-MM-GKS)
%
% Outputs:
%   x              - reconstructed solution
%   muHist         - history of regularization parameters
%   info           - struct with fields:
%                      .Rerr      - relative reconstruction error history
%                      .saveX     - solution at each outer iteration
%                      .stop_flag - 'max_iter', 'rel_change', or 'residual'
%                      .total_iter - total inner iterations
%                      .outer_iter - outer cycles completed
%                      .oldV      - final compressed subspace (for recycling)
%   saveinfo_inner - cell array of per-cycle inner iteration info
%
%
% Algorithms implemented:
%   Algorithm 3.3 (LM-MM-GKS) — this function
%   Algorithm 3.1 (Enlarge)    — enlarge_space subfunction
%   Algorithm 3.2 (Compress)   — compress_subspace subfunction
% =====================================================================

% Default compression
if nargin < 10 || isempty(compress)
    compress = 'TSVD';
end
compress = upper(compress);

% Validate compression choice
valid = {'TSVD', 'RBD', 'SOC', 'SEC'};
if ~ismember(compress, valid)
    error('LMMGKS:invalidCompress', ...
        'compress must be one of: ''TSVD'', ''RBD'', ''SOC'', ''SEC''. Got ''%s''.', compress);
end

% Handle optional inputs
if nargin < 11, x_true = []; end
if nargin < 12, V0 = []; end
if isempty(x_true), x_true = []; end

% Initializations
b = b(:);
x = A' * b;
muHist = [];
s = kiter;           
kmax = r + s;        

%% ====================================================================
%  INITIALIZATION (paper Alg 3.3, lines 1-25)
% =====================================================================
if ~isempty(V0)
    V = V0;
    ll = size(V, 2);
    u = L * x;
    wr = (u.^2 + e^2).^(q/2 - 1);

    % Compute QR decompositions
    for j = 1:ll
        AV(:, j) = A * V(:, j);
        LV(:, j) = L * V(:, j);
    end
    [QA, RA] = qr(AV, 0);
    LL = bsxfun(@times, LV, wr.^(1/2));
    [~, RL] = qr(LL, 0);

    % Compute lambda and first solution
    [~, mu] = solveProjTikhonovGCV(RA * inv(RL), QA' * b);
    muHist = [muHist mu];
    y = [RA; sqrt(mu)*RL] \ [QA'*b; zeros(size(RL,1), 1)];
    x = V * y;
    u = L * x;
else
    ll = 15;
    [~, V, ~, ~] = GKB(A, b, ll);
    for j = 1:ll
        AV(:, j) = A * V(:, j);
        LV(:, j) = L * V(:, j);
    end

    u = L * x;
    wr = (u.^2 + e^2).^(q/2 - 1);
    [QA, RA] = qr(AV, 0);
    LL = bsxfun(@times, LV, wr.^(1/2));
    [~, RL] = qr(LL, 0);

    [~, mu] = solveProjTikhonovGCV(RA * inv(RL), QA' * b);
    muHist = [muHist mu];

    y = [RA; sqrt(mu)*RL] \ [QA'*b; zeros(size(RL,1), 1)];
    x = V * y;

    u = L * x;
    wr_new = (u.^2 + e^2).^(q/2 - 1);
    P_sq = wr_new;  

    r_init = A' * (A*x - b) + mu * L' * (P_sq .* (L*x));

    kmin_gkb = max(r - 2, 1);
    Atd = A' * b;
    V_seed = zeros(length(Atd), kmin_gkb);
    v_cur = Atd / norm(Atd);
    V_seed(:, 1) = v_cur;
    beta_prev = 0;
    v_prev = zeros(size(v_cur));
    for jj = 1:kmin_gkb
        % w = (A^T A + mu * L^T P^2 L) * v_cur
        w = A' * (A * v_cur) + mu * L' * (P_sq .* (L * v_cur));
        w = w - beta_prev * v_prev;
        alpha_j = v_cur' * w;
        w = w - alpha_j * v_cur;
        % Reorthogonalize
        for jjj = 1:jj
            w = w - (V_seed(:,jjj)' * w) * V_seed(:,jjj);
        end
        beta_j = norm(w);
        if beta_j < 1e-14, break; end
        v_prev = v_cur;
        beta_prev = beta_j;
        v_cur = w / beta_j;
        if jj < kmin_gkb
            V_seed(:, jj+1) = v_cur;
        end
    end

    [V, ~] = qr([V_seed, x(:), r_init(:)], 0);

    AV = [];
    LV = [];
    for j = 1:size(V, 2)
        AV(:, j) = A * V(:, j);
        LV(:, j) = L * V(:, j);
    end
end

%% ====================================================================
%  Initialize tracking
% =====================================================================
Rerr = zeros(iter+1, 1);
if ~isempty(x_true)
    Rerr(1) = norm(x(:) - x_true(:)) / norm(x_true(:));
end
saveinfo_inner = {};
stop_flag = 'max_iter';

%% ====================================================================
%  MAIN LOOP
% =====================================================================
for k = 1:iter
    x_before_enlarge = x;
    t1 = cputime;

    % --- ENLARGE (paper Algorithm 3.1) ---
    [x, u, V, QA, RA, QL, RL, mu, info_inner, y, res] = ...
        enlarge_space(A, b, L, u, x, V, e, q, AV, LV, x_true, s, tol);
    saveinfo_inner{k} = info_inner;
    muHist = [muHist mu];

    saveX(:, k) = x;
    if ~isempty(x_true)
        Rerr(k+1) = norm(x(:) - x_true(:)) / norm(x_true(:));
    end
    u = L * x;

    V = compress_subspace(V, RA, RL, QA, b, y, res, x, mu, r, compress);

    AV = [];
    LV = [];
    for j = 1:size(V, 2)
        AV(:, j) = A * V(:, j);
        LV(:, j) = L * V(:, j);
    end

    t2 = cputime;
    savetime(k) = t2 - t1;

    rel_change = norm(x - x_before_enlarge) / norm(x_before_enlarge);
    res_norm = norm(res);
    if rel_change <= tol
        stop_flag = 'rel_change';
        fprintf('Outer loop stopped: relative change %.2e <= tol = %.2e at cycle %d\n', ...
            rel_change, tol, k);
        break;
    end
    if res_norm <= 1e-12
        stop_flag = 'residual';
        fprintf('Outer loop stopped: residual norm %.2e at cycle %d\n', ...
            res_norm, k);
        break;
    end
end

%% Assemble output info
info.Rerr = Rerr(1:k+1);
info.saveX = saveX;
info.stop_flag = stop_flag;
info.total_iter = k * s;
info.outer_iter = k;
info.oldV = V;
if exist('savetime', 'var')
    info.timesaved = savetime;
end
end


%% ========================================================================
%  COMPRESS_SUBSPACE
% =========================================================================
function V = compress_subspace(V, RA, RL, QA, b, y, res, x, mu, r, method)

r_compress = max(r - 2, 1); 

switch method
    case 'TSVD'
       
        RARL = [RA; sqrt(mu)*RL];
        [~, ~, VV] = svd(RARL);
        VV = VV(:, 1:r_compress);
        if size(VV, 1) == size(V, 2)
            V = V * VV;
        else
            V = V(:, 1:size(VV,1)) * VV;
        end

    case 'RBD'

        RARL = [RA; sqrt(mu)*RL];
        [VV, ~] = RBD(RARL', 0.00001, r_compress);
        ncols = min(size(VV, 1), size(V, 2));
        V = V(:, 1:ncols) * VV;

    case 'SOC'

        epsTol = 1;
        Im = find(abs(y) > epsTol);
        [~, Jm] = maxk(abs(y), r_compress);
        Km = intersect(Im, Jm);
        if isempty(Km)
            [~, Km] = max(abs(y));
        end
        Km = Km(Km <= size(V, 2));
        if isempty(Km), [~, Km] = max(abs(y)); end
        V = V(:, Km);

    case 'SEC'

        rhs_sec = QA' * b;
        rho = mu;
        eps_sec = 1e-4;
        mm_iter = 20;
        z_sec = y;
        for mm = 1:mm_iter
            Rz = RL * z_sec;
            w_l1 = 1 ./ sqrt(Rz.^2 + eps_sec^2);
            W_l1 = diag(w_l1);
            z_sec = (RA'*RA + rho * RL' * W_l1 * RL) \ (RA' * rhs_sec);
        end
        epsTol = 1;
        Im = find(abs(z_sec) > epsTol);
        [~, Jm] = maxk(abs(z_sec), r_compress);
        Km = intersect(Im, Jm);
        if isempty(Km)
            [~, Km] = max(abs(z_sec));
        end
        Km = Km(Km <= size(V, 2));
        if isempty(Km), [~, Km] = max(abs(z_sec)); end
        V = V(:, Km);
end

newv = x - V * (V' * x);
newv = newv / norm(newv);
V = [V, newv];
new_res = res - V * (V' * res);   
new_res = new_res / norm(new_res);
V = [V, new_res];
end


%% ========================================================================
%  ENLARGE_SPACE — paper Algorithm 3.1
% =========================================================================
function [x, u, V, QA, RA, QL, RL, mu, info_inner, y, r] = ...
    enlarge_space(A, b, L, u, x, V, e, q, AV, LV, x_true, s, tol)

Rerr = [];
if ~isempty(x_true)
    Rerr(1) = norm(x(:) - x_true(:)) / norm(x_true(:));
end

for k = 1:s
    x_old = x;
    wr = (u.^2 + e^2).^(q/2 - 1);

    LL = bsxfun(@times, LV, wr.^(1/2));
    [QA, RA] = qr(AV, 0);
    [QL, RL] = qr(LL, 0);

    [~, mu] = solveProjTikhonovGCV(RA * inv(RL), QA' * b);

    y = [RA; sqrt(mu)*RL] \ [QA'*b; zeros(size(RL,1), 1)];
    x = V * y;

    if ~isempty(x_true)
        Rerr(k+1) = norm(x(:) - x_true(:)) / norm(x_true(:));
    end

  
    u = LV * y;

    wr_new = (u.^2 + e^2).^(q/2 - 1);
    ra = A' * (AV*y - b);
    rb = L' * (wr_new .* (LV*y));
    r = ra + mu * rb;

    r = r - V * (V' * r);
    r = r - V * (V' * r);

    vn = r / norm(r);
    V = [V, vn];
    AV = [AV, A * vn];
    LV = [LV, L * vn];

    if norm(x_old - x) / norm(x_old) <= tol
        break;
    end
end

info_inner.Rerr = Rerr;
info_inner.Citer = k;
end
