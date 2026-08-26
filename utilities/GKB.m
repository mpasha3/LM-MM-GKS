function [B, V, U, k] = GKB(A, b, steps)
% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% GKB  Golub-Kahan Bidiagonalization.
% =====================================================================
% Computes steps of the Golub-Kahan bidiagonalization of A with
% starting vector b, producing an orthonormal basis V for the
% Krylov subspace K_steps(A^T A, A^T b).
%
% The decomposition satisfies:
%   A * V(:,1:k) = U(:,1:k+1) * B(1:k+1, 1:k)
%
% where V has orthonormal columns (n x k), U has orthonormal columns
% (m x k+1), and B is lower bidiagonal ((k+1) x k).
%
% Usage:
%   [B, V, U, k] = GKB(A, b, steps)
%
% Inputs:
%   A     - forward operator (m x n matrix or function handle)
%   b     - starting vector (m x 1)
%   steps - number of bidiagonalization steps
%
% Outputs:
%   B - bidiagonal matrix ((steps+1) x steps)
%   V - right basis vectors (n x steps), orthonormal columns
%   U - left basis vectors (m x (steps+1)), orthonormal columns
%   k - number of steps completed
%
% Reference:
%   Golub & Kahan, "Calculating the singular values and pseudo-inverse
%   of a matrix", SIAM J. Numer. Anal., 1965.
%   Paper Algorithm 2.1 uses this for initialization (eq 2.7).
% =====================================================================

m = size(b, 1);
n = size(A' * b, 1);  % infer n from A^T * b (works for both matrix and Aclass)

B = zeros(steps+1, steps);
U = zeros(m, steps+1);
V = zeros(n, steps);

% Initialize
beta = norm(b);
u = b / beta;
U(:, 1) = u;
v = zeros(n, 1);

for k = 1:steps
    % Compute r = A^T u_k - beta * v_{k-1}
    r = A' * u - beta * v;

    % Reorthogonalize against previous V columns
    for j = 1:k-1
        r = r - (V(:,j)' * r) * V(:,j);
    end

    alpha = norm(r);
    v = r / alpha;
    B(k, k) = alpha;
    V(:, k) = v;

    % Compute p = A v_k - alpha * u_k
    p = A * v - alpha * u;

    % Reorthogonalize against previous U columns
    for j = 1:k
        p = p - (U(:,j)' * p) * U(:,j);
    end

    beta = norm(p);
    u = p / beta;
    B(k+1, k) = beta;
    U(:, k+1) = u;
end
end
