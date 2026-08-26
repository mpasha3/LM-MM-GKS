function [y, lambda] = solveProjTikhonovGCV(M, b)
% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%

k = size(M, 1);
if(k ==1)
    M = M(1:k, 1:k);
    [U, S, V] = svd(M(1:k, 1:k), 'econ');
    b = b(1:k);
    s = S(1, 1);
else
    [U, S, V] = svd(M, 'econ');
    U = U(:, 1:k);
    V = V(:, 1:k);
    S = S(1:k, 1:k);
    s = diag(S);
end
%beta = U'*b;
%delta = norm(b - U*beta);
%bhat = norm(b)*eye(k+1,1);
%[y1,lambda1] = discrep(U,s,V,b,delta);
[lambda,G,reg_param] = gcv(U,s,b,'Tikh');
%         [U,S,V] = csvd(M);
%         [y1, lambda] = discrep(U,S,V,b,delta);

%         dS = diag(S); 
%         dS2 = dS.^2; 
%         Utbi = U'*b;
%         RegParam_fn = @(a) abs(sum((a^2*Utbi.^2)./(dS2+a).^2)-delta^2);
%         lambda = fminbnd(RegParam_fn,0,1);

dS      = s; 
dS2     = dS.^2; 
Utb     = U'*b;
dSfilt    = dS./(dS2+lambda^2);
y    = V*(dSfilt.*Utb);


% lambda = newton(bhathat, s, eta*delta, lambda);
% k = size(M, 2);
%  y = (M'*M + lambda*eye(k))\(M'*b);
end



