function [M,N]=size(A)
if strcmp(A.bc,'none')
    N=A.n*A.m;
    z=zeros(n,m);
    z=z(r+1:end-r,r+1:end-r);
    M=numel(z);
else
    N=A.n*A.m;
    M=A.n*A.m;
end