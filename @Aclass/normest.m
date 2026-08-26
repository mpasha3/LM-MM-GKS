function nrm=normest(A)

L=A.eigA;
nrm=sqrt(max(abs(L(:))).^2);