function A = ctranspose(A)
A.eigA=conj(A.eigA);
if A.transpose==1
    A.transpose=0;
else
    A.transpose=1;
end
end