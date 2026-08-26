function A=Aclass(PSF,center,bc,n,m)

switch bc
    case 'none'
        radius=ceil(max(size(PSF)/2));
        PSFp=padarray(PSF,[n,m]-size(PSF),0,'post');
        eigA=fft2(circshift(PSFp,1-center));
    case 'periodic'
        radius=0;
        PSFp=padarray(PSF,[n,m]-size(PSF),0,'post');
        eigA=fft2(circshift(PSFp,1-center));
    otherwise
        radius=ceil(max(size(PSF)/2));
        PSFp=padarray(PSF,[n+2*radius,m+2*radius]-size(PSF),0,'post');
        eigA=fft2(circshift(PSFp,1-center));
end
A.radius=radius;
A.eigA=eigA;
A.bc=bc;
A.n=n;
A.m=m;
A.PSF=padarray(PSF,[n,m]-size(PSF),0,'post');
A.center=center;
A.transpose=0;
A=class(A,'Aclass');
end