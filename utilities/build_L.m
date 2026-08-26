function  Ls = build_L(nx, ny, order)
switch order
case 1
Lx = spdiags([0*ones(nx,1) ones(nx,1) -ones(nx,1)],-1:1,nx,nx);
Lx = Lx(1:end-1, 1:end);
Ly = spdiags([0*ones(ny,1) ones(ny,1) -ones(ny,1)],-1:1,ny,ny);
Ly = Ly(1:end-1, 1:end);
ILx = kron(eye(nx), Lx);
LyI = kron(Ly, eye(ny));
Ls = [ILx; LyI];
case 2
Lx = spdiags([-1*ones(nx,1) 2*ones(nx,1) -ones(nx,1)],-1:1,nx,nx);
Lx = Lx(1:end-1, 1:end);
Ly = spdiags([-1*ones(ny,1) 2*ones(ny,1) -ones(ny,1)],-1:1,ny,ny);
Ly = Ly(1:end-1, 1:end);
Is = eye(nx);
ILx = kron(Is, Lx);
LyI = kron(Ly, Is);
Ls = [ILx; LyI]; 
end
end