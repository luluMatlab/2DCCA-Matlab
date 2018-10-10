function [Rx, Ry, Lx, Ly, Mx, My] = TDCCA(X, Y, dim, iters)
% Two-Dimensional Canonical Correlation Analysis
% Parameters:
%   'X'          - a x_height * x_width * N tensor (N is the sample size)
%   'Y'          - a x_height * x_width * N tensor (N is the sample size)
%   'dim'        - the dimension of the final feature matrix is dim*dim
%   'iters'      - iterative number
% Return:
%   'Lx, Rx'     - transformation matrices for view 1
%   'Ly, Ry'     - transformation matrices for view 2
%   'Mx'         - the mean for view 1
%   'My'         - the mean for view 2
%  Author info: 
%    a reimplementation of 2DCCA by Zhao Zhang, Oct. 2017
%    it might be useful:  www.zhaozhang.xyz

% Get dimension information of inputted samples. N is the number of samples.
[mx,nx,N] =size(X);
[my,ny,~] =size(Y);

% Remove mean
Mx=mean(X,3);
My=mean(Y,3);
X=bsxfun(@minus,X,Mx);
Y=bsxfun(@minus,Y,My);

% Initialize right projection matrices
Rx=randn(nx,dim);
Ry=randn(ny,dim);

for i=1:iters
    [Lx,Ly,~]=update_L(X,Y,dim,Rx,Ry);
    [Rx,Ry,~]=update_R(X,Y,dim,Lx,Ly);
end
end

function [Lx,Ly,corr]=update_L(X,Y,dim,Rx,Ry)

[mx,~,N] =size(X);
[my,~,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(mx,my);
for i=1:N
    s=s+X(:,:,i)*(Rx*Ry')*Y(:,:,i)';
end
Sxy = (1.0/N)*s;

s=zeros(mx,mx);
for i=1:N
    s=s+X(:,:,i)*(Rx*Rx')*X(:,:,i)';
end
  % Sxx = (1.0/N)*s+(1e-12)*eye(mx);
Sxx = (1.0/N)*s;

s=zeros(my,my);
for i=1:N
    s=s+Y(:,:,i)*(Ry*Ry')*Y(:,:,i)';
end
  % Syy = (1.0/N)*s+(1e-12)*eye(my);
Syy = (1.0/N)*s;

% For numerical stability and derive the inverse matrices
[Vx,Dx] = eig((Sxx+Sxx')/2);
[Vy,Dy] = eig((Syy+Syy')/2);
Dx = diag(Dx);
Dy = diag(Dy);
  % idx1 = find(Dx>1e-12); Dx = Dx(idx1); Vx = Vx(:,idx1);
  % idx2 = find(Dy>1e-12); Dy = Dy(idx2); Vy = Vy(:,idx2);
Sxx_inv = Vx*diag(real(Dx.^(-1/2)))*Vx';
Syy_inv = Vy*diag(real(Dy.^(-1/2)))*Vy';

% Singular value decomposition
T = Sxx_inv*Sxy*Syy_inv;
[U,D,V] = svd(T,0);
D=diag(D);
Lx = Sxx_inv*U(:,1:dim);
Ly = Syy_inv*V(:,1:dim);
D = D(1:dim);
corr=sum(D);
end

function [Rx,Ry,corr]=update_R(X,Y,dim,Lx,Ly)
[~,nx,N] =size(X);
[~,ny,~] =size(Y);

% Compute the auto covariance matrices
s=zeros(nx,ny);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Ly')*Y(:,:,i);
end
Kxy = (1.0/N)*s;

s=zeros(nx,nx);
for i=1:N
    s=s+X(:,:,i)'*(Lx*Lx')*X(:,:,i);
end
  % Kxx = (1.0/N)*s+(1e-12)*eye(nx);
Kxx = (1.0/N)*s;
s=zeros(ny,ny);
for i=1:N
    s=s+Y(:,:,i)'*(Ly*Ly')*Y(:,:,i);
end
% Kyy = (1.0/N)*s+(1e-12)*eye(ny);
Kyy = (1.0/N)*s;

% For numerical stability and derive the inverse matrices
[Vx,Dx] = eig((Kxx+Kxx')/2);
[Vy,Dy] = eig((Kyy+Kyy')/2);
Dx = diag(Dx);
Dy = diag(Dy);
  % idx1 = find(Dx>1e-12); Dx = Dx(idx1); Vx = Vx(:,idx1);
  % idx2 = find(Dy>1e-12); Dy = Dy(idx2); Vy = Vy(:,idx2);
Kxx_inv = Vx*diag(real(Dx.^(-1/2)))*Vx';
Kyy_inv = Vy*diag(real(Dy.^(-1/2)))*Vy';

% Singular value decomposition
T = Kxx_inv*Kxy*Kyy_inv;
[U,D,V] = svd(T,0);
D=diag(D);
Rx = Kxx_inv*U(:,1:dim);
Ry = Kyy_inv*V(:,1:dim);
D = D(1:dim);
corr=sum(D);
end


