function [X,S,resccs] = MTVLRR(Y,A,lambda,im_size,display)
% merge total variation into low rank representation 
% min |HX|_*+lambda*|S|_2,1 s.t. Y = AX+S
c = 1;
[L,f] = size(Y);
m = size(A,2);

maxIter = 400;
mu = 1e-6;%or 1e-8
mu_bar = 1e10;%μmax
rho = 1.5; 
% initialization
X0 = 0;
atx_A = inv(A'*A+eye(m));

if nargin < 7
    display = true;
end

%%
% build handlers and necessary stuff
% horizontal difference operators    
FDh = zeros(im_size);
FDh(1,1) = -1;
FDh(1,end) = 1;
FDh = fft2(FDh);
FDhH = conj(FDh);

% vertical difference operator
FDv = zeros(im_size);
FDv(1,1) = -1;
FDv(end,1) = 1;
FDv = fft2(FDv);
FDvH = conj(FDv);

IL = 1./( FDhH.* FDh + FDvH.* FDv + 1);

Dh = @(x) real(ifft2(fft2(x).*FDh));
DhH = @(x) real(ifft2(fft2(x).*FDhH));

Dv = @(x) real(ifft2(fft2(x).*FDv));
DvH = @(x) real(ifft2(fft2(x).*FDvH));

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
if X0 == 0
    X = zeros(m, f);
end

index = 1;

% initialize V variables
V = cell(2,1);

% initialize D variables (scaled Lagrange Multipliers)
D = cell(3,1);

%  data term (always present)
V{1} = X;                                                                                                                                                                                                                                          
D{1} = zeros(size(Y));
D{2} = zeros(m, f);  

% convert X into a cube
U_im = reshape(X',im_size(1),im_size(2),m);

V{index+1} = cell(m,2);
D{index+2} = cell(m,2);
for kk = 1:m
  
    V{index+1}{kk}{1} = Dh(U_im(:,:,kk));   % horizontal differences
    V{index+1}{kk}{2} = Dv(U_im(:,:,kk));   %   vertical differences
   
    D{index+2}{kk}{1} = zeros(im_size);   % horizontal differences
    D{index+2}{kk}{2} = zeros(im_size);   %   vertical differences
end
clear U_im;

% L1
S = sparse(L,f);


%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol = 1e-4;%sqrt(f)*1e-4
% tol = 0.6;
iter = 1;
res = inf;
while (iter <= maxIter) && (sum(abs(res)) > tol)

    % solve the quadratic step (all terms depending on X)
    Xi = A'*(Y-D{1}-S);
    for j = 1
        Xi = Xi+ V{j} - D{j+1};
    end
    X = atx_A*Xi;
    
            

          
        % TV 
      if  j == 1
          nu_aux = X + D{j+1};
          % convert nu_aux into image planes
          % convert X into a cube
          nu_aux5_im = reshape(nu_aux',im_size(1), im_size(2),m);
          
          for k = 1:m
             
              V1_im(:,:,k) = real(ifft2(IL.*fft2(DhH(V{j+1}{k}{1}-D{j+2}{k}{1}) ...
                    +  DvH(V{j+1}{k}{2}-D{j+2}{k}{2}) +  nu_aux5_im(:,:,k))));
             
                aux_h = Dh(V1_im(:,:,k));
                aux_v = Dv(V1_im(:,:,k));
                temp3 = aux_h + D{j+2}{k}{1};
                temp4 = aux_v + D{j+2}{k}{2};
                
                [Us,sigma,Vs] = svd(temp3,'econ');
                [Us1,sigma1,Vs1] = svd(temp4,'econ');
               
                sigma = diag(sigma);
                svp = length(find(sigma>1/mu));
             if svp >= 1
                sigma = sigma(1:svp)-1/mu;
             else
                svp = 1;
                sigma = 0;
             end
                sigma1 = diag(sigma1);
                svp1 = length(find(sigma1>1/mu));
             if svp1 >= 1
                sigma1 = sigma1(1:svp1)-1/mu;
             else
                svp1 = 1;
                sigma1 = 0;
             end
             
             V{j+1}{k}{1} = Us(:,1:svp)*diag(sigma)*Vs(:,1:svp)';  %singular value thresholding
             V{j+1}{k}{2} = Us1(:,1:svp1)*diag(sigma1)*Vs1(:,1:svp1)';
                        
             D{j+2}{k}{1} =  D{j+2}{k}{1} + (aux_h - V{j+1}{k}{1});
             D{j+2}{k}{2} =  D{j+2}{k}{2} + (aux_v - V{j+1}{k}{2});
      end
           
            V{j} = reshape(V1_im, prod(im_size),m)';          
        
    end

    S = solve_l1l2(Y-A*X-D{1},lambda/mu);
    
    % update Lagrange multipliers    
    for j = 1:2
        if  j == 1
            D{j} = D{j} - (Y-A*X-S);
        else
            D{j} = D{j} + (X-V{j-1});
        end
    end
    % compute residuals
    if mod(iter,10) == 1
        st = []; 
        for j = 1:2
            if  j == 1
                res(j) = norm(Y-A*X-S,'fro');%Frobenius norm
                st = strcat(st,sprintf(' res(%i) = %2.6f',j,res(j) ));
            else
                res(j) = norm(X-V{j-1},'fro');
                st = strcat(st,sprintf('  res(%i) = %2.6f',j,res(j) ));
            end       
        end
         if display
         fprintf(strcat(sprintf('iter = %i -',iter),st,'\n'));
         end       
        end
     resccs(c) = sum(abs(res));  
     c = c+1;
            iter = iter + 1;    
    mu = min(mu*rho, mu_bar);

end

end


function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end