function [predY,w,obj] = EMDC_normS(B,U,V,paras)
% multiview FDC
% Input:
% - X: the data cell of size 1*V, X{1,v}: nSmp * nFea, where each row is a sample point;
% - B: the anchor matrix cell of size 1*V, B{1,v}: nSmp * mAnc;
% - c: the number of clusters;
% - paras:
% - lambda: parameter lambda;
% - IterMax: the maximum number of iterations;
% - k: Nummber of nearest Anchors
% Output:
% - predY: the discrete cluster labels;
% - obj: the objective value
% This code is implemented by Qianyao Qiang

if nargin < 8
    islocal = 1;
end
if (~exist('paras','var'))
    paras.lambda = 0.1;
    paras.IterMax = 50;
end

IterMax = paras.IterMax;
lambda = paras.lambda;
mu=1e-4;
rho=1.01;

[n, c] = size(U);
m = size(V,1); nv = size(B,2);
predY=[U; V];
FF = U'*U + V'*V;
FF1 = spdiags(1./sqrt(diag(FF)),0,c,c);

Bs = zeros(n,m);
w = ones(nv,1)./nv;
Bvec = [];
for v = 1:nv
    Bs = Bs+w(v,1).*B{1,v};
    Bvec = [Bvec,reshape(B{1,v},[n*m 1])];
end
Bs = full(Bs);
M = Bvec'*Bvec;

idxa = cell(n,1);
for i=1:n
    if islocal == 1
        idxa0 = find(Bs(i,:)>0);
    else
        idxa0 = 1:m;
    end
    idxa{i} = idxa0;
end

idxam = cell(m,1);
for i=1:m
    if islocal == 1
        idxa0 = find(Bs(:,i)>0);
    else
        idxa0 = 1:n;
    end
    idxam{i} = idxa0;
end

D1 = 1; D2 = 10;
for iter = 1:IterMax
    %% update P
    U1 = U * FF1; V1 = V * FF1; % H = F(F'F)^-(1/2)
    H1 = D1 * U1; H2 = D2 * V1;
    distH = L2_distance_1(H1',H2');
    P = zeros(n,m);
    for i=1:n
        idxa0 = idxa{i};
        ai = Bs(i,idxa0);
        di = distH(i,idxa0);
        ad = (ai-0.5*lambda*di);
        %S(i,idxa0) = EProjSimplex_new(ad);
        
        nn = length(ad);
        %v0 = ad-mean(ad) + 1/nn;
        v0 = ad-sum(ad)/nn + 1/nn;
        vmin = min(v0);
        if vmin < 0
            lambda_m = 0;
            while 1
                v1 = v0 - lambda_m;
                %posidx = v1>0; npos = sum(posidx);
                posidx = find(v1>0); npos = length(posidx);
                g = -npos;
                f = sum(v1(posidx)) - 1;
                if abs(f) < 10^-6
                    break;
                end
                lambda_m = lambda_m - f/g;
            end
            vv = max(v1,0);
            P(i,idxa0) = vv;
        else
            P(i,idxa0) = v0;
        end
    end
    
    Pm = zeros(m,n);
    for i=1:m
        idxa0 = idxam{i};
        ai = Bs(idxa0,i);
        di = distH(idxa0,i);
        ad = (ai-0.5*lambda*di);
        Pm(i,idxa0) = EProjSimplex_new(ad);
    end
    
    P = sparse(P);
    Pm = sparse(Pm);
    PP = (P+Pm')/2; P = PP;
    d1 = sum(PP,2);
    D1 = spdiags(1./sqrt(d1),0,n,n);
    d2 = sum(PP,1);
    D2 = spdiags(1./sqrt(d2'),0,m,m);
    P1 = D1*PP*D2;
    %% update weight vector
    p = reshape(P,[n*m 1]);
    b = 2*Bvec'*p;
    [w, w_obj]=SimplexQP(M, b);
%     [w, w_obj,~,~] = ALM(M,b,mu,rho);
    Bs = zeros(n,m);
    for v = 1:nv
        Bs = Bs+w(v).*B{v};
    end
    Bs = full(Bs);

    %% update F
    S=sparse(n+m,n+m);  S(1:n,n+1:end)=P1; S(n+1:end,1:n)=P1';
    y_ind = vec2ind(predY')';
    [y_ind, objY] = fast_cd(S,y_ind,n,m);
    predY = ind2vec(y_ind')';
    U =  predY(1:n,:); V =  predY(n+1:n+m,:);    
    
    %% convergence
    FF = U'*U + V'*V;
    FF1 = spdiags(1./sqrt(diag(FF)),0,c,c);
    FF12 = spdiags(1./(diag(FF)),0,c,c);
    obj(iter) = norm(Bs-P,'fro')^2 + lambda*(n+m-2*trace(FF12*(V'*P1'*U)));
    %     if iter>2 && ( obj(iter-1)-obj(iter) )/obj(iter-1) < 1e-10
    if iter>2 && abs( obj(iter-1)-obj(iter)) < 1e-8
        break;
    end
end
end

