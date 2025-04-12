function [result,runtime,w,obj] = run_EMDC(X,Y,numAnchor,numNearestAnchors,lambda,tag_ini,tag_anchor)
% run EMDC
[Y, idx] = sort(Y);
nv = length(X);
c = length(unique(Y));
n = size(Y,1);
for v = 1:nv
    X{v} = X{v}(idx, :);
    X{v} = double(X{v});
%     for  j = 1:n
%         X{v}(j,:) = ( X{v}(j,:) - mean( X{v}(j,:) ) ) / std( X{v}(j,:) ) ;
%     end
end
countC = histcounts(Y);
fprintf('The clusters number are: %d \n',countC);

m = 2^numAnchor;
paras.IterMax = 50;
paras.lambda = lambda; % 0.1

% Anchor Generation
switch tag_anchor
    case 'BKHK' % here we use BKHK algorithm
        [Z,B,BB,locAnchors] = GraphConstruction_B(X,numAnchor,numNearestAnchors);
    case 'kmeans'
        [Z,B,BB,locAnchors] = GraphConstruction_K_R(X,numAnchor,numNearestAnchors,tag_anchor);        
    case 'random'
        [Z,B,BB,locAnchors] = GraphConstruction_K_R(X,numAnchor,numNearestAnchors,tag_anchor);
        
end
% F initialization
switch tag_ini
    case 'SVD_km'
        Bs = zeros(n,m);
        for v = 1:nv
            Bs = Bs+B{1,v};
        end
        Bs = Bs./sqrt(nv);
        BBs = full(Bs'*Bs);
        [V, ev0, ev]=eig1(BBs,m);
        V = V(:,1:c);
        U=(Bs*V)./(ones(n,1)*sqrt(ev0(1:c)'));
        % k-menas
        U = sqrt(2)/2*U; V = sqrt(2)/2*V;
        Us = kmeans(U, c); Vs = kmeans(V, c);
        U = ind2vec(Us')'; V = ind2vec(Vs')'; % U = TransformL(Us, c); V = TransformL(Vs, c);
    case 'km'
        % k-means on X{1,1} and M{1,1}
        U = kmeans(X{1,1}, c); U = ind2vec(U'); U=U';
        V = kmeans(locAnchors{1,1}, c); V = ind2vec(V'); V=V';
    case 'random'
        U = zeros(n,c);
        V = zeros(m,c);
        gl = round(rand(1,n)*(c-1))+1;
        for i = 1:n
            U(i,gl(i)) = 1;   % O(n)
        end
        g2 = round(rand(1,m)*(c-1))+1;
        for i = 1:m
            V(i,g2(i)) = 1;   % O(n)
        end
end


% Our Fast muti-view Discrete Graph Clustering
tic;
[predY,obj] = EMDC(Z,U,V,paras);
% [predY,w,obj] = EMDC_normS(Z,U,V,paras);
runtime = toc;
[~, col] = find(predY==1);

result = ClusteringMeasure(Y, col(1:n,:));
f=compute_f(Y,  col(1:n,:));ari=compute_RandIndex(Y, col(1:n,:));
result=[result f ari];
end

