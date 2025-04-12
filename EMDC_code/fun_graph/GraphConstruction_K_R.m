function [Z,B,BB,locAnchor] = GraphConstruction_K_R(X,numAnchors,numNearestAnchors,tag)
% anchor-based graph construction
% ds:dataset
% numAnchors: number of anchors
% numNearestAnchors: number of nearest anchors
V = length(X);
num = size(X{1,1},1);
% anchor generation
Xc = []; dim = zeros(V,1);
for v=1:V
    Xc = [Xc,X{v}];
    dim(v,1) = size(X{v},2);
end
% [~,locAnchors] = hKM(Xc',[1:num],numAnchors,1);
numAnchor = 2^numAnchors;
switch tag
    case 'kmeans'
    [~,~,~,locAnchors] = kmeans_fastest(Xc',numAnchor);
    case 'random'
    sample_anchor = randperm(num,numAnchor);
    locAnchors = Xc(sample_anchor,:)';
end

locAnchor = cell(1,V);
locAnchor{1,1} = locAnchors(1:dim(1,1),:)';
for v = 2:V
    locAnchor{1,v} = locAnchors(dim(v-1,1)+[1:dim(v,1)],:)';
end

% anchor based graph
Z = cell(1,V); B = Z; BB = Z;
for v = 1:V
    Z{1,v} = ConstructA_NP(X{v}',locAnchor{v}',numNearestAnchors);
    sumZ = sum(Z{1,v});
    sqrtZ = sumZ.^(-0.5);
    B{1,v} = Z{1,v}*(diag(sqrtZ));
    BB{1,v} = B{1,v}*B{1,v}';
end
end

