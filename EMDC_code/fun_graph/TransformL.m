function InitF = TransformL(F, c)
% F = F-1;
% [n,~] = size(F);
% InitF = zeros(n,c);
% index = find(F==0);
% InitF(index) = 1;
% index = find(F==1);
% InitF(index,2) = 1;

% Transform label matrix from n*1 to n*c
[n,~] = size(F);
InitF = zeros(n,c);
class_set = 1:c;
for i=1:c
InitF((F == class_set(i)),i)=1;
end