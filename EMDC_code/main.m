clear all
close all
clc

addpath('fun_graph','funs','datasets');
load XXX.mat

numNearestAnchors = 5;
cnto=0;

%% Multiview FDC
for numAnchor = [6,7,8,9,10,11,12,13]
    for lambda = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3]
        ACC_ours = []; NMI_ours = []; Purity_ours = []; Fscore_ours = []; ARI_ours = []; T_ours = [];
        cnto = cnto+1;
        for cnt = 1:cntN
            [result,runtime,w,obj] = run_EMDC(X,Y,numAnchor,numNearestAnchors,lambda,'SVD_km','BKHK');
            ACC_ours(cnt,:)=result(1); NMI_ours(cnt,:)=result(2); Purity_ours(cnt,:)=result(3);Fscore_ours(cnt,:)=result(4); ARI_ours(cnt,:)=result(5); T_ours(cnt,:)=runtime;
            fprintf(fid, 'multiFDC ACC = %0.4f NMI = %0.4f Purity = %0.4f Fscore = %0.4f ARI = %0.4f RunTime= %f \r\n',ACC_ours(cnt,1), NMI_ours(cnt,1),Purity_ours(cnt,1), Fscore_ours(cnt,1),ARI_ours(cnt,1),T_ours(cnt,1));
        end
        fprintf(fid, 'multiFDC mean ACC = %0.4f NMI = %0.4f  Purity = %0.4f Fscore = %0.4f ARI = %0.4f RunTime= %0.4f \r\n',mean(ACC_ours),mean(NMI_ours),mean(Purity_ours),std(Fscore_ours),mean(ARI_ours),mean(T_ours));
        fprintf(fid, 'multiFDC std ACC = %0.4f NMI = %0.4f  Purity = %0.4f Fscore = %0.4f ARI = %0.4f RunTime= %0.4f  \r\n',std(ACC_ours),std(NMI_ours),std(Purity_ours),std(Fscore_ours),std(ARI_ours),std(T_ours));
    end
end