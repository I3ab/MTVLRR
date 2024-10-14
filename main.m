clc,clear
close all
addpath(genpath('..\CTV和TV-LRR结果对比\MTVLRR\data'))
load HYDICE
load AHYDICE
gt = groundtruth;

Y = data;
[no_lines,no_rows,no_bands] = size(Y);
Y = Normalize(Y);
Y = reshape(Y,no_lines*no_rows,no_bands)';

im_size = [no_lines,no_rows];
display = true;
lambda = 0.7;

tic
[X,S,resccs] = MTVLRR(Y,A,lambda,im_size,display);
toc       
re = reshape(sqrt(sum((S).^2)),no_lines,no_rows);
re  = (re-min(re(:)))./(max(re(:))-min(re(:)));
[fpr,tpr,t] = perfcurve(gt(:),re(:),'1');
AUC_mtvlrr     = -sum((fpr(1:end-1)-fpr(2:end)).*(tpr(2:end) + tpr(1:end-1))/2);
AUC_mtvlrr_PDtau   = sum((t(1:end-1)-t(2:end)).*(tpr(2:end)+tpr(1:end-1))/2);
AUC_mtvlrr_PFtau   = sum((t(1:end-1)-t(2:end)).*(fpr(2:end)+fpr(1:end-1))/2);



