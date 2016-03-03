t=dlmread('truth.txt')
s=dlmread('scores.txt')
plotroc(t,s)
[X,Y,T,AUC] =perfcurve(t(1,:),s(1,:),1);
