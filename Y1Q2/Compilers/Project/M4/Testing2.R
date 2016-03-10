y = as.matrix(read.table('Temp/k1_iris.dat'))
y = as.matrix(read.table('Temp/k1_cancer.dat'))
y = as.matrix(read.table('Temp/k1_wine.dat'))
y = as.matrix(read.table('Temp/k1_seeds.dat'))
y = as.matrix(read.table('Temp/k1_zoo.dat'))

x = seq(0.01,1,length.out = length(y))

plot(x,y,type='l',col='red',lwd=2)

a = 3.296e-09
b = 19
c = 0.02
lines(x,a*exp(b*(x)),col='blue',lwd=2)

dr = 0.4
a = 1 - dr
b = 7
c = 0
lines(x,0.2*exp(b*(x-1))+c,col='green',lwd=2)

