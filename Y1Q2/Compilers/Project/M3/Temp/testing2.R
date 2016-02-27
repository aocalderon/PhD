e = 0.98 - as.matrix(read.table('as_iris_k1.dat'))
e = 0.94 - as.matrix(read.table('as_seeds_k1.dat'))
e = 0.84 - as.matrix(read.table('as_zoo_k1.dat'))
e = 0.98 - as.matrix(read.table('as_wine_k1.dat'))
e = 0.98 - as.matrix(read.table('as_cancer_k1.dat'))
x = seq(0.01,1,length.out = length(e))

a = 3.296e-09
b = 1.911e+01
c = 0.02
plot(x,e,type='l',col='red')
lines(x,a*exp(b*(x))+c,col='blue',lwd=2)

dr = 0.4333
a = 1 - dr
b = 19
c = 0
lines(x,a*exp(b*(x-1))+c,col='green')


