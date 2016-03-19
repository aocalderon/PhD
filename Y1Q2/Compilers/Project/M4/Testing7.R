y = as.matrix(read.table('Temp/k1_seeds.dat'))
x = seq(0.00,1,length.out = length(y))
k=1
dr = 0.3333
a = k * (1 - dr)
N = 210
C = 3
b = N*(dr)
plot(x,k*(dr)*exp(dr*(x-1)),col='blue',lwd=2,lty=2,type='l',ylim=c(0,1))
lines(x,y,col='red',lwd=2,lty=2,type='l')
