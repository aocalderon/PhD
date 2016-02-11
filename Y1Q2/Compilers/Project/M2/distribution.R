n1 <- 150
n2 <- 150
p <- 0.02
c <- c()

for(i in 1:n1){
  x <- 0
  for(j in 1:n2){
    if(runif(1) < p){
      x <- x + 1
    }  
  }
  c <- c(c, (x / n2))
}
print(c)
h <- hist(c, breaks = 25)
abline(v=p, col=3, lwd=2, lty=1)
abline(v=mean(c), col=2, lwd=2, lty=3)
abline(v=mean(c) - sd(c), col=4, lwd=1, lty=3)
abline(v=mean(c) + sd(c), col=4, lwd=1, lty=3)
abline(v=mean(c) - 2*sd(c), col=4, lwd=1, lty=3)
abline(v=mean(c) + 2*sd(c), col=4, lwd=1, lty=3)
xfit<-seq(min(c),max(c),length=100) 
yfit<-dnorm(xfit,mean=mean(c),sd=sd(c)) 
yfit <- yfit*diff(h$mids[1:2])*length(c) 
lines(xfit, yfit, col="blue", lwd=2)