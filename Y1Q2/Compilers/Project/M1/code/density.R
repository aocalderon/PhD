x <- accuracy_error
h<-hist(x, xlab = "Accuracy", main="") 
xfit<-seq(min(x),max(x),length=100) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)
abline(v=accuracy_orig, lty=3)
abline(v=mean(accuracy_error), lty=2)