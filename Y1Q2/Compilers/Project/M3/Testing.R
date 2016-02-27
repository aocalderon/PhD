x=seq(0.01,1,0.01)

y=read.table('as_iris_k1.dat')
y=as.matrix(y)

d <- data.frame(x,y)  ## need to use data in a data.frame for predict()
logEstimate <- lm(y~log(x),data=d)

plot(x, y,main=paste("k = ", k),xlab="p",ylab="Accuracy",pch=21,bg=1,cex=0.4)

xvec <- seq(0,100,length=101)
logpred <- predict(logEstimate,newdata=data.frame(x=xvec))
lines(xvec,logpred)

coef(logEstimate)
## (Intercept)      log(x) 
##  0.6183839   0.0856404 
curve(0.61838+0.08564*log(x),add=TRUE,col=2)

with(as.list(coef(fit)), curve(`(Intercept)`+`log(e)`*log(e),add=TRUE,col=4))

est1 <- predict(lm(y~x,data=d),newdata=data.frame(x=xvec))
plot(est1,logpred)



