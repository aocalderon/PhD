## Load the splines library
library(splines)

## Load the function for fitting penalized splines
fit.p.spline = function(y,X,lambda=0){
  ## Fit a penalized spline.
  ## Define the penalty matrix
  D = diff(diag(ncol(X)),differences=2)
  ## Compute coefficients
  b = solve(t(X) %*% X + lambda * t(D) %*% D,t(X) %*% y)
  ## Compute fit
  fit = X %*% b
  ## Compute residuals
  residuals = y-fit
  return(list(coefficients=b,
              residuals=residuals,
              fitted.values=fit))
}

plotSpline = function(data, k, plotExtras = F){
  ## Plot the raw data
  plot(data,main=paste("k = ", k),xlab="p",ylab="Accuracy",pch=21,bg=1,cex=0.4)
  
  ## Set the degree of the model, define the knots, and generate the
  ## design matrix
  K = 100   # Number of knots
  knots = (1:K)/(K+1)
  
  X = bs(data$p,knots=knots,intercept=TRUE)
  
  ## Fit the simple linear regression model
  lambda = 10                           # Smoothing parameter
  lmfit = fit.p.spline(data$Accuracy,X,lambda=lambda)
  
  ## Overlay the fitted line on the plot of the raw data
  lines(data$p,lmfit$fit,col="red",lwd=2,lty=3)
  if(plotExtras){
    plot(data$p,lmfit$resid,xlab="p",ylab="Residual") 
    abline(h=0)
    plot(data$Accuracy,lmfit$resid,xlab="Accuracy",ylab="Residual") 
    abline(h=0)    
  }
}