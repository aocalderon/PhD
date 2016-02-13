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
  

  
