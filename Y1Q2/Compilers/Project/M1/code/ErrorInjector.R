injectError <- function(kdist, p = 0.02){
  n <- nrow(kdist)
  m <- n
  mn <- .Machine$double.xmin
  mx <- .Machine$double.xmax
  for(i in 1:(m-1)){
    for(j in (i+1):n){
      if(runif(1) < p){
        random <- runif(1, mn, mx)
        kdist[i,j] <- random
        kdist[j,i] <- random
        # print(paste(i,j))
      }
    }
  }
  return(kdist)
}
