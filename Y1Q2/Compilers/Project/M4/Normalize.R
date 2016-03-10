normalize <- function(a){
  return((a-min(a))/(max(a)-min(a)))
}

getAccuracy <- function(obs, pre){
  n <- length(obs)
  c <- 0
  for(i in 1:n){
    if(obs[i] == pre[i]){
      c <- c +1
    }
  }
  return(c/n)
}




