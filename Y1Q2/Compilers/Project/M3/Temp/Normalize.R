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

data <- read.csv('zoo.csv', header = F)
data <- data[,2:ncol(data)]

class <- data[,ncol(data)]
data <- apply(data[,-ncol(data)],2,normalize)



