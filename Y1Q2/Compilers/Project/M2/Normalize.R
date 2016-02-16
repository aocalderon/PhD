normalize <- function(a){
  return((a-min(a))/(max(a)-min(a)))
}

data <- read.csv('zoo.csv', header = F)
data <- data[,2:ncol(data)]

class <- data[,ncol(data)]
data <- apply(data[,-ncol(data)],2,normalize)



