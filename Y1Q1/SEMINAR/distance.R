require('lsa')
require('rgl')

setwd('/home/and/Documents/Projects/C++/word2vec/trunk/')
vectors <- read.table('vectors_small_3.txt', header = F, sep = " ")
dims <- dim(vectors)[2] - 1
vectors <- vectors[,1:dims]

getDistance <- function(vectors, word){
  n <- dim(vectors)[1]
  vectors$d <- rep(0.0, n)
  v1 <- as.numeric(vectors[vectors[,1] == word, 2:dims])
  i <- 1
  for(w in 1:n){
    v2 <- as.numeric(vectors[w, 2:dims])
    vectors[i,dims + 1] <- cosine(v1, v2)
    i <- i + 1
  }
  return(vectors[order(-vectors$d),])
}

# california3 <- getDistance(vectors, 'california')

plot3d(california3[1:10, 2:4], col = 'red', cex = 10, xlab = '', ylab = '', zlab = '')
points3d(california3[1:2000, 2:4], add = T)
text3d(california3[1:10, 2],california3[1:10, 3],california3[1:10, 4],california[1:10,1], cex = 0.5)