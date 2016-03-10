require(knnflex)

banknote <- read.csv('banknote.csv', header = F, sep = ',')
n <- nrow(banknote)
m <- ncol(banknote)
x <- banknote[,1:(m-1)]
x <- apply(x,2,normalize)
y <- banknote[,m]

set.seed(456)
train <- sample(1:n,n*0.7)
test <- (1:n)[-train]
kdist <- knn.dist(x)

for(k in seq(1,1,2)){
  preds <- knn.predict(train,test,y,kdist,k=k, agg.meth = "majority")
  accuracy_orig <- getAccuracy(y[test],preds)
  as <- c()
  range <- seq(0.1,1,0.1)
  for(p in range){
    accuracy_error <- c()
    for(q in 1:1){
      kdist_error <- injectError(kdist, p)
      preds_error <- knn.predict(train,test,y,kdist_error,k=k, agg.meth = "majority")
      accuracy_error <- c(accuracy_error, getAccuracy(y[test],preds_error))
    }
    as <- c(as, mean(accuracy_error))
  }
  e = accuracy_orig - as
  write.table(e, paste0('k',k,'_banknote.dat'),col.names=F,row.names=F)
}
