require(knnflex)
require(caret)

seeds <- read.csv('seeds.tsv', header = F, sep = '\t')
n <- nrow(seeds)
m <- ncol(seeds)
x <- seeds[,1:(m-1)]
x <- apply(x,2,normalize)
y <- seeds[,m]

set.seed(456)
train <- sample(1:n,n*0.7)
test <- (1:n)[-train]
kdist <- knn.dist(x)

k <- 1 
for(k in seq(1,9,2)){
  preds <- knn.predict(train,test,y,kdist,k=k, agg.meth = "majority")
  accuracy_orig <- getAccuracy(y[test],preds)
  as <- c()
  range <- seq(0.01,1,0.01)
  for(p in range){
    kdist_error <- injectError(kdist, p)
    preds_error <- knn.predict(train,test,y,kdist_error,k=k, agg.meth = "majority")
    accuracy_error <- getAccuracy(y[test],preds_error)
    print(paste(accuracy_orig,accuracy_error,accuracy_orig-accuracy_error))
    as <- c(as, accuracy_error)
  }
  
  data <- data.frame(p=range, Accuracy=as)
  plotSpline(data, k)
  abline(h = accuracy_orig, col='blue', cex=0.1, lty=3)
}
