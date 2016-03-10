require(knnflex)

n <- nrow(iris)
x <- iris[,1:4]
x <- apply(x,2,normalize)
y <- iris[,5]

set.seed(456)
train <- sample(1:n,n*0.7)
test <- (1:n)[-train]
kdist <- knn.dist(x)

for(k in seq(1,20,2)){
  preds <- knn.predict(train,test,y,kdist,k=k)
  accuracy_orig <- getAccuracy(y[test],preds)
  as <- c()
  range <- seq(0.01,1,0.01)
  for(p in range){
    accuracy_error <- c()
    for(q in 1:25){
      kdist_error <- injectError(kdist, p)
      preds_error <- knn.predict(train,test,y,kdist_error,k=k, agg.meth = "majority")
      accuracy_error <- c(accuracy_error, getAccuracy(y[test],preds_error))
    }
    as <- c(as, mean(accuracy_error))
    print(paste0('Iris: q->',q,' p->',p,' k->',k))
  }
  e = accuracy_orig - as
  write.table(e, paste0('Temp2/k',k,'_iris.dat'),col.names=F,row.names=F)
#   data <- data.frame(p=range, Accuracy=as)
#   pdf(paste0("figures/Iris_k",k,".pdf"), 7.83, 5.17)
#   plotSpline(data, k)
#   abline(h = accuracy_orig, col='blue', cex=0.1, lty=3)
#   text(0.97,accuracy_orig-0.05,paste("acc=",round(accuracy_orig,2)),cex=0.6,col="blue")
#   dev.off()
#   print(k)
}
