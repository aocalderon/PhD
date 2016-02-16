require(knnflex)
require(caret)

n <- nrow(iris)
x <- iris[,1:4]
x <- apply(x,2,normalize)
y <- iris[,5]

set.seed(456)
train <- sample(1:n,n*0.7)
test <- (1:n)[-train]
kdist <- knn.dist(x)

k <- 1 
for(k in seq(1,9,2)){
  preds <- knn.predict(train,test,y,kdist,k=k)
  cm <- confusionMatrix(y[test],preds)
  accuracy_orig <- cm$overall['Accuracy']
  as <- c()
  range <- seq(0.01,1,0.01)
  for(p in range){
    kdist_error <- injectError(kdist, p)
    preds_error <- knn.predict(train,test,y,kdist_error,k=k)
    cm_error <- confusionMatrix(y[test],preds_error)
    accuracy_error <- cm_error$overall['Accuracy']
    as <- c(as, accuracy_error)
  }
  
  data <- data.frame(p=range, Accuracy=as)
  pdf(paste0("figures/Iris_k",k,".pdf"), 7.83, 5.17)
  plotSpline(data, k)
  abline(h = accuracy_orig, col='blue', cex=0.1, lty=3)
  text(0.97,accuracy_orig-0.05,paste("acc=",round(accuracy_orig,2)),cex=0.6,col="blue")
  dev.off()
  print(k)
}
