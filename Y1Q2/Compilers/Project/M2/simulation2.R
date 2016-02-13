require(knnflex)
require(caret)

n <- nrow(iris)
x <- iris[,1:4]
y <- iris[,5]
set.seed(456)

k <- 1 
for(k in seq(1,9,2)){
  train <- sample(1:n,n*0.7)
  test <- (1:n)[-train]
  kdist <- knn.dist(x)
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
    print(paste(accuracy_orig,accuracy_error,accuracy_orig-accuracy_error))
    as <- c(as, accuracy_error)
  }
  
  data <- data.frame(p=range, Accuracy=as)
  plotSpline(data)
  abline(h = accuracy_orig, col='blue', cex=0.1, lty=3)
}
