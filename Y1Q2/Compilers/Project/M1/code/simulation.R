require(knnflex)
require(caret)

n <- nrow(iris)
x <- iris[,1:4]
y <- iris[,5]
train <- sample(1:n,n*0.5)
test <- (1:n)[-train]
kdist <- knn.dist(x)
preds <- knn.predict(train,test,y,kdist)
cm <- confusionMatrix(y[test],preds)

accuracy_orig <- cm$overall['Accuracy']

runs <- 1000
accuracy_error <- c()
for(i in 1:runs){
  kdist_error <- injectError(kdist, 0.02)
  preds_error <- knn.predict(train,test,y,kdist_error)
  cm_error <- confusionMatrix(y[test],preds_error)
  
  accuracy_error <- c(accuracy_error, cm_error$overall['Accuracy'])
}

print(paste("Original accuracy=",accuracy_orig))
print(summary(accuracy_error))
print(sd(accuracy_error))
print(length(accuracy_error[accuracy_error == accuracy_orig]) / runs)
plot(accuracy_error, cex=0.5,xlab="Run",ylab="Accuracy")
abline(h = accuracy_orig, col='red', cex=0.2)

