require(knnflex)
require(caret)

n <- nrow(iris)
x <- iris[,1:4]
y <- iris[,5]
train <- sample(1:n,n*0.75)
test <- (1:n)[-train]
kdist <- knn.dist(x)
preds <- knn.predict(train,test,y,kdist)
cm <- confusionMatrix(y[test],preds)

accuracy_orig <- cm$overall['Accuracy']

  runs <- 100
  accuracy_error <- c()
  for(i in 1:runs){
    kdist_error <- injectError(kdist, 0.02)
    preds_error <- knn.predict(train,test,y,kdist_error)
    cm_error <- confusionMatrix(y[test],preds_error)
    
    accuracy_error <- c(accuracy_error, cm_error$overall['Accuracy'])
  }
  
  print(paste("Original accuracy=",accuracy_orig))
  print(paste("     New accuracy=",mean(accuracy_error)))
  print(paste("Std Dev: ",sd(accuracy_error)))
  print(summary(accuracy_error))
  print(length(accuracy_error[accuracy_error >= accuracy_orig]) / runs)
  plot(accuracy_error, cex=0.4,xlab="Run",ylab="Accuracy",pch=21,bg=1)
  abline(h = accuracy_orig, col='red', cex=0.2)
  abline(h = mean(accuracy_error), col='blue', cex=0.2, lty=2)
  abline(h = median(accuracy_error), col='blue', cex=0.2, lty=3)

  
  base=data.frame(X=accuracy_orig2,Y=accuracy_error)
  reg=lm(X~Y,data=base)
  sm=summary(reg)
  mean(sm$residuals^2)
  