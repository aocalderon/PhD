require(knnflex)
require(caret)

# a quick classification example
n <- 200
set.seed(123)
x1 <- c(rnorm(n/2,mean=2.5),rnorm(n/2,mean=7.5))
x2 <- c(rnorm(n/2,mean=7.5),rnorm(n/2,mean=2.5))
x  <- cbind(x1,x2)
y <- c(rep(1,n/2),rep(0,n/2))
train <- sample(1:n,n*0.75)
test <- (1:n)[-train]
# plot the training cases
plot(x1[train],x2[train],col=y[train]+1,xlab="x1",ylab="x2"
     ,xlim=c(-1,10),ylim=c(-1,10))
# predict the other cases
kdist <- knn.dist(x)
preds <- knn.predict(train,test,y,kdist,k=3,agg.meth="majority")
# add the predictions to the plot
points(x1[test],x2[test],col=as.integer(preds)+1,pch="+")
# display the confusion matrix
confusionMatrix(y[test],preds)