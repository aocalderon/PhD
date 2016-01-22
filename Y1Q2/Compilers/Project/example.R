#a quick classification example
# a quick classification example
x1 <- c(rnorm(5,mean=1),rnorm(5,mean=5))
x2 <- c(rnorm(5,mean=5),rnorm(5,mean=1))
x  <- cbind(x1,x2)
y <- c(rep(1,5),rep(0,5))
train <- sample(1:10,7)
# plot the training cases
plot(x1[train],x2[train],col=y[train]+1,xlab="x1",ylab="x2")
# predict the other cases
test <- (1:10)[-train]
kdist <- knn.dist(x)
preds <- knn.predict(train,test,y,kdist,k=3,agg.meth="majority")
# add the predictions to the plot
points(x1[test],x2[test],col=as.integer(preds)+1,pch="+")
# display the confusion matrix
table(y[test],preds)