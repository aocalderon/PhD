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


# the iris example used by knn(class)
library(class)
data(iris3)
train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
# how to get predictions from knn(class)
pred<-knn(train, test, cl, k = 3)
# display the confusion matrix
table(pred,cl)

# how to get predictions with knn.dist and knn.predict
x <- rbind(train,test)
kdist <- knn.dist(x)
pred <- knn.predict(1:75, 76:150, cl, kdist, k=3)
# display the confusion matrix
table(pred,cl)

# note any small differences are a result of both methods
# breaking ties in majority class randomly


# 5-fold cross-validation to select k for above example
fold <- sample(1:5,75,replace=TRUE)
cvpred <- matrix(NA,nrow=75,ncol=10)
for (k in 1:10)
  for (i in 1:5)
    cvpred[which(fold==i),k] <- knn.predict(train=which(fold!=i),test=which(fold==i),cl,kdist,k=k)
# display misclassification rates for k=1:10
apply(cvpred,2,function(x) sum(cl!=x))

