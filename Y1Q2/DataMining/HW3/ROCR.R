require(ROCR)
data(iris)
iris$setosa <- factor(1*(iris$Species == 'setosa'))
iris.rf <- randomForest(setosa ~ ., data=iris[,-5])
summary(predict(iris.rf, iris[,-5]))
summary(iris.preds <- predict(iris.rf, iris[,-5], type = 'prob'))
preds <- iris.preds[,2]
plot(performance(prediction(preds, iris$setosa), 'tpr', 'fpr'))


