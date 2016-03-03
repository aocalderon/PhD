allpools = read.csv('allpools.txt', header = F)
allpools$V4 = as.factor(allpools$V4)
names(allpools) = c('R','G','B','P')

model = 'knn' # knn - svm - ctree - naive - rpart - mlp

M=fit(P~.,allpools,model=model, task='c')
P=predict(M,finalpools)
thepredictions = P

finalpools = read.csv('newfinal.txt', header = F)
finalpools = finalpools[,3:5]
names(finalpools) = c('R','G','B')

thelabels = c(0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0)
thelabels=as.factor(thelabels)
print(mmetric(thelabels,thepredictions,"CONF")$conf) # confusion matrix

M=fit(P~.,allpools,model=model, task='p')
P=predict(M,finalpools)
print(mmetric(thelabels,P,"AUC")) # AUC

thescores = P[,2]
pred <- prediction(thescores, thelabels)
perf <- performance(pred,"tpr","fpr")
plot(perf,lwd=1,lty=2)
pred <- prediction(thescores, thelabels)
perf <- performance(pred,"rch")
plot(perf,add=TRUE,lwd=2,col=2)

#data(iris)
#M=fit(Species~.,iris,model="knn")
#P=predict(M,iris)
#print(mmetric(iris$Species,P,"CONF")) # confusion matrix
