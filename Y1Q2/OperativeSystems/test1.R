library("sqldf", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

test1 <- read.csv("~/Documents/PhD/Y1Q2/AdvancedOperativeSystems/Labs/test1.csv", header=FALSE)
test1 <- test1[, c(15, 20, 25, 26)]
names(test1) <- c('wins1', 'wins3', 'wins8', 'time')
test1$wins1 <- test1$wins1 - 930
test1$wins3 <- test1$wins3 - 606
test1$wins8 <- test1$wins8 - 155
test1$time  <- test1$time  - 2012
test1 <- sqldf("SELECT AVG(wins1) AS wins1, AVG(wins3) AS wins3, AVG(wins8) AS wins8, time FROM test1 GROUP BY time")
test1 <- test1[seq(1,91,5),]
plot(test1$time,test1$wins8,type='l',xlab='Time (secs)',ylab='Lottery wins',col=2)
points(test1$time,test1$wins8,col=2,cex=0.6,pch=22,bg=2)
lines(test1$time,test1$wins3,col=3)
points(test1$time,test1$wins3,col=3,cex=0.6,pch=23,bg=3)
lines(test1$time,test1$wins1,col=4)
points(test1$time,test1$wins1,col=4,cex=0.6,pch=24,bg=4)
legend('topleft', legend=c("8 tickets", "3 tickets","1 ticket"),inset=c(0.01,0.02)
       ,col=2:4,lty=c(1,1,1),pch=22:24,pt.bg=2:4, cex=0.9)
