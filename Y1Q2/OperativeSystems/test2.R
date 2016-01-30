library("sqldf", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

test2 <- read.csv("~/Documents/PhD/Y1Q2/AdvancedOperativeSystems/Labs/test2.csv", header=FALSE)
test2 <- test2[, c(15, 19, 20, 21)]
names(test2) <- c('wins1', 'tickets2', 'wins2', 'time')
test2$wins1 <- test2$wins1 - 1119
test2$wins2 <- test2$wins2 - 82
test2$time  <- test2$time - 580
test2 <- sqldf("SELECT AVG(wins1) AS wins1, AVG(wins2) AS wins2, AVG(tickets2) AS tickets2, time FROM test2 GROUP BY time")
test2 <- test2[seq(1,116,5),]
plot(test2$time,test2$wins2,type='l',xlab='Time (secs)',ylab='Lottery wins',col=2)
points(test2$time,test2$wins2,col=2,cex=0.6,pch=22,bg=2)
lines(test2$time,test2$wins1,col=4)
points(test2$time,test2$wins1,col=4, cex=0.6,pch=24,bg=4)
legend('topleft', legend=c("Inflation ticket", "Fixed ticket"),inset=c(0.01,0.02)
       ,col=c(2,4),lty=c(1,1),pch=c(22,24),pt.bg=c(2,4), cex=0.9)