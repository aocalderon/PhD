data <- read.csv('test1.1.csv', header = F)
data <- data[,c(4,6,2)]
names(data) <- c('w1','w3','w8')
data$time <- seq(0,200,5)
data$w1 <- data$w1 - data$w1[1]
data$w3 <- data$w3 - data$w3[1]
data$w8 <- data$w8 - data$w8[1]
data <- data[2:nrow(data),]
pdf('test1a.pdf')
plot(data$time,data$w8,type='l',xlab='Time (secs)',ylab='Lottery wins',col=2)
points(data$time,data$w8,col=2,cex=0.6,pch=22,bg=2)
lines(data$time,data$w3,col=3)
points(data$time,data$w3,col=3,cex=0.6,pch=23,bg=3)
lines(data$time,data$w1,col=4)
points(data$time,data$w1,col=4,cex=0.6,pch=24,bg=4)
legend('topleft', legend=c("8 tickets", "3 tickets","1 ticket"),inset=c(0.01,0.02)
       ,col=2:4,lty=c(1,1,1),pch=22:24,pt.bg=2:4, cex=0.9)
dev.off()
data$total <- data$w1 + data$w3 + data$w8
data$p1 <- data$w1 / data$total
data$p3 <- data$w3 / data$total
data$p8 <- data$w8 / data$total
pdf('test1b.pdf')
plot(data$time,data$p1,ylim=c(0.05,0.7),pch=24,bg=4,cex=0.5,col=4
     ,xlab='Time (sec)',ylab='Proportion share')
abline(h=(1/12),col=4)
text(150,0.12,'1 ticket (p=0.083)',cex=0.8,col=4)
points(data$time,data$p3,pch=23,bg=3,cex=0.5,col=3)
abline(h=(3/12),col=3)
text(150,0.29,'3 tickets (p=0.250)',cex=0.8,col=3)
points(data$time,data$p8,pch=22,bg=2,cex=0.5,col=2)
abline(h=(8/12),col=2)
text(150,0.62,'8 tickets (p=0.666)',cex=0.8,col=2)
dev.off()