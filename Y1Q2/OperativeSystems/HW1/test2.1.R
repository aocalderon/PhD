data <- read.csv('test2.1.csv', header = F)
data <- data[,c(2,4)]
names(data) <- c('w1','w2')
data$time <- seq(0,200,5)
data$w1 <- data$w1 - data$w1[1]
data$w2 <- data$w2 - data$w2[1]
data <- data[2:nrow(data),]
#pdf('test2.pdf')
plot(data$time,data$w2,type='l',xlab='Time (secs)',ylab='Lottery wins',col=2)
points(data$time,data$w2,col=2,cex=0.6,pch=22,bg=2)
lines(data$time,data$w1,col=4)
points(data$time,data$w1,col=4,cex=0.6,pch=23,bg=4)
legend('topleft', legend=c("Ticket inflation", "Fixed-ticket allocation")
       ,inset=c(0.01,0.02)
       ,col=c(2,4),lty=c(1,1),pch=22:24,pt.bg=c(2,4), cex=0.9)
#dev.off()

