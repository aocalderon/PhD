pdf("part3.pdf")
data <- read.csv('results01.csv', header = FALSE)
x <- data$V1
cex <- 0.8
bg <- 2
pch <- 21
plot(x, data$V2,ylab='Time (sec)',xlab='Block size',type='l',axes=F,col=bg)
points(x,data$V2,cex=cex,pch=pch,bg=bg,col=bg)
box()
axis(1,at=x,labels=x,cex.axis = 0.8)
axis(2)
abline(h = 94.342688381,col=4,lty=2)
legend('topright', legend=c("Simple triple loop","Blocked version"),
       inset=c(0.01,0.02),col=c(4,2),lty=c(2,1),
       pch=c(NA,21),pt.bg=c(4,2), cex=0.9)
dev.off()
