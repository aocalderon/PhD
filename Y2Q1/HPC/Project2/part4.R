data <- read.csv('results03.csv', header = F)
x <- c(4,8,16,32,64,128,256)
cex <- 0.8
bg <- 2
pch <- 21
pdf('part4.pdf')
plot(x, data$V2,ylab='Time (sec)',xlab='Block size',type='l',axes=F,col=bg,ylim = c(5,125))
points(x,data$V2,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 4
pch <- 22
lines(x, data$V3,col=bg)
points(x,data$V3,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 3
pch <- 23
lines(x, data$V4,col=bg)
points(x,data$V4,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 5
pch <- 24
lines(x, data$V5,col=bg)
points(x,data$V5,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 6
pch <- 25
lines(x, data$V6,col=bg)
points(x,data$V6,cex=cex,pch=pch,bg=bg,col=bg)
box()
axis(1,at=x,labels=x,cex.axis = 0.8)
axis(2)
legend('topright', legend=c("default","gcc-4.7.2","gcc-4.7.2 -O1","gcc-4.7.2 -O2","gcc-4.7.2 -O3"),
       inset=c(0.01,0.02),col=c(2,4,3,5,6),lty=c(1,1,1,1,1),
       pch=21:25,pt.bg=c(2,4,3,5,6), cex=0.7)
dev.off()
