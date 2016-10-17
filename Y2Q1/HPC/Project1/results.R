data <- read.csv('results3.tsv', sep = '\t', header = FALSE)
x <- c(64, 128, 256, 512, 1024, 2048)
cex <- 0.8
bg <- 2
pch <- 21
pdf('plot.pdf')
plot(x, data$V1,ylab='Time (sec)',xlab='Matrix size',type='l',axes=F,col=bg)
points(x,data$V1,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 4
pch <- 22
lines(x, data$V2,col=bg)
points(x,data$V2,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 3
pch <- 23
lines(x, data$V3,col=bg)
points(x,data$V3,cex=cex,pch=pch,bg=bg,col=bg)
bg <- 5
pch <- 24
lines(x, data$V4,col=bg)
points(x,data$V4,cex=cex,pch=pch,bg=bg,col=bg)
box()
axis(1,at=x,labels=x)
axis(2)
legend('topleft', legend=c("dgemm0","dgemm1","dgemm2","dgemm3"),
       inset=c(0.01,0.02),col=c(2,4,3,5),lty=c(1,1,1,1),
       pch=21:24,pt.bg=c(2,4,3,5), cex=0.9)
dev.off()

#perf1 <- data.frame(dgemm0=data$V1,gflops0=x,dgemm1=data$V2,gflops1=x)
a = (2 * x ^ 3) / data$V1
b = (2 * x ^ 3) / data$V2
plot(x,b, ylim = c(84548990,227358196))
points(x,a,col=2)