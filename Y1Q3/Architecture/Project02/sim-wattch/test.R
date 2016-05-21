interval = 1000
target = 60000
command = paste0("./sim-outorder -DVFSInterval ", interval," -DVFSTargetPower ", target," ../benchmarks/go.alpha")
system(command)
data = read.table('test.txt', sep = ':')
summary(data)
plot(1:nrow(data), data$V1, type = 'l', xlab='Iteration', ylab='Total Power per Cycle',
     col=4)
abline(h=mean(data$V1), col=4, lwd=1, lty=2)
abline(h=target, col=2, lwd=1, lty=2)

plot(1:nrow(data), data$V1, type = 'l', xlab='Iteration', ylab='Total Power per Cycle',
     col=4)
abline(h=mean(data$V1), col=4, lwd=1, lty=2)
abline(h=target, col=2, lwd=1, lty=2)
