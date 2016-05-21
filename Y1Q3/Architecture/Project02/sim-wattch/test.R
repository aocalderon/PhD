interval = 10000
target = 500000
increment = 0.1

turnoff = FALSE
command = paste0("./sim-outorder -DVFSInterval ",interval," -DVFSTargetPower ",target," -DVFSTurnOff ",turnoff," -DVFSIncrement ",increment," ../benchmarks/go.alpha")
system(command)
data = read.table('test.txt', sep = ':')
plot(1:nrow(data), data$V1, type = 'l', xlab='Iteration', ylab='Total Power per Cycle',
     col=4)
abline(h=mean(data$V1), col=4, lwd=1, lty=2)
abline(h=target, col=2, lwd=1, lty=2)
DVFSOn = data


turnoff = TRUE
command = paste0("./sim-outorder -DVFSInterval ",interval," -DVFSTargetPower ",target," -DVFSTurnOff ",turnoff," -DVFSIncrement ",increment," ../benchmarks/go.alpha")
system(command)
data = read.table('test.txt', sep = ':')
plot(1:nrow(data), data$V1, type = 'l', xlab='Iteration', ylab='Total Power per Cycle',
     col=4)
abline(h=mean(data$V1), col=4, lwd=1, lty=2)
abline(h=target, col=2, lwd=1, lty=2)
DVFSOff = data

print(paste0("DVFS Controller ON"))
print(summary(DVFSOn))
print(paste0("DVFS Controller OFF"))
print(summary(DVFSOff))
