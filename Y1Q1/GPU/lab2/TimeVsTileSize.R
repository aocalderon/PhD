require('sqldf')

# setwd("~/Documents/PhD/Code/Y1Q1/GPU/lab2")
TS04 <- read.csv('times_tiled_50-1000_4.dat', header = F)
TS08 <- read.csv('times_tiled_50-1000_8.dat', header = F)
TS16 <- read.csv('times_tiled_50-1000_16.dat', header = F)
TS32 <- read.csv('times_tiled_50-1000_32.dat', header = F)

i <- rep(seq(50,1000,50), each = 3)

data <- data.frame(i, TS04, TS08, TS16, TS32)
names(data) <- c('I', 'TS04', 'TS08', 'TS16', 'TS32')
sql <- "SELECT AVG(TS04) AS TS04, AVG(TS08) AS TS08, AVG(TS16) AS TS16, AVG(TS32) AS TS32 FROM data GROUP BY I"
data <- sqldf(sql)

pdf('TilesizePerformance_50-1K.pdf')
plot(seq(50,1000,50), data$TS04, pch=0, col=2, 
     xlab="Number of elements N in a NxN matrix.",
     ylab = "Time (s)", 
     main = "TILE_SIZE Performance")
lines(seq(50,1000,50), data$TS04, col=2)
points(seq(50,1000,50), data$TS08, pch=1, col=3)
lines(seq(50,1000,50), data$TS08, col=3)
points(seq(50,1000,50), data$TS16, pch=2, col=4)
lines(seq(50,1000,50), data$TS16, col=4)
points(seq(50,1000,50), data$TS32, pch=3, col=5)
lines(seq(50,1000,50), data$TS32, col=5)
legend('topleft', inset=c(0.01,0.02), title='TILE_SIZE', 
       c("4","8","16","32"), lty=c(1,1,1,1), pch=c(0,1,2,3), col=c(2,3,4,5)) 
dev.off()

###
TS16 <- read.csv('times_tiled_1000-20000_16.dat', header = F)
TS32 <- read.csv('times_tiled_1000-20000_32.dat', header = F)

i <- rep(seq(1000,20000,1000), each = 3)

data <- data.frame(i, TS16, TS32)
names(data) <- c('I', 'TS16', 'TS32')
sql <- "SELECT AVG(TS16) AS TS16, AVG(TS32) AS TS32 FROM data GROUP BY I"
data <- sqldf(sql)

pdf('TilesizePerformance_1K-20K.pdf')
plot(seq(1000,20000,1000), data$TS16, pch=0, col=2, 
     xlab="Number of elements N in a NxN matrix.",
     ylab = "Time (s)", 
     main = "TILE_SIZE Performance")
lines(seq(1000,20000,1000), data$TS16, col=2)
points(seq(1000,20000,1000), data$TS32, pch=1, col=3)
lines(seq(1000,20000,1000), data$TS32, col=3)
legend('topleft', inset=c(0.01,0.02), title='TILE_SIZE', c("16","32"), lty=c(1,1), pch=c(0,1), col=c(2,3)) 
dev.off()

###
NTiled <- read.csv('../lab1/times_no_tiled_1000-20000.dat', header = F)
Tiled  <- read.csv('times_tiled_1000-20000_16.dat', header = F)

s <- seq(1000,20000,1000)
i <- rep(s, each = 3)

data <- data.frame(i, NTiled, Tiled)
names(data) <- c('I', 'NTiled', 'Tiled')
sql <- "SELECT AVG(NTiled) AS NTiled, AVG(Tiled) AS Tiled FROM data GROUP BY I"
data <- sqldf(sql)

pdf('NTVsT_1K-20K.pdf')
plot(s, data$NTiled, pch=0, col=2, 
     xlab="Number of elements N in a NxN matrix.",
     ylab = "Time (s)", 
     main = "No Tiled Vs Tiled Performance")
lines(s, data$NTiled, col=2)
points(s, data$Tiled, pch=1, col=3)
lines(s, data$Tiled, col=3)
legend('topleft', inset=c(0.01,0.02), title='Version', c("No Tiled","Tiled"), lty=c(1,1), pch=c(0,1), col=c(2,3)) 
dev.off()

###
NTiled <- read.csv('../lab1/times_no_tiled_50-1000.dat', header = F)
Tiled  <- read.csv('times_tiled_50-1000_16.dat', header = F)

s <- seq(50,1000,50)
i <- rep(s, each = 3)

data <- data.frame(i, NTiled, Tiled)
names(data) <- c('I', 'NTiled', 'Tiled')
sql <- "SELECT AVG(NTiled) AS NTiled, AVG(Tiled) AS Tiled FROM data GROUP BY I"
data <- sqldf(sql)

pdf('NTVsT_50-1K.pdf')
plot(s, data$NTiled, pch=0, col=2, 
     xlab="Number of elements N in a NxN matrix.",
     ylab = "Time (s)", 
     main = "No Tiled Vs Tiled Performance")
lines(s, data$NTiled, col=2)
points(s, data$Tiled, pch=1, col=3)
lines(s, data$Tiled, col=3)
legend('topleft', inset=c(0.01,0.02), title='Version', c("No Tiled","Tiled"), lty=c(1,1), pch=c(0,1), col=c(2,3)) 
dev.off()