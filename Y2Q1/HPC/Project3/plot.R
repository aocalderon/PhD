library(data.table)
library(ggplot2)

data <- read.csv("S_10P10_P10-30_01.csv", header = F)
data <- rbind(data, read.csv("S_10P10_P10-30_02.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_03.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_04.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_05.csv", header = F))
data <- data[,c(1,3,5)]
names(data) = c("S","Cores","Time")
data <- data.table(data)
data <- unique(data[, Time:=mean(Time), by=list(S, Cores)])

g = ggplot(data=data, aes(x=factor(Cores), y=Time, group=S, colour=S, shape=S)) +
  geom_line(aes(linetype=S)) +
  geom_point(size=2.5) +
  labs(x="Number of processors", y="Time (sec)") +
  scale_colour_discrete(name  ="Optimization",
                        breaks=c("S0", "S1","S2","S3"),
                        labels=c("Control", "Remove evens", "Remove Bcast", "Reorder loops")) +
  scale_shape_discrete(name  ="Optimization",
                       breaks=c("S0", "S1","S2","S3"),
                       labels=c("Control", "Remove evens", "Remove Bcast", "Reorder loops")) +
  scale_linetype_discrete(name  ="Optimization",
                          breaks=c("S0", "S1","S2","S3"),
                          labels=c("Control", "Remove evens", "Remove Bcast", "Reorder loops")) 
pdf("plot.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()