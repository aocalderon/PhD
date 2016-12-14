library(ggplot2)

data <- read.csv("Cologne_PBFE_N10M-40M_E1.csv", header = F)
local <- data[data$V6 == 0,]
node1 <- data[data$V6 == 1,]
node1$V7 <- local$V5 / node1$V5
node2 <- data[data$V6 == 2,]
node2$V7 <- local$V5 / node2$V5
node3 <- data[data$V6 == 3,]
node3$V7 <- local$V5 / node3$V5
node4 <- data[data$V6 == 4,]
node4$V7 <- local$V5 / node4$V5
data <- rbind(node1,node2,node3,node4)
rm(node4,node3,node2,node1)

data <- data[,c(2,3,5,6,7)]
names(data) = c("Epsilon", "Dataset", "Time", "Nodes", "Speedup")

reorder = c("20M", "30M", "40M", "50M", "60M", "70M")
data$Dataset = factor(data$Dataset, levels = reorder)

title="Speedup in Cologne dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Speedup, fill=factor(Nodes))) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf("Cologne_Speedup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()

title="Scaleup in Cologne dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=factor(Nodes))) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")), y="Time(sec)")
pdf("Cologne_Scaleup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()
