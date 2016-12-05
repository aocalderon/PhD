library(ggplot2)
library(sqldf)

core1 <- read.csv("Porto_PBFE_LOCAL[1]_N1M-16M_E2-10.csv", header = F)
#sigma <- read.csv("Porto_BFE_N1M-16M_E2-10.csv", header = F)
node4 <- read.csv("Porto_PBFE_N1M-16M_E2-10.csv", header = F)
node4$V6 <- "4" 
node4$V7 <- core1$V5 / node4$V5  
node4$V8 <- core1$V5 
#node4$V9 <- sigma$V5 / node4$V5  
#node4$V10 <- sigma$V5 
node3 <- read.csv("Porto_PBFE_NODE3_N1M-16M_E2-10.csv", header = F)
node3$V6 <- "3" 
node3$V7 <- core1$V5 / node3$V5  
node3$V8 <- core1$V5 
#node3$V9 <- sigma$V5 / node3$V5  
#node3$V10 <- sigma$V5 
node2 <- read.csv("Porto_PBFE_NODE2_N1M-16M_E2-10.csv", header = F)
node2$V6 <- "2" 
node2$V7 <- core1$V5 / node2$V5  
node2$V8 <- core1$V5 
#node2$V9 <- sigma$V5 / node2$V5  
#node2$V10 <- sigma$V5 
node1 <- read.csv("Porto_PBFE_NODE1_N1M-16M_E2-10.csv", header = F)
node1$V6 <- "1" 
node1$V7 <- core1$V5 / node1$V5  
node1$V8 <- core1$V5 
#node1$V9 <- sigma$V5 / node1$V5  
#node1$V10 <- sigma$V5 
data <- rbind(node1,node2,node3,node4)
write.table(data, file = 'Porto_SPEEDUP_N1M-16M_E2-10.csv', col.names = F, row.names = F, sep = ',', quote = F)
rm(node4,node3,node2,node1)

data <- data[,c(2,3,5,6,7)]
names(data) = c("Epsilon", "Dataset", "Time","Nodes", "Speedup")

reorder = c("1M", "2M", "4M", "8M", "16M")
data$Dataset = factor(data$Dataset, levels = reorder)
# data = data[data$Epsilon == 2,]

data <- data[data$Epsilon >= 5,]
title="Speedup in Porto dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Speedup, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf("Porto_Speedup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()

title="Scaleup in Porto dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")), y="Time(sec)")
pdf("Porto_Scaleup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()