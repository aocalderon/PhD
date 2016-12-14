library(ggplot2)

local <- read.csv("Beijing_PBFE_LOCAL_N10K-100K_E10-200.csv", header = F)
node4 <- read.csv("Beijing_PBFE_NODE4_N10K-100K_E10-200.csv", header = F)
node4$V6 <- "4" 
node4$V7 <- local$V5 / node4$V5  
node4$V8 <- local$V5 
node3 <- read.csv("Beijing_PBFE_NODE3_N10K-100K_E10-200.csv", header = F)
node3$V6 <- "3" 
node3$V7 <- local$V5 / node3$V5  
node3$V8 <- local$V5
node2 <- read.csv("Beijing_PBFE_NODE2_N10K-100K_E10-200.csv", header = F)
node2$V6 <- "2" 
node2$V7 <- local$V5 / node2$V5  
node2$V8 <- local$V5
node1 <- read.csv("Beijing_PBFE_NODE1_N10K-100K_E10-200.csv", header = F)
node1$V6 <- "1" 
node1$V7 <- local$V5 / node1$V5  
node1$V8 <- local$V5
data <- rbind(node1,node2,node3,node4)
write.table(data, file = 'Beijing.csv', col.names = F, row.names = F, sep = ',', quote = F)
rm(node4,node3,node2,node1)

data <- data[,c(2,3,5,6,7)]
names(data) = c("Epsilon", "Dataset", "Time", "Nodes", "Speedup")
# data <- data[data$Epsilon == 200,]

reorder = c("10K", "20K", "30K", "40K", "50K", "60K", "70K", "80K", "90K", "100K")
data$Dataset = factor(data$Dataset, levels = reorder)
#data <- data[data$Dataset == "50K",]

#data <- data[data$Epsilon >= 150,]
title="Speedup in Beijing dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Speedup, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf("Beijing_Speedup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()

title="Scaleup in Beijing dataset."
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")), y="Time(sec)")
pdf("Beijing_Scaleup.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()
