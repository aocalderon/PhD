library(ggplot2)

core1 <- read.csv("Beijing_PBFE_LOCAL[1]_N10K-100K_E10-200.csv", header = F)
sigma <- read.csv("Beijing_N10K-100K_E10-200.csv", header = F)
sigma <- sigma[sigma$V1 == "BFE",]
node4 <- read.csv("Beijing_PBFE_NODE4_N10K-100K_E10-200.csv", header = F)
node4$V6 <- "4 Nodes" 
node4$V7 <- core1$V5 / node4$V5  
node4$V8 <- core1$V5 
node4$V9 <- sigma$V5 / node4$V5  
node4$V10 <- sigma$V5 
node3 <- read.csv("Beijing_PBFE_NODE3_N10K-100K_E10-200.csv", header = F)
node3$V6 <- "3 Nodes" 
node3$V7 <- core1$V5 / node3$V5  
node3$V8 <- core1$V5 
node3$V9 <- sigma$V5 / node3$V5  
node3$V10 <- sigma$V5 
node2 <- read.csv("Beijing_PBFE_NODE2_N10K-100K_E10-200.csv", header = F)
node2$V6 <- "2 Nodes" 
node2$V7 <- core1$V5 / node2$V5  
node2$V8 <- core1$V5 
node2$V9 <- sigma$V5 / node2$V5  
node2$V10 <- sigma$V5 
node1 <- read.csv("Beijing_PBFE_NODE1_N10K-100K_E10-200.csv", header = F)
node1$V6 <- "1 Node" 
node1$V7 <- core1$V5 / node1$V5  
node1$V8 <- core1$V5 
node1$V9 <- sigma$V5 / node1$V5  
node1$V10 <- sigma$V5 
data <- rbind(node1,node2,node3,node4)
write.table(data, file = 'Beijing_SPEEDUP_N10K-100K_E10-200.csv', col.names = F, row.names = F, sep = ',', quote = F)
rm(node4,node3,node2,node1)

data <- data[,c(2,3,6,7,9)]
names(data) = c("Epsilon", "Dataset", "Nodes", "Singlecore", "BFE")
data <- data[data$Epsilon == 200,]

reorder = c("10K", "20K", "30K", "40K", "50K", "60K", "70K", "80K", "90K", "100K")
data$Dataset = factor(data$Dataset, levels = reorder)
# data <- data[data$Dataset == "100K",]

title="Speedup by data size (number of points) in Beijing dataset."
g = ggplot(data=data, aes(x=factor(Nodes), y=Singlecore)) +
  geom_bar(stat="identity") +
  facet_wrap(~Dataset)
# g = ggplot(data=data, aes(x=factor(Nodes), y=Singlecore) +
#   geom_bar( stat = "identity" ) +
#   labs(title=title,y="Speedup") + 
#   scale_x_discrete(expression(paste(epsilon, "(mts)")), breaks=seq(20,200,50)) +
#   theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
#   facet_wrap(~Dataset)

