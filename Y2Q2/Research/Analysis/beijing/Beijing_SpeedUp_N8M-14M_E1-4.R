#!/usr/bin/env Rscript

library(ggplot2)
library(sqldf)
library(Hmisc)

DATASET = "beijing"
DATASET_TAG = capitalize(DATASET)

data <- data.frame()
for(i in seq(0, 14)){
  filename <- paste0("../",DATASET,"1/", DATASET_TAG,"_PBFE_N8M-14M_E1-4_C",i,".csv")
  temp <- read.csv(filename, header = F)
  temp <- temp[, c(2, 3, 5)]
  temp$Tag <- i
  data <- rbind(data, temp)
}

for(i in seq(0, 14)){
  filename <- paste0("../",DATASET,"2/", DATASET_TAG,"_PBFE_N8M-14M_E1-4_C",i,".csv")
  temp <- read.csv(filename, header = F)
  temp <- temp[, c(2, 3, 5)]
  temp$Tag <- i
  data <- rbind(data, temp)
}

names(data) <- c("Epsilon", "N", "Time", "Tag")
data <- sqldf("SELECT Epsilon, N, AVG(Time) AS Time, Tag FROM data GROUP BY Epsilon, N, Tag")
data$N <- factor(data$N, levels = c("8M","10M","12M","14M","16M"))

data1 <- data[data$Tag < 4, ]
data2 <- data[4 <= data$Tag & data$Tag < 10, ]
data3 <- data[10 <= data$Tag & data$Tag < 14, ]
data4 <- data[data$Tag == 14, ]

data1 <- sqldf("SELECT Epsilon, N, AVG(Time) AS Time FROM data1 GROUP BY Epsilon, N")
data1$Nodes = "1" 
data2 <- sqldf("SELECT Epsilon, N, AVG(Time) AS Time FROM data2 GROUP BY Epsilon, N")
data2$Nodes = "2" 
data3 <- sqldf("SELECT Epsilon, N, AVG(Time) AS Time FROM data3 GROUP BY Epsilon, N")
data3$Nodes = "3" 
data4 <- sqldf("SELECT Epsilon, N, AVG(Time) AS Time FROM data4 GROUP BY Epsilon, N")
data4$Nodes = "4" 

data <- rbind(data1, data2, data3, data4)
legend_title = "Nodes"
temp_title = paste("(radius of disk in mts) and", legend_title, "in", DATASET_TAG, "dataset.")
title = substitute(paste("Execution time by ",epsilon) ~temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(N), y=Time, group=factor(Nodes), colour=factor(Nodes), shape=factor(Nodes))) +
  geom_line(aes(linetype=factor(Nodes))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
pdf(paste0("1_",DATASET_TAG,"_Nodes.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

core1 <- read.csv(paste0(DATASET_TAG,"_PBFE_LOCAL_N8M-14M_E1-4.csv"), header = F)
data1$Speedup <- core1$V5 / data1$Time
data2$Speedup <- core1$V5 / data2$Time
data3$Speedup <- core1$V5 / data3$Time
data4$Speedup <- core1$V5 / data4$Time

data <- rbind(data1, data2, data3, data4)

title = paste("Speedup in", DATASET_TAG,"dataset.")
g = ggplot(data=data, aes(x=factor(Epsilon), y=Speedup, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf(paste0("2_",DATASET_TAG,"_Speedup.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

title = paste("Scaleup in", DATASET_TAG,"dataset.")
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")), y="Time(sec)")
pdf(paste0("3_",DATASET_TAG,"_Scaleup.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()
