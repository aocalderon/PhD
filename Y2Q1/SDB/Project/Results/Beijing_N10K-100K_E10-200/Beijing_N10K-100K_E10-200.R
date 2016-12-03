library(ggplot2)

data <- read.csv("Beijing_N10K-100K_E10-200.csv", header = F)
data <- data[,c(1,2,3,5)]
names(data) = c("Algorithm","Epsilon", "Dataset", "Time")
reorder = c("10K", "20K", "30K", "40K", "50K", "60K", "70K", "80K", "90K", "100K")
data$Dataset = factor(data$Dataset, levels = reorder)

title=expression(paste("Execution time by ",epsilon," (radius of disk in mts) in Beijing dataset."))
g = ggplot(data=data, aes(x=factor(Dataset), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
  geom_line(aes(linetype=Algorithm)) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points", breaks=c("20K","40K","60K","80K", "100K")) +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon)

pdf("Beijing_N10K-100K_E10-200.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()
