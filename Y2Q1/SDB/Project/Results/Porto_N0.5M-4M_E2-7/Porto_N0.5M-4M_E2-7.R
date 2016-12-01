library(ggplot2)

data <- read.csv("Porto_N0.5M-4M_E2-7.csv", header = F)

data <- data[,c(1,2,3,5)]
names(data) = c("Algorithm","Epsilon", "Dataset", "Time")
reorder = c("0.5M","1M","1.5M","2M","2.5M","3M","3.5M","4M")
data$Dataset = factor(data$Dataset, levels = reorder)

title=expression(paste("Execution time by ",epsilon," (radius of disk in mts) in Porto dataset."))
g = ggplot(data=data, aes(x=factor(Dataset), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
  geom_line(aes(linetype=Algorithm)) +
  geom_point(size=2.5) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon)

pdf("Porto_N0.5M-4M_E2-7.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()