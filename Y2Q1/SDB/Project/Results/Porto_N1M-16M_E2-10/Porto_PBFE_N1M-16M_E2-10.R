library(ggplot2)

data <- read.csv("Porto_PBFE_N1M-16M_E2-10.csv", header = F)

data <- data[,c(1,2,3,5)]
names(data) = c("Algorithm","Epsilon", "Dataset", "Time")
reorder = c("1M","2M","4M","8M","16M")
data$Dataset = factor(data$Dataset, levels = reorder)

title=expression(paste("Execution time by ",epsilon," (radius of disk in mts) in Porto dataset."))
g = ggplot(data=data, aes(x=factor(Dataset), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
  geom_line(aes(linetype=Algorithm)) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon)

pdf("Porto_PBFE_N1M-16M_E2-10.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()