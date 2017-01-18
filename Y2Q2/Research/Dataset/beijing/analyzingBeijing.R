library(ggplot2)

data <- data.frame()
for(i in seq(0, 14)){
  filename <- paste0("Beijing_PBFE_N50K-100K_E100-200_C",i,".csv")
  temp <- read.csv(filename, header = F)
  temp <- temp[, c(2, 3, 5)]
  temp$Tag <- i
  data <- rbind(data, temp)
}

names(data) <- c("Epsilon", "N", "Time", "Tag")
data$N <- factor(data$N, levels = c("50K","60K","70K","80K","90K","100K"))
COMBINATIONS <- c('0','1','2','3','0 1','0 2','0 3','1 2','1 3','2 3','0 1 2','0 1 3','0 2 3','1 2 3','0 1 2 3')


data1 <- data[data$Tag < 4, ]
legend_title = "Racks"
breaks = c("0","1","2","3")
labels = c("10", "11", "12", "14")

title=expression(paste("Execution time by Racks and ",epsilon," (radius of disk in mts) in Beijing dataset."))
g = ggplot(data=data1, aes(x=factor(N), y=Time, group=factor(Tag), colour=factor(Tag), shape=factor(Tag))) +
  geom_line(aes(linetype=factor(Tag))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  scale_fill_discrete(name = "Nodes") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title, breaks = breaks, labels = labels) +
  scale_shape_discrete(name = legend_title,  breaks = breaks, labels = labels) +
  scale_linetype_discrete(name = legend_title, breaks = breaks, labels = labels)
plot(g)

data2 <- data[4 <= data$Tag & data$Tag < 10, ]
legend_title = "Racks"
breaks = c("4","5","6","7","8","9")
labels = c("10 and 11","10 and 12","10 and 14","11 and 12","11 and 14","12 and 14")

title=expression(paste("Execution time by Racks and ",epsilon," (radius of disk in mts) in Beijing dataset."))
g = ggplot(data=data2, aes(x=factor(N), y=Time, group=factor(Tag), colour=factor(Tag), shape=factor(Tag))) +
  geom_line(aes(linetype=factor(Tag))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  scale_fill_discrete(name = "Nodes") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title, breaks = breaks, labels = labels) +
  scale_shape_discrete(name = legend_title,  breaks = breaks, labels = labels) +
  scale_linetype_discrete(name = legend_title, breaks = breaks, labels = labels)
plot(g)

data3 <- data[10 <= data$Tag & data$Tag < 14, ]
legend_title = "Racks"
breaks = c("10","11","12","13")
labels = c("10, 11 and 12","10, 11 and 14","10, 12 and 14","11, 12 and 14")

title=expression(paste("Execution time by Racks and ",epsilon," (radius of disk in mts) in Beijing dataset."))
g = ggplot(data=data3, aes(x=factor(N), y=Time, group=factor(Tag), colour=factor(Tag), shape=factor(Tag))) +
  geom_line(aes(linetype=factor(Tag))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  scale_fill_discrete(name = "Nodes") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title, breaks = breaks, labels = labels) +
  scale_shape_discrete(name = legend_title,  breaks = breaks, labels = labels) +
  scale_linetype_discrete(name = legend_title, breaks = breaks, labels = labels)
plot(g)
