library(ggplot2)

filename <- "Beijing_PBFE_N8M-14M_E1-4_C14.csv"
data <- read.csv(filename, header = F)
data <- data[, c(2, 3, 4)]

names(data) <- c("Epsilon", "N", "Disks")
data$N <- factor(data$N, levels = c("8M","10M","12M","14M"))

title=expression(paste("Number of disks by ",epsilon," (radius of disk in mts) in Beijing dataset."))
legend_title=expression(paste(epsilon, "(mts)"))
g = ggplot(data=data, aes(x=factor(N), y=Disks, group=factor(Epsilon), colour=factor(Epsilon))) +
  geom_line(aes(linetype=factor(Epsilon))) +
  geom_point(size=2) +
  labs(title=title,y="Number of disks") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
ggsave(file="beijing.png")

filename <- "Cologne_PBFE_N8M-16M_E1-5_C14.csv"
data <- read.csv(filename, header = F)
data <- data[, c(2, 3, 4)]

names(data) <- c("Epsilon", "N", "Disks")
data$N <- factor(data$N, levels = c("8M","10M","12M","14M","16M"))

title=expression(paste("Number of disks by ",epsilon," (radius of disk in mts) in Cologne dataset."))
legend_title=expression(paste(epsilon, "(mts)"))
g = ggplot(data=data, aes(x=factor(N), y=Disks, group=factor(Epsilon), colour=factor(Epsilon))) +
  geom_line(aes(linetype=factor(Epsilon))) +
  geom_point(size=2) +
  labs(title=title,y="Number of disks") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
ggsave(file="cologne.png")

filename <- "Porto_PBFE_N1M-16M_E2-10.csv"
data <- read.csv(filename, header = F)
data <- data[, c(2, 3, 4)]
names(data) <- c("Epsilon", "N", "Disks")
data$N <- factor(data$N, levels = c("1M","2M","4M","8M","16M"))

title=expression(paste("Number of disks by ",epsilon," (radius of disk in mts) in Porto dataset."))
legend_title=expression(paste(epsilon, "(mts)"))
g = ggplot(data=data, aes(x=factor(N), y=Disks, group=factor(Epsilon), colour=factor(Epsilon))) +
  geom_line(aes(linetype=factor(Epsilon))) +
  geom_point(size=2) +
  labs(title=title,y="Number of disks") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
ggsave(file="porto.png")
