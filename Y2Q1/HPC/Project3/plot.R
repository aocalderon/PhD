data <- read.csv("S_10P10_P10-30_01.csv", header = F)
data <- data[,c(1,3,5)]
names(data) = c("Implementation","Cores","Time")
data$Implementation <- as.character(data$Implementation)
data$Implementation[data$Implementation == "S0"] <- "1.Without Optimization"
data$Implementation[data$Implementation == "S1"] <- "2.Removing Evens"
data$Implementation[data$Implementation == "S2"] <- "3.Removing Bcast"
data$Implementation[data$Implementation == "S3"] <- "4.Reorder Looops"
g = ggplot(data=data, aes(x=factor(Cores), y=Time, group=Implementation)) +
  geom_line(aes(linetype=Implementation, color=Implementation)) +
  geom_point(aes(shape=Implementation, color=Implementation)) +
  labs(x="Number of processors", y="Time (sec)")
plot(g)