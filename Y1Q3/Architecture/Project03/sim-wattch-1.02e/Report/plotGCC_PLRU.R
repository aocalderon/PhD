library(ggplot2)

Data <- read.csv('gcc_plru.csv', sep = ';', header = F)
Data$V1 <- factor(Data$V1, levels = c("Baseline","PLRU"))
d <- Data[1:2,]
xlab = " "
ylab = "Total simulation time in cycles"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/gcc_plru_1.pdf'))

d <- Data[3:4,]
xlab = " "
ylab = "Cycles per instruction"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/gcc_plru_2.pdf'))

d <- Data[5:6,]
xlab = " "
ylab = "Total power per cycle"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/gcc_plru_3.pdf'))

d <- Data[7:8,]
xlab = " "
ylab = "Average total power per instruction"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/gcc_plru_4.pdf'))
