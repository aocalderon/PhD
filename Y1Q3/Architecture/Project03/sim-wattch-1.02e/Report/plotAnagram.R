library(ggplot2)

Data <- read.csv('anagram_stats.csv', sep = ';', header = F)
Data$V1 <- factor(Data$V1, levels = c("Baseline","Stream buffers","Victim cache"))
d <- Data[1:3,]
xlab = " "
ylab = "Total simulation time in cycles"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/anagram_1.pdf'))

d <- Data[4:6,]
xlab = " "
ylab = "Cycles per instruction"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/anagram_2.pdf'))

d <- Data[7:9,]
xlab = " "
ylab = "Total power per cycle"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/anagram_3.pdf'))

d <- Data[10:12,]
xlab = " "
ylab = "Average total power per instruction"
g <- ggplot(data=d, 
            aes(x=factor(V1), y=V3,group=V2)) + 
  geom_line() + 
  geom_point() +
  scale_x_discrete(xlab) +
  scale_y_continuous(ylab) 
plot(g)
ggsave(paste0('Plots/anagram_4.pdf'))
