#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2)

bfe = read.csv('Beijing_BFE_N20K-80K_E2-10.csv',header = F)
pbfe = read.csv('Beijing_PBFE3_N20K-80K_E2-10.csv',header = F)

bfe = bfe[,c(1,2,3,5)]
pbfe = pbfe[,c(1,2,3,4)]
n = c('Algorithm', 'Epsilon', 'N', 'Time')
names(bfe) = n
names(pbfe) = n
pbfe$Algorithm = 'PBFE'

data <- rbind(bfe, pbfe)
legend_title = "Algorithm"
temp_title = paste("(radius of disk in mts) in Beijing dataset.")
title = substitute(paste("Execution time by ",epsilon) ~temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(N), y=Time, group=factor(Algorithm), colour=factor(Algorithm), shape=factor(Algorithm))) +
  geom_line(aes(linetype=factor(Algorithm))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
pdf('beijing.pdf', width = 10.5, height = 7.5)
plot(g)
dev.off()

##############################################################################################################

bfe = read.csv('Porto_BFE_N50K-400K_E5-10.csv',header = F)
pbfe = read.csv('Porto_PBFE3_N50K-400K_E5-10.csv',header = F)

bfe = bfe[,c(1,2,3,5)]
pbfe = pbfe[,c(1,2,3,5)]
n = c('Algorithm', 'Epsilon', 'N', 'Time')
names(bfe) = n
names(pbfe) = n
pbfe$Algorithm = 'PBFE'

data <- rbind(bfe, pbfe)
reorder = c('50K','100K','150K','200K','250K','300K','350K','400K')
data$N = factor(data$N, levels = reorder)
legend_title = "Algorithm"
temp_title = paste("(radius of disk in mts) in Porto dataset.")
title = substitute(paste("Execution time by ",epsilon) ~temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(N), y=Time, group=factor(Algorithm), colour=factor(Algorithm), shape=factor(Algorithm))) +
  geom_line(aes(linetype=factor(Algorithm))) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete(name = legend_title) +
  scale_shape_discrete(name = legend_title) +
  scale_linetype_discrete(name = legend_title)
pdf('porto.pdf', width = 10.5, height = 7.5)
plot(g)
dev.off()