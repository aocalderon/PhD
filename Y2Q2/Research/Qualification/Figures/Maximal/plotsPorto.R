#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2)

bfe = read.csv('Beijing_BFE_N10K-90K_E2-10.csv',header = F)
pbfe = read.csv('Beijing_PBFE3_N10K-100K_E2-10.csv',header = F)

bfe = bfe[,c(1,2,3,5)]
pbfe = pbfe[,c(1,2,3,6)]
n = c('Algorithm', 'Epsilon', 'N', 'Time')
names(bfe) = n
names(pbfe) = n
pbfe$Algorithm = 'PBFE'
bfe = bfe[bfe$N!='100K',]
pbfe = pbfe[pbfe$N!='100K',]
bfe = bfe[bfe$N!='10K',]
pbfe = pbfe[pbfe$N!='10K',]
bfe = bfe[bfe$N!='90K',]
pbfe = pbfe[pbfe$N!='90K',]

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

plot(g)