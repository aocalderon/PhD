#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, sqldf, Hmisc)

data1 = read.csv('benchmarkPBFE_2016-12-01.csv', header = F)
data1 = data1[data1$V1 == 'PBFE', c(2,3,4,5)]
data1$V1 = 'PFLOCK1'
data2 = read.csv('benchmarkPBFE_2017-05-04.csv', header = F)
data2 = data2[, c(2,3,4,5)]
data2$V1 = 'PFLOCK2'

data = rbind(data1, data2)
names(data) = c('Epsilon', 'Dataset', 'N', 'Time','Implementation')
data$Dataset <- factor(data$Dataset, levels = paste0(seq(10,100,10),"K"))
temp_title = paste("(radius of disk in mts) in Beijing dataset.")
title = substitute(paste("Execution time by ",epsilon) ~temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Dataset), y=Time, group=Implementation, colour=Implementation, shape=Implementation)) +
  geom_line(aes(linetype=Implementation)) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  scale_x_discrete("Number of points", breaks = paste0(seq(20,100,20),"K")) +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete() +
  scale_shape_discrete() +
  scale_linetype_discrete()
pdf(paste0("benchmark.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()
