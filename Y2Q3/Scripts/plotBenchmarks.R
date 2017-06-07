#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table)
args = commandArgs(trailingOnly=TRUE)

filename = args[1]
data <- fread(paste0('http://www.cs.ucr.edu/~acald013/public/Results/', filename))
data = data[, c(2, 3, 5, 1)]

names(data) = c('Epsilon', 'Dataset', 'Time', 'Implementation')
temp_title = paste("(radius of disk in mts) in Beijing dataset.")
title = substitute(paste("Execution time by ",epsilon) ~temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Dataset), y=Time, group=Implementation, colour=Implementation, shape=Implementation)) +
  geom_line(aes(linetype=Implementation)) +
  geom_point(size=2) +
  labs(title=title,y="Time (sec)") + 
  ## scale_x_discrete("Number of points", breaks = paste0(seq(20,100,20),"K")) +
  scale_x_discrete("Number of points") +
  theme(axis.text.x=element_text(size=8, angle=90),axis.text.y=element_text(size=8)) +
  facet_wrap(~Epsilon) +
  scale_colour_discrete() +
  scale_shape_discrete() +
  scale_linetype_discrete()
filename = paste0(strsplit(filename, '\\.')[[1]][1:length(strsplit(filename, '\\.')[[1]]) - 1], collapse = '.')
pdf(paste0("~/Documents/PhD/Code/Y2Q3/Plots/",filename, ".pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()
