#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table)
args = commandArgs(trailingOnly = TRUE)

filename ="Beijing_N10K-100K_E5.0-50.0"
core1 = read.csv("Results/Beijing_N10K-100K_E5.0-50.0_NODE1.csv")
core1 = core1[, c(2, 3, 5, 1)]
names(core1) = c("Epsilon", "Dataset", "Time", "Nodes")

data1 = read.csv("Results/Beijing_N10K-100K_E5.0-50.0_NODE1.csv")
data1 = data1[, c(2, 3, 5, 1)]
names(data1) = c("Epsilon", "Dataset", "Time", "Nodes")
data1$Nodes = "1"
data1$Scaleup = core1$Time / data1$Time
data2 = read.csv("Results/Beijing_N10K-100K_E5.0-50.0_NODE2.csv")
data2 = data2[, c(2, 3, 5, 1)]
names(data2) = c("Epsilon", "Dataset", "Time", "Nodes")
data2$Nodes = "2"
data2$Scaleup = core1$Time / data2$Time
data3 = read.csv("Results/Beijing_N10K-100K_E5.0-50.0_NODE3.csv")
data3 = data3[, c(2, 3, 5, 1)]
names(data3) = c("Epsilon", "Dataset", "Time", "Nodes")
data3$Nodes = "3"
data3$Scaleup = core1$Time / data3$Time

data = rbind(data1, data2, data3)
data$Dataset <- factor(data$Dataset, levels = paste0(seq(10, 100, 10), "K"))
temp_title = paste("(radius of disk in mts) in Beijing dataset.")
title = substitute(paste("Execution time by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data = data, aes(x = factor(Dataset), y = Scaleup, group = Nodes, colour = Nodes, shape = Nodes)) + 
    geom_line(aes(linetype = Nodes)) + 
    geom_point(size = 2) + 
    labs(title = title, y = "Scaleup") + 
    scale_x_discrete("Number of points", breaks = paste0(seq(20, 100, 20), "K")) +
    theme(axis.text.x = element_text(size = 8, angle = 90), axis.text.y = element_text(size = 8)) + 
    facet_wrap(~Epsilon) + 
    scale_colour_discrete() + 
    scale_shape_discrete() + 
    scale_linetype_discrete()
pdf(paste0("~/Documents/PhD/Code/Y2Q3/Plots/", filename, "_Scaleup1.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

title = paste("Scaleup in", filename,"dataset.")
g = ggplot(data=data, aes(x=factor(Epsilon), y=Scaleup, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf(paste0("~/Documents/PhD/Code/Y2Q3/Plots/", filename,"_Scaleup2.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

#setwd("~/Documents/PhD/Code")
#system("git add .", wait=TRUE)
#system(paste0("git commit -m 'Uploading ", filename, "...'"), wait=TRUE)
#system("git pull", wait=TRUE)
#system("git push", wait=TRUE)
