#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table)

PHD_HOME = Sys.getenv(c("PHD_HOME"))
filename ="Beijing_N10K-100K_E10.0-45.0"
core1 = read.csv(paste0(PHD_HOME,"/Y2Q3/Scripts/Results/Beijing_N10K-100K_E5.0-45.0_C1_1497233788.csv"), header = F)
core1 = core1[, c(2, 3, 5, 6)]
names(core1) = c("Epsilon", "Dataset", "Time", "Cores")

data1 = read.csv(paste0(PHD_HOME,"/Y2Q3/Scripts/Results/Beijing_N10K-100K_E5.0-45.0_C6_1497233788.csv"), header = F)
data1 = data1[, c(2, 3, 5, 6)]
names(data1) = c("Epsilon", "Dataset", "Time", "Cores")
data1$Nodes = "1"
data1$Scaleup = core1$Time / data1$Time
data2 = read.csv(paste0(PHD_HOME,"/Y2Q3/Scripts/Results/Beijing_N10K-100K_E5.0-45.0_C12_1497214566.csv"), header = F)
data2 = data2[, c(2, 3, 5, 6)]
names(data2) = c("Epsilon", "Dataset", "Time", "Cores")
data2$Nodes = "2"
data2$Scaleup = core1$Time / data2$Time
data3 = read.csv(paste0(PHD_HOME,"/Y2Q3/Scripts/Results/Beijing_N10K-100K_E5.0-45.0_C18_1497206154.csv"), header = F)
data3 = data3[, c(2, 3, 5, 6)]
names(data3) = c("Epsilon", "Dataset", "Time", "Cores")
data3$Nodes = "3"
data3$Scaleup = core1$Time / data3$Time
data4 = read.csv(paste0(PHD_HOME,"/Y2Q3/Scripts/Results/Beijing_N10K-100K_E5.0-45.0_C24_1497201715.csv"), header = F)
data4 = data4[, c(2, 3, 5, 6)]
names(data4) = c("Epsilon", "Dataset", "Time", "Cores")
data4$Nodes = "4"
data4$Scaleup = core1$Time / data4$Time

data = rbind(data1, data2, data3, data4)
data$Dataset = factor(data$Dataset, levels = paste0(seq(10, 100, 10), "K"))
data$Cores = factor(data$Cores)
data = data[data$Epsilon > 5 & data$Epsilon < 50, ]
temp_title = paste("(radius of disk in mts) in Beijing dataset.")
title = substitute(paste("Execution time by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data = data, aes(x = factor(Dataset), y = Scaleup, group = Cores, colour = Cores, shape = Cores)) + 
    geom_line(aes(linetype = Cores)) + 
    geom_point(size = 2) + 
    labs(title = title, y = "Scaleup") + 
    scale_x_discrete("Number of points", breaks = paste0(seq(20, 100, 20), "K")) +
    theme(axis.text.x = element_text(size = 8, angle = 90), axis.text.y = element_text(size = 8)) + 
    facet_wrap(~Epsilon) + 
    scale_colour_discrete() + 
    scale_shape_discrete() + 
    scale_linetype_discrete()
pdf(paste0(PHD_HOME,"/Y2Q3/Plots/", filename, "_Scaleup1.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

title = paste("Scaleup in", filename,"dataset.")
g = ggplot(data=data, aes(x=factor(Epsilon), y=Scaleup, fill=Nodes)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf(paste0(PHD_HOME,"/Y2Q3/Plots/", filename,"_Scaleup2.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

setwd(PHD_HOME)
system("git add .", wait=TRUE)
system(paste0("git commit -m 'Uploading scaleup analysis for ", filename, "...'"), wait=TRUE)
system("git pull", wait=TRUE)
system("git push", wait=TRUE)
