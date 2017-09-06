#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table, foreach, sqldf)

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "/Y2Q3/Scripts/Results/"
filename ="Beijing_N10K-100K_E5.0-45.0"
dataset = strsplit(filename, '_')[[1]][1]

files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C1_*.csv"), intern = T)
core1 = data.frame()
foreach(f = files) %do% {
  core1 = rbind(core1, read.csv(f, header = F))
}
core1 = core1[, c(2, 3, 5)]
names(core1) = c("Epsilon", "Dataset", "Time1")
core1 = sqldf("SELECT Epsilon, Dataset, AVG(Time1) AS Time1 FROM core1 GROUP BY 1, 2")

files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C*.csv"), intern = T)
data = data.frame()
foreach(f = files) %do% {
  data = rbind(data, read.csv(f, header = F))
}
data = data[, c(2, 3, 6, 5)]
names(data) = c("Epsilon", "Dataset", "Cores", "Time")

data = sqldf("SELECT Epsilon, Dataset, Cores, AVG(Time) AS Time FROM data GROUP BY 1, 2, 3")
data = sqldf("SELECT Epsilon, Dataset, Cores, Time1/Time AS Scaleup FROM data NATURAL JOIN core1")
data$Dataset = factor(data$Dataset, levels = paste0(seq(10, 100, 10), "K"))
data$Cores = factor(data$Cores)

data = data[data$Epsilon > 5 & data$Epsilon < 50, ]
data = data[data$Cores != 1, ]

temp_title = paste0("(radius of disk in mts) in ", dataset, " dataset.")
title = substitute(paste("Scaleup by ", epsilon) ~ temp_title, list(temp_title = temp_title))
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
pdf(paste0(PHD_HOME,PATH,filename,"_Scaleup.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

title = paste("Speedup in", dataset,"dataset.")
g = ggplot(data=data, aes(x=factor(Epsilon), y=Scaleup, fill=Cores)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
pdf(paste0(PHD_HOME,PATH,filename,"_Speedup.pdf"), width = 10.5, height = 7.5)
plot(g)
dev.off()

setwd(PHD_HOME)
system("git add .", wait=TRUE)
system(paste0("git commit -m 'Uploading scaleup analysis for ", filename, "...'"), wait=TRUE)
system("git pull", wait=TRUE)
system("git push", wait=TRUE)
