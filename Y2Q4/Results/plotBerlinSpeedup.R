#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table, foreach, sqldf)

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y2Q4/Results/Speedup/"
filename ="Berlin_N160K-160K_E10.0-100.0"
files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C1_*.csv"), intern = T)
core1 = data.frame()
foreach(f = files) %do% {
  core1 = rbind(core1, read.csv(f, header = F))
}
core1 = core1[, c(2, 3, 5, 6)]
names(core1) = c("Epsilon", "Dataset", "TimeA", "TimeB")
core1$Time1 = core1$TimeA + core1$TimeB
core1 = sqldf("SELECT Epsilon, Dataset, AVG(Time1) AS Time1 FROM core1 GROUP BY 1, 2")

files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C*.csv"), intern = T)
data = data.frame()
foreach(f = files) %do% {
  data = rbind(data, read.csv(f, header = F))
}
data = data[, c(2, 3, 5, 6, 7)]
names(data) = c("Epsilon", "Dataset", "TimeA", "TimeB", "Cores")
data$Time = data$TimeA + data$TimeB
data = sqldf("SELECT Epsilon, Dataset, Cores, AVG(Time) AS Time FROM data GROUP BY 1, 2, 3")
data = sqldf("SELECT Epsilon, Dataset, Cores, Time1/Time AS Speedup FROM data NATURAL JOIN core1")
data$Cores = factor(data$Cores)

data = data[data$Epsilon > 10, ]
data = data[data$Cores != 1, ]

###

temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Speedup by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Epsilon), y=Speedup, fill=Cores)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
plot(g)
