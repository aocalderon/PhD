#!/usr/bin/env Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table, foreach, sqldf)

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y2Q4/Results/Scaleup/"
filename ="Berlin_N50K-150K_E10.0-100.0"

files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C*.csv"), intern = T)
data = data.frame()
foreach(f = files) %do% {
  data = rbind(data, read.csv(f, header = F))
}
data = data[, c(2, 3, 5, 6, 7)]
names(data) = c("Epsilon", "Dataset", "TimeA", "TimeB", "Cores")
data$Time = data$TimeA + data$TimeB
data = sqldf("SELECT Epsilon, Dataset, Cores, AVG(Time) AS Time FROM data GROUP BY 1, 2, 3")
data05 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '50K' AND Cores = 5")
data10 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '100K' AND Cores = 10")
data15 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '150K' AND Cores = 15")
data = rbind(data05, data10, data15)
data$Cores = factor(data$Cores)
data = data[data$Epsilon > 10, ]

###

temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Scaleup by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Cores)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, x=expression(paste(epsilon,"(mts)")))
plot(g)
