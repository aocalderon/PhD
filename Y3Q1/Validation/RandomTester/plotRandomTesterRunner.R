#! /usr/bin/env Rscript

## Downloadig needed libraries if needed...
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, sqldf, stringr)

## Getting input file name as parameter...
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file)", call.=FALSE)
} 

## Reading the data...
path = args[1]
lines = readLines(path)
data0 = c()
for(line in lines){
  if(grepl("^0\\d\\.", line, perl = T)){
    data0 = c(data0, line)
  }
}
data1 = as.data.frame(str_split_fixed(data0, ",", 5))
names(data1) = c("Stage", "Time", "Results", "Epsilon", "Cores")

## Taking average time for all runs...
data2 = sqldf("SELECT Stage, Epsilon, Cores, avg(Time) AS Time FROM data1 GROUP BY 1, 2, 3")
data2$Cores = factor(data2$Cores)

## Plotting data...
temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Execution time by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=data2, aes(x=factor(Epsilon), y=Time, fill=Cores)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, y="Time(s)", x=expression(paste(epsilon,"(mts)"))) +
  facet_wrap(~Stage)
pdf("plot.pdf")
plot(g)
dev.off()