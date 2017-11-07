#!/usr/bin/Rscript
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, sqldf)

###################
# Setting global variables...
###################

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Datasets/"
EXTENSION = ".csv"
DATASET = "olden_PFlock"
filename = paste0(PHD_HOME,PATH,DATASET,EXTENSION)
points = read.csv(filename, header = F)
names(points) = c("id", "x", "y")
itemset = sqldf("SELECT * FROM points WHERE id IN (94, 160, 270)")
x = as.matrix(itemset[,c("x","y")])
dist(x)