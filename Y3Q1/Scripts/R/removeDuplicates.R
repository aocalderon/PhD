#!/usr/bin/Rscript
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, sqldf)

###################
# Setting global variables...
###################

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Datasets/"
EXTENSION = ".csv"
DATASET = "B80K"
filename = paste0(PHD_HOME,PATH,DATASET,EXTENSION)
data = read.table(filename, header = F, sep = ',')

###################
# Reading data...
###################

data = as.data.table(data)
names(data) = c('id', 'x', 'y')
data$t = 0

###################
# Prunning possible duplicates...
###################

data = data[ , list(id = min(id)), by = c('x', 'y', 't')]

###################
# Writing back...
###################

write.table(data[ , c('id', 'x', 'y')]
          , file = paste0(PHD_HOME,PATH,DATASET,"_PFlock.csv")
          , row.names = F
          , col.names = F
          , sep = ','
          , quote = F)
write.table(data[ , c('id', 'x', 'y', 't')]
            , file = paste0(PHD_HOME,PATH,DATASET,"_BFE.tsv")
            , row.names = F
            , col.names = F
            , sep = '\t'
            , quote = F)
write.table(data[ , c('id', 't', 'x', 'y')]
            , file = paste0(PHD_HOME,PATH,DATASET,"_FPFlock.tsv")
            , row.names = F
            , col.names = F
            , sep = '\t'
            , quote = F)
