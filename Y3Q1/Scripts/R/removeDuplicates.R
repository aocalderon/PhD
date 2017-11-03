#!/usr/bin/Rscript
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, sqldf)

###################
# Setting global variables...
###################

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Datasets/"
EXTENSION = ".csv"
DATASET = "B20K"
filename = paste0(PHD_HOME,PATH,DATASET,EXTENSION)
data = read.table(filename, header = F, sep = ',')

###################
# Reading data...
###################

data = as.data.table(data)
names(data) = c('x', 'y', 't', 'id')
data = data[ , c('id', 'x', 'y', 't')]

###################
# Prunning possible duplicates...
###################

data = data[ , list(id = min(id)), by = c('x', 'y', 't')]
data = data[ , c('id', 'x', 'y', 't')]

###################
# Writing back...
###################

write.table(data
          , file = paste0(PHD_HOME,PATH,DATASET,".csv")
          , row.names = F
          , col.names = F
          , sep = ','
          , quote = F)
write.table(data
            , file = paste0(PHD_HOME,PATH,DATASET,".tsv")
            , row.names = F
            , col.names = F
            , sep = '\t'
            , quote = F)
