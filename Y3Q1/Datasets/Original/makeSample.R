#!/usr/bin/Rscript
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, sqldf)

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Datasets/Original/"
EXTENSION = ".csv"
DATASET = "B160K_3068"
filename = paste0(PHD_HOME,PATH,DATASET,EXTENSION)
data0 = read.table(filename, header = F, sep = ',')

nSample = 60000
nData0 = nrow(data0)
nDecimals = 2
x = sample(1:nData0, nSample)
data = as.data.table(data0[x,])
data$V2 = round(data$V2, nDecimals)
data$V3 = round(data$V3, nDecimals)
names(data) = c('id', 'x', 'y')
data$t = 0
data = data[ , list(id = min(id)), by = c('x', 'y', 't')]

DATASET = "B60K"
write.table(data[ , c('id', 'x', 'y')]
            , file = paste0(PHD_HOME,PATH,DATASET,".csv")
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

