#!/usr/bin/Rscript
library(data.table)
library(sp)
library(rgdal)

###################
# Setting global variables...
###################

LAYERS = 15
FIRST_LAYER = 117
WGS84 = "+init=epsg:4326"
DHDN = "+init=epsg:3068"
DATASET = "/opt/Datasets/Berlin/berlin.csv"
# DATASET = "~/Datasets/Berlin/berlin.csv"
OUTPUT = "Berlin"
SAMPLE_SIZE = 15000

###################
# Reading data...
###################

# data = read.csv(DATASET, header = F)
berlin = as.data.table(data[,c(3,4,1,2)])
names(berlin) = c('x','y','t','id')

###################
# Taking a sample...
###################

berlin = berlin[berlin$t >= FIRST_LAYER, c('x', 'y', 't', 'id')]
berlin = berlin[berlin$t < FIRST_LAYER + LAYERS,]
berlin = berlin[sample(1:nrow(berlin), SAMPLE_SIZE),]
berlin = berlin[order(t),]
head(berlin)

###################
# Prunning possible duplicates...
###################

berlin = berlin[ , list(id = min(id)), by = c('x', 'y', 't')]

###################
# Computing number of point per timestamp...
###################

berlin[ , `:=`( count = .N , idx = 1:.N ) , by = t ]
avg = mean(berlin$count)
print(avg)
# [1] 18433.02

###################
# Reprojecting locations...
###################

berlin = berlin[, c('x', 'y', 't', 'id')]
coordinates(berlin) = ~x+y
proj4string(berlin) <- CRS(WGS84)
berlin = spTransform(berlin, CRS(DHDN))

###################
# writing output...
###################

if(avg < 1000){A=paste0(as.integer(avg))}else if(avg < 1000000){A=paste0(as.integer(avg/1000),'K')}else if(avg < 1000000000){A=paste0(as.integer(avg/1000000),'M')} 
N = nrow(berlin)
if(N < 1000){N=paste0(N,'')} else if(N < 1000000){N=paste0(as.integer(N/1000),'K')} else if(N < 1000000000){N=paste0(as.integer(N/1000000),'M')} else {N=paste0(as.integer(N/1000000000),'G')}
filename = paste0(OUTPUT,'_N',N,'_A',A,'_T',LAYERS,'.csv')
write.table(cbind(coordinates(berlin), berlin@data)
          , file = filename
          , row.names = F
          , col.names = F
          , sep = ','
          , quote = F)
