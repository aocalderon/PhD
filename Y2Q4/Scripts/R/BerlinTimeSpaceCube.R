#!/usr/bin/Rscript
library(data.table)
library(plotly)

###################
# Reading data...
###################

#data = read.csv('/opt/Datasets/Berlin/berlin.csv', header = F)
berlin = as.data.table(data[,c(2,3,4,1)])
names(berlin) = c('id','x','y','t')
berlin = berlin[berlin$t >= 117, ]

###################
# Computing number of point per timestamp...
###################

# berlin[ , `:=`( count = .N , idx = 1:.N ) , by = t ]
# mean(berlin$count)
# [1] 18433.02

###################
# Prunning duplicates...
###################

berlin = berlin[ , list(id = min(id)), by = c('x', 'y', 't')]

###################
# Render scatterplot 3D...
###################

p <- plot_ly(berlin, x = ~x, y = ~y, z = ~t, color = ~t) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Lon'),
                      yaxis = list(title = 'Lat'),
                      zaxis = list(title = 'Time')))
p