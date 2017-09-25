#!/usr/bin/Rscript


library(data.table)
library(plotly)

###################
# Setting credentials...
###################

Sys.setenv("plotly_username"="aocalderon1978")
Sys.setenv("plotly_api_key"="dx4LIeqcXzokLrO2SUHF")

###################
# Reading data...
###################

# data = read.csv('/opt/Datasets/Berlin/berlin.csv', header = F)
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
# Taking a sample...
###################

berlin = berlin[berlin$t < 127, ]
berlin = berlin[sample(1:nrow(berlin), 10000) ,]

###################
# Render scatterplot 3D...
###################

p <- plot_ly(berlin, x = ~x, y = ~y, z = ~t, color = ~t) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Lon'),
                      yaxis = list(title = 'Lat'),
                      zaxis = list(title = 'Time')))
chart_link = api_create(p, filename="test", fileopt = "overwrite")
