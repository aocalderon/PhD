#!/usr/bin/Rscript

library(data.table)
library(OpenStreetMap)
library(rgl)
library(rgdal)

###################
# Setting global variables...
###################

EXAGGERATION = 1000
LAYERS = 15
FIRST_LAYER = 117
WGS84 = "+init=epsg:4326"
DHDN = "+init=epsg:3068"
SOURCE = "bing"
DATASET = "/opt/Datasets/Berlin/berlin.csv"
DATASET = "~/Datasets/Berlin/berlin.csv"
rgl.viewpoint(  zoom = .8 )

###################
# Mapping functions...
###################

map3d <- function(map, stc_data = NULL){
  if(length(map$tiles)!=1){stop("multiple tiles not implemented") }
  nx = map$tiles[[1]]$xres
  ny = map$tiles[[1]]$yres
  xmin = map$tiles[[1]]$bbox$p1[1]
  xmax = map$tiles[[1]]$bbox$p2[1]
  ymin = map$tiles[[1]]$bbox$p1[2]
  ymax = map$tiles[[1]]$bbox$p2[2]
  xc = seq(xmin,xmax,len=ny)
  yc = seq(ymin,ymax,len=nx)
  colours = matrix(map$tiles[[1]]$colorData,ny,nx)
  m = matrix(FIRST_LAYER * EXAGGERATION,ny,nx)
  surface3d(xc,yc,m,col=colours,lit=FALSE)
  coordinates(stc_data) <- c("x", "y")
  proj4string(stc_data) <- CRS(WGS84)
  stc_data = spTransform(stc_data, CRS(DHDN))
  points3d(x=coordinates(stc_data)[,"x"]
           , y=coordinates(stc_data)[,"y"]
           , z=stc_data$t * EXAGGERATION
           , col=stc_data$t - FIRST_LAYER + 1)
}

createBaseMap <- function(dataframe, Zoom = NULL, Type = "osm", MergeTiles = TRUE, Title = "Test", proj = NULL) {
  ## Retrieve Upper Left / Lower Right lat and long
  UpperLeft <- c(max(dataframe$y),min(dataframe$x))
  ifelse(UpperLeft[1] <= 80, UpperLeft [1] <- UpperLeft[1],UpperLeft[1] <- 90)
  ifelse(UpperLeft[2] <= 170, UpperLeft[2] <- UpperLeft[2], UpperLeft[2] <- 180) 
  LowerRight <- c(min(dataframe$y),max(dataframe$x))
  ifelse(LowerRight[1] >= -80, LowerRight[1] <- LowerRight[1], LowerRight[1] <- -90) 
  ifelse(LowerRight[2] >= -170, LowerRight[2] <- LowerRight[2], LowerRight[2] <- -180) 
  print("Bounding Box Lat/Long Boundary =")
  print(paste("Upper Left Lat/Long =",UpperLeft[1],",",UpperLeft[2]))
  print(paste("Lower Right Lat/Long =",LowerRight[1],",",LowerRight[2]))
  ## retrieve the open map
  datamap <- openmap(upperLeft = UpperLeft
                     ,lowerRight = LowerRight
                     ,zoom = Zoom
                     ,type = Type
                     ,mergeTiles = MergeTiles)
  datamap = openproj(datamap, projection = DHDN)
  plot(datamap,raster = T,main = paste(Title, "test visualization"))
  return(datamap)
}

###################
# Reading data...
###################

data = read.csv(DATASET, header = F)
berlin = as.data.table(data[,c(2,3,4,1)])
names(berlin) = c('id','x','y','t')
berlin = berlin[berlin$t >= FIRST_LAYER, ]

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

berlin = berlin[berlin$t < FIRST_LAYER + LAYERS, ]
berlin = berlin[sample(1:nrow(berlin), 10000) ,]

###################
# Render scatterplot 3D...
###################

map = createBaseMap(berlin, Type = SOURCE)
map3d(map, berlin)
writeWebGL("/tmp/test")


###################
# Plotly code...
###################
#
# p <- plot_ly(berlin, x = ~x, y = ~y, z = ~t, marker = list(size = 1), color = ~t) %>%
#  add_markers() %>%
#  layout(scene = list(xaxis = list(title = 'Lon'),
#                      yaxis = list(title = 'Lat'),
#                      zaxis = list(title = 'Time')))
###################
# Setting credentials...
###################
#
# Sys.setenv("plotly_username"="aocalderon1978")
# Sys.setenv("plotly_api_key"="dx4LIeqcXzokLrO2SUHF")
# chart_link = api_create(p, filename="test", fileopt = "overwrite")

