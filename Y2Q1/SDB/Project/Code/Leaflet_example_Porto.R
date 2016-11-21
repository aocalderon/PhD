library(sp)
library(leaflet)
library(sqldf)

epsilon = 100
r2 = (epsilon/2)^2

distance <- function(x1, y1, x2, y2){
  return(sqrt((x1-x2)^2+(y1-y2)^2))
}

calculateDisk <- function(x1, y1, x2, y2){
  x = x1 - x2
  y = y1 - y2
  d2 = x^2 + y^2
  if(d2 != 0){
    root = sqrt(abs(4 * (r2 / d2) - 1))
    h1 = ((x + y * root) / 2) + x2
    h2 = ((x - y * root) / 2) + x2
    k1 = ((y - x * root) / 2) + y2
    k2 = ((y + x * root) / 2) + y2
    return(data.frame(lng=c(h1, h2), lat=c(k1, k2)))
  }
  return(NULL)
}

data = read.csv('/opt/Datasets/Lisbon/L100K.csv', header = F)
data = data[1:500,]

points <- data.frame(lng=data$V1, lat=data$V2)
d <- SpatialPointsDataFrame(coords = points, data = points, proj4string = CRS("+init=epsg:4326"))
d_mrc <- spTransform(d, CRS("+init=epsg:27493"))

d1 = as.data.frame(coordinates(d_mrc))
d2 = d1

d = sqldf("SELECT * FROM d1 JOIN d2 WHERE d1.lng <> d2.lng AND d1.lat <> d2.lat")
centers = data.frame(lng=c(1),lat=c(1))

for(i in seq(1,nrow(d))){
  if(distance(d[i,1],d[i,2],d[i,3],d[i,4]) < epsilon){
    centers = rbind(centers,calculateDisk(d[i,1],d[i,2],d[i,3],d[i,4]))
  }
}
c_merc <- SpatialPointsDataFrame(coords = centers, data = centers, proj4string = CRS("+init=epsg:27493"))
c <- spTransform(c_merc, CRS("+init=epsg:4326"))
centers = as.data.frame(coordinates(c))

map = leaflet() %>% addProviderTiles("Esri.WorldStreetMap") %>% 
  addCircles(lng=centers$lng, lat=centers$lat, weight=2, fillOpacity=0.35, color="red", radius = epsilon/2) %>%
  addCircleMarkers(lng=points$lng, lat=points$lat, radius = 0.1) 
map  
