library(sp)
library(leaflet)

epsilon = 200
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

points <- data.frame(lng=as.numeric(c(-77.2858, -77.2850)),
                     lat=as.numeric(c(1.2059, 1.2050)))
d <- SpatialPointsDataFrame(coords = points, data = points, proj4string = CRS("+init=epsg:4326"))
d_mrc <- spTransform(d, CRS("+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs"))

centers = calculateDisk(coordinates(d_mrc)[1,1], coordinates(d_mrc)[1,2], coordinates(d_mrc)[2,1], coordinates(d_mrc)[2,2])
c_merc <- SpatialPointsDataFrame(coords = centers, data = centers, proj4string = CRS("+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs"))
c <- spTransform(c_merc, CRS("+init=epsg:4326"))
centers = as.data.frame(coordinates(c))

map = leaflet() %>% addProviderTiles("Esri.WorldStreetMap") %>% 
      addCircles(lng=centers$lng, lat=centers$lat, weight=2, fillOpacity=0.35, color="red", radius = epsilon/2) %>%
      addCircleMarkers(lng=points$lng, lat=points$lat, radius = 0.1) 
map  
