library(sp)

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:4799")

points = read.csv('p.csv', header = F)
names(points) = c("ID","lat","lng")

coordinates(points) = ~lng+lat
proj4string(points) = wgs84
points = spTransform(points, mercator)


library("ggmap")
map <- get_map(location = c(lon[1], lat[2], lon[2], lat[1]),
               maptype = "roadmap", source = "osm", zoom = 11)