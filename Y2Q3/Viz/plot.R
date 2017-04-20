library(sp)

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:4799")

points = read.csv('p.csv', header = F)
names(points) = c("ID","lat","lng")

coordinates(points) = ~lng+lat
proj4string(points) = wgs84
points = spTransform(points, mercator)


library("ggmap")
qmap(location = "Beijing")
qmap(location = "Beijing", source="osm")
