library(sp)

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:4799")

p = read.csv('POIs.csv')
p$lat <- as.double(p$lat)
p$lon <- as.double(p$lon)
coordinates(p) = ~lon+lat
proj4string(p) = wgs84
p = spTransform(p, mercator)
write.table(p, 'samplePOIs.csv', row.names = F, sep = ',')
p = read.csv('samplePOIs.csv')
