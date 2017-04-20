library(sp)

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:4799")

# points = read.csv('p.csv', header = F)
# names(points) = c("ID","lat","lng")

# points = read.csv('mdisks.csv')
# points = points[, c('id','lng1','lat1')]
# names(points) = c("ID","lng","lat")

# points = read.csv('disks.csv')
# points1 = points[,c('lng1','lat1')]
# names(points1) = c("lng","lat")
# points2 = points[,c('lng2','lat2')]
# names(points2) = c("lng","lat")
# points = rbind(points1, points2)
# points$ID = 1:nrow(points)
# points = points[,c('ID','lng','lat')]

points = read.csv('maximal.csv')
points = points[, c('pids','lng1','lat1')]
names(points) = c("ID","lng","lat")

coordinates(points) = ~lng+lat
proj4string(points) = wgs84
points = spTransform(points, mercator)
points = data.frame(ID=points$ID, lng=coordinates(points)[,1], lat=coordinates(points)[,2])

#write.table(points, 'p_4799.csv', row.names = F, col.names = F, sep = ',')
#write.table(points, 'f0_4799.csv', row.names = F, col.names = F, sep = ',')
#write.table(points, 'f1_4799.csv', row.names = F, col.names = F, sep = ',')
write.table(points, 'f2_4799.csv', row.names = F, col.names = F, sep = ',')
