
library(SparkR)
library(leaflet)
library(sp)
options(digits=15)

epsilon = 100
source("pbfe.R")

data = read.csv("sample_small.csv")
names(data) = c("ID","lat","lng")
head(data)

map = leaflet() %>%
  addTiles() %>%
  addCircleMarkers(lng=data$lng, lat=data$lat,weight=2,fillOpacity=1,color="blue",radius=2)

file = 'map1.html'
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

sc <- sparkR.init("local[*]", "SparkR")
sqlContext <- sparkRSQL.init(sc)

dataRDD = SparkR:::textFile(sc,"sample_small.csv")
dataRDD = SparkR:::map(dataRDD, transformCoords)

schema <- structType(structField("id", "double"), structField("lng", "double"), structField("lat", "double"))
points <- createDataFrame(sqlContext, dataRDD, schema = schema)
cache(points)

head(points)
count(points)

registerTempTable(points, "p1")
registerTempTable(points, "p2")

sql = paste0("SELECT * FROM p1 DISTANCE JOIN p2 ON POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), ",epsilon,") WHERE p2.id < p1.id")
pairs = sql(sqlContext,sql)
head(pairs)
nrow(pairs)

centers <- SparkR:::map(pairs, calculateDisk)
schema <- structType(structField("id1", "double"), structField("id2", "double"), structField("lng1", "double"), structField("lat1", "double"), structField("lng2", "double"), structField("lat2", "double"))
d <- createDataFrame(sqlContext, centers, schema = schema)
head(d)
count(d)

centers_lnglat <- SparkR:::map(centers, transformCenters)
disks <- as.data.frame(createDataFrame(sqlContext,centers_lnglat))
names(disks) = c("id1","id2","lng1","lat1","lng2","lat2")
head(disks)
nrow(disks)

p = sort(unique(c(disks$id1,disks$id2)))
data2 = data[p,]
map = leaflet() %>% setView(lat = 39.990010, lng = 116.317406, zoom = 15) %>% addTiles() %>% 
        addCircles(lng=disks$lng1, lat=disks$lat1, weight=2, fillOpacity=0.25, color="red", radius = epsilon/2) %>%
        addCircles(lng=disks$lng2, lat=disks$lat2, weight=2, fillOpacity=0.25, color="red", radius = epsilon/2) %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2) %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2) %>% 
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>% 
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"))

file = 'map2.html'
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

registerTempTable(d, "d")
registerTempTable(points, "p")

sql = paste0("SELECT d.lng1 AS lng, d.lat1 AS lat, id AS id_member FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng1, d.lat1), ",(epsilon/2)+0.01,")")
mdisks = sql(sqlContext,sql)
registerTempTable(mdisks, "m")
sql = "SELECT lng, lat FROM m GROUP BY lng, lat HAVING count(id_member) >= 3"
mdisks1 = sql(sqlContext,sql)

sql = paste0("SELECT d.lng2 AS lng, d.lat2 AS lat, id AS id_member FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng2, d.lat2), ",(epsilon/2)+0.01,")")
mdisks = sql(sqlContext,sql)
registerTempTable(mdisks, "m")
sql = "SELECT lng, lat FROM m GROUP BY lng, lat HAVING count(id_member) >= 3"
mdisks2 = sql(sqlContext,sql)

mdisks = as.data.frame(rbind(mdisks1, mdisks2))
id = seq(1,nrow(mdisks))
mdisks$id = id

coordinates(mdisks) = ~lng+lat
proj4string(mdisks) = mercator
mdisks = spTransform(mdisks, wgs84)
mdisks$lng1 = coordinates(mdisks)[,1]
mdisks$lat1 = coordinates(mdisks)[,2]

map = leaflet() %>% setView(lat = 39.990010, lng = 116.317406, zoom = 15) %>% addTiles() %>% 
        addCircles(lng=mdisks$lng1, lat=mdisks$lat1, weight=2, fillOpacity=0.25, color="blue", radius = epsilon/2) %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2) %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2) %>% 
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>% 
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"))

file = 'map3.html'
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

v = c(0,1,2,3,4,5,6,7,8,9)

k = c(7,5,2,9)

prod(is.element(k, v) )

k = c(7,5,20,9)

prod(is.element(k, v) )

m <- as.data.frame(rbind(mdisks1, mdisks2))
m$id = seq(1,nrow(m))
m = createDataFrame(sqlContext, m)
head(m)
count(m)
registerTempTable(m, "m")

head(points)
count(points)

sql = paste0("SELECT m.id AS mid, p.id AS pid FROM m DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE (POINT(m.lng, m.lat), ",(epsilon/2)+0.01,")")
t = sql(sqlContext,sql)
head(t, 20)
count(t)

library(sqldf)

t = as.data.frame(t)
g = sqldf("SELECT mid, group_concat(pid) AS pids FROM t GROUP BY mid")
head(g)
nrow(g)

j = sqldf("SELECT * FROM g AS g1 JOIN g AS g2 WHERE g1.mid < g2.mid")
head(j)
nrow(j)




