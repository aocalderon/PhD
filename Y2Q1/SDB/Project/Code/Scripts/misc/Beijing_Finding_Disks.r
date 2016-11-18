
library(SparkR)
library(leaflet)
library(sp)
options(digits=15)

epsilon = 100
source("examples.R")

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

sql = paste0("SELECT * FROM p1 JOIN p2 ON POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), ",epsilon,") WHERE p2.id < p1.id")
pairs = sql(sqlContext,sql)
head(pairs)
nrow(pairs)

centers <- SparkR:::map(pairs, calculateDisk)
d <- createDataFrame(sqlContext, centers)
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
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2) %>% addProviderTiles("Esri.WorldImagery", group = "ESRI") %>% addLayersControl(baseGroup = c("OSM(default", "ESRI"))

file = 'map2.html'
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

336*2


