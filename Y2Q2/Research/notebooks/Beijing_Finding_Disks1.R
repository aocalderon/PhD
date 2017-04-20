#!/usr/bin/Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(optparse, leaflet, rgdal)
 
option_list = list(
	make_option(c("-o", "--out"), type="character", default="map" , help="Prefix for output html files", metavar="character"),
	make_option(c("-i", "--input"), type="character", default="sample_small.csv" , help="Input file", metavar="character"),
	make_option(c("-x", "--lat"), type="double", default=39.976057, help="Latitude [default:%default]", metavar="double"),
	make_option(c("-y", "--lng"), type="double", default=116.330243, help="Longitude [default:%default]", metavar="double"),
	make_option(c("-e", "--epsilon"), type="double", default=100.0, help="Epsilon [default:%default]", metavar="double"),
	make_option(c("-m", "--mu"), type="integer", default=3, help="Mu [default:%default]", metavar="integer"),
	make_option(c("-z", "--zoom"), type="integer", default=15, help="Zoom [default:%default]", metavar="integer")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

library(sqldf)
library(SparkR)
library(leaflet)
library(sp)
options(digits=15)

source("input.R")

the_lat = opt$lat
the_lng = opt$lng
the_zoom = opt$zoom
epsilon = opt$epsilon
mu = opt$mu
source("pbfe.R")
output = paste0(opt$out, "_E",  epsilon, "_M", mu)

data = read.csv(opt$input, header = F)
names(data) = c("ID","lat","lng")
# head(data)

# map = leaflet() %>%
#	addTiles() %>%
#	addCircleMarkers(lng=data$lng, lat=data$lat,weight=2,fillOpacity=1,color="blue",radius=2)

# file = 'map.html'
# htmlwidgets::saveWidget(map, file = file, selfcontained = F)
# IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

Sys.setenv(SPARK_HOME = "/opt/Simba")
sc <- sparkR.init("local[*]", "SparkR")
sqlContext <- sparkRSQL.init(sc)

dataRDD = SparkR:::textFile(sc,opt$input)
dataRDD = SparkR:::map(dataRDD, transformCoords)

print("Transforming coordinates...")

schema <- structType(structField("id", "double"), structField("lng", "double"), structField("lat", "double"))
points <- createDataFrame(sqlContext, dataRDD, schema = schema)
# cache(points)

# head(points)
# count(points)

registerTempTable(points, "p1")
registerTempTable(points, "p2")

print("Running distance join...")

sql = paste0("SELECT * FROM p1 DISTANCE JOIN p2 ON POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), ",epsilon,") WHERE p2.id < p1.id")
pairs = sql(sqlContext,sql)
# head(pairs)
# nrow(pairs)

centers <- SparkR:::map(pairs, calculateDisk)
schema <- structType(structField("id1", "double"), structField("id2", "double"), structField("lng1", "double"), structField("lat1", "double"), structField("lng2", "double"), structField("lat2", "double"))
d <- createDataFrame(sqlContext, centers, schema = schema)
# head(d)
# count(d)

print("Transforming centers...")

centers_lnglat <- SparkR:::map(centers, transformCenters)
disks <- as.data.frame(createDataFrame(sqlContext,centers_lnglat))
names(disks) = c("id1","id2","lng1","lat1","lng2","lat2")
# head(disks)
# nrow(disks)

p = sort(unique(c(disks$id1,disks$id2)))
data2 = data[p,]

print("Saving map with initial set...")

map = leaflet() %>% setView(lat = the_lat, lng = the_lng, zoom = the_zoom) %>% addTiles() %>%
        addCircles(lng=disks$lng1, lat=disks$lat1, weight=2, fillOpacity=0.25, color="red", radius = epsilon/2) %>%
        addCircles(lng=disks$lng2, lat=disks$lat2, weight=2, fillOpacity=0.25, color="red", radius = epsilon/2) %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2) %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2, popup=paste(data2$ID)) %>%
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>%
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"))

file = paste0(output,'_P1.html')
htmlwidgets::saveWidget(map, file = file, selfcontained = F)

registerTempTable(d, "d")
registerTempTable(points, "p")

print("Pruning disks with less than mu objects...")

sql = paste0("SELECT d.lng1 AS lng, d.lat1 AS lat, id AS id_member FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng1, d.lat1), ",(epsilon/2)+0.01,")")
mmdisks1 = sql(sqlContext,sql)
registerTempTable(mmdisks1, "m1")
sql = paste0("SELECT lng, lat FROM m1 GROUP BY lng, lat HAVING count(id_member) >= ", mu)
mdisks1 = sql(sqlContext,sql)

sql = paste0("SELECT d.lng2 AS lng, d.lat2 AS lat, id AS id_member FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng2, d.lat2), ",(epsilon/2)+0.01,")")
mmdisks2 = sql(sqlContext,sql)
registerTempTable(mmdisks2, "m2")
sql = paste0("SELECT lng, lat FROM m2 GROUP BY lng, lat HAVING count(id_member) >= ", mu)
mdisks2 = sql(sqlContext,sql)

mdisks = as.data.frame(rbind(mdisks1, mdisks2))
id = seq(1,nrow(mdisks))
mdisks$id = id

coordinates(mdisks) = ~lng+lat
proj4string(mdisks) = mercator
mdisks = spTransform(mdisks, wgs84)
mdisks$lng1 = coordinates(mdisks)[,1]
mdisks$lat1 = coordinates(mdisks)[,2]

print("Saving map after pruning 1...")

map = leaflet() %>% setView(lat = the_lat, lng = the_lng, zoom = the_zoom) %>% addTiles() %>%
        addCircles(lng=mdisks$lng1, lat=mdisks$lat1, weight=2, fillOpacity=0.25, color="blue", radius = epsilon/2) %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2) %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2) %>%
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>%
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"))

file = paste0(output,'_P2.html')
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
# IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

print("Detecting redundant disks...")

m <- as.data.frame(rbind(mdisks1, mdisks2))
m$id = seq(1,nrow(m))
m = createDataFrame(sqlContext, m)
# head(m)
# count(m)
registerTempTable(m, "m")

# head(points)
# count(points)

sql = paste0("SELECT m.id AS mid, p.id AS pid FROM m DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE (POINT(m.lng, m.lat), ",(epsilon/2)+0.01,") ORDER BY mid, pid")
t = sql(sqlContext,sql)
# head(t, 30)
# count(t)


t = as.data.frame(t)
g = sqldf("SELECT mid, group_concat(CAST(pid AS INT)) AS pids FROM t GROUP BY mid ORDER BY count(pid)")
# head(g)
# nrow(g)
# tail(g)

n = c()
r = nrow(g)
for(i in 1:r){
    disk_i = as.numeric(unlist(strsplit(g[i,2], ',')))
    flag = 1
    for(j in (i+1):r){
        disk_j = as.numeric(unlist(strsplit(g[j,2], ',')))
        # print(paste0("Disk 1: ", disk_i, " Disk 2: ", disk_j, " Result: ", is.element(disk_i, disk_j)))

        if(prod(is.element(disk_i, disk_j)) == 1){
            flag = 0
            break
        }
    }
    if(flag == 1){
        n = c(n, i)
    }
}
n = c(n, r)
n = unique(n)

g = g[n,]

m = as.data.frame(m)
maximal = sqldf("SELECT lng, lat, pids FROM g JOIN m ON(id = mid)")
# head(maximal)
# nrow(maximal)

coordinates(maximal) = ~lng+lat
proj4string(maximal) = mercator
maximal = spTransform(maximal, wgs84)
maximal$lng1 = coordinates(maximal)[,1]
maximal$lat1 = coordinates(maximal)[,2]

print("Saving map after pruning 2...")

map = leaflet() %>% setView(lat = the_lat, lng = the_lng, zoom = the_zoom) %>% addTiles() %>%
        addCircles(lng=maximal$lng1, lat=maximal$lat1, weight=2, fillOpacity=0.25, color="orange", radius = epsilon/2, popup = maximal$pids) %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2) %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius = 2) %>%
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>%
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"))

file = paste0(output,'_P3.html')
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
# IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

print("Saving final map...")

map = leaflet() %>% setView(lat = the_lat, lng = the_lng, zoom = the_zoom) %>% addTiles() %>%
        addCircleMarkers(lng=data$lng, lat=data$lat, weight=2, fillOpacity=1,radius = 2, group = "Points") %>%
        addCircles(lng=disks$lng1, lat=disks$lat1, weight=2, fillOpacity=0.10, color="red", radius = epsilon/2, group = "Initial set") %>%
        addCircles(lng=disks$lng2, lat=disks$lat2, weight=2, fillOpacity=0.10, color="red", radius = epsilon/2, group = "Initial set") %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius=2, group = "Initial set") %>%
        addCircles(lng=mdisks$lng1, lat=mdisks$lat1, weight=2, fillOpacity=0.20, color="blue", radius = epsilon/2, group = "Prune less than mu") %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius=2, group = "Prune less than mu") %>%
        addCircles(lng=maximal$lng1, lat=maximal$lat1, weight=2, fillOpacity=0.40, color="orange", radius = epsilon/2, popup = maximal$pids, group = "Prune redundant") %>%
        addCircleMarkers(lng=data2$lng, lat=data2$lat, weight=2, fillOpacity=1, color="purple", radius=2, group = "Prune redundant") %>%
        addProviderTiles("Esri.WorldImagery", group = "ESRI") %>%
        addLayersControl(baseGroup = c("OSM(default)", "ESRI"), overlayGroups = c("Initial set", "Prune less than mu", "Prune redundant", "Points"), options = layersControlOptions(collapsed = FALSE, autoZIndex = FALSE))

file = paste0(output,'_All.html')
htmlwidgets::saveWidget(map, file = file, selfcontained = F)
# IRdisplay::display_html(paste("<iframe width=100% height=400 src=' ", file, " ' ","/>"))

write.csv(disks, "disks.csv", row.names = F)
write.csv(maximal, "maximal.csv", row.names = F)
write.csv(mdisks, "prune.csv", row.names = F)
write.csv(maximal, "maximal.csv", row.names = F)
