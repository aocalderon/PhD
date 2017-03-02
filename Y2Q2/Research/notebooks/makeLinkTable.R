#!/usr/bin/Rscript

library(optparse)
 
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
names(data) = c("pid","lat","lng")

sc <- sparkR.init("local[*]", "SparkR")
sqlContext <- sparkRSQL.init(sc)

dataRDD = SparkR:::textFile(sc,opt$input)
dataRDD = SparkR:::map(dataRDD, transformCoords)

print("Transforming coordinates...")

schema <- structType(structField("pid", "double"), structField("lng", "double"), structField("lat", "double"))
points <- createDataFrame(sqlContext, dataRDD, schema = schema)
# cache(points)

# head(points)
# count(points)

registerTempTable(points, "p1")
registerTempTable(points, "p2")

print("Running distance join...")

sql = paste0("SELECT * FROM p1 DISTANCE JOIN p2 ON POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), ",epsilon,") WHERE p2.pid < p1.pid")
pairs = sql(sqlContext,sql)
# head(pairs)
# nrow(pairs)

centers <- SparkR:::map(pairs, calculateDisk)
schema <- structType(structField("pid1", "double"), structField("pid2", "double"), structField("lng1", "double"), structField("lat1", "double"), structField("lng2", "double"), structField("lat2", "double"))
d <- createDataFrame(sqlContext, centers, schema = schema)
# head(d)
# count(d)

print("Transforming centers...")

centers_lnglat <- SparkR:::map(centers, transformCenters)
disks <- as.data.frame(createDataFrame(sqlContext,centers_lnglat))
names(disks) = c("pid1","pid2","lng1","lat1","lng2","lat2")
# head(disks)
# nrow(disks)

p = sort(unique(c(disks$id1,disks$id2)))
data2 = data[p,]


registerTempTable(d, "d")
registerTempTable(points, "p")

print("Pruning disks with less than mu objects...")

sql = paste0("SELECT d.lng1 AS lng, d.lat1 AS lat, pid FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng1, d.lat1), ",(epsilon/2)+0.01,")")
mmdisks1 = sql(sqlContext,sql)
sql = paste0("SELECT d.lng2 AS lng, d.lat2 AS lat, pid FROM d DISTANCE JOIN p ON POINT(p.lng, p.lat) IN CIRCLERANGE(POINT(d.lng2, d.lat2), ",(epsilon/2)+0.01,")")
mmdisks2 = sql(sqlContext,sql)

mmdisks = as.data.frame(rbind(mmdisks1, mmdisks2))
names(mmdisks) = c('lng', 'lat', 'pid')
udisks = sqldf("SELECT lng, lat FROM mmdisks GROUP BY lng, lat")
udisks$did = seq(1,nrow(udisks))
mmdisks = sqldf("SELECT u.did AS did, m.pid AS pid FROM udisks AS u JOIN mmdisks AS m ON(u.lat=m.lat AND u.lng= m.lng)")
coordinates(udisks) = ~lng+lat
proj4string(udisks) = mercator
udisks = spTransform(udisks, wgs84)
write.csv(data, "points.csv", row.names = F)
write.csv(udisks, 'disks.csv', row.names = F)
write.csv(mmdisks, 'links.csv', row.names = F)
