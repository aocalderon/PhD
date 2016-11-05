library(SparkR)

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:3857")

transformCoords <- function(line){
  library(sp)
  p = as.data.frame(t(strsplit(line, ",")[[1]]),stringsAsFactors=F)
  p$V2 <- as.double(p$V2)
  p$V3 <- as.double(p$V3)
  coordinates(p) = ~V3+V2
  proj4string(p) = wgs84
  p = spTransform(p, mercator)
  
  return(c(p$V1,p@coords))
}

#dataRDD = SparkR:::textFile(sc,"/home/and/notebooks/sample_small.csv")
#dataRDD_splitted = SparkR:::map(dataRDD, transformCoords)
#points <- createDataFrame(sqlContext, dataRDD_splitted)
#head(points)

r = "898,40.3586583,116.5952766"
transformCoords(r)
