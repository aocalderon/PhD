
r2 = (epsilon/2)^2

wgs84 = CRS("+init=epsg:4326")
mercator = CRS("+init=epsg:4799")

transformCoords <- function(line){
  library(sp)
  p = as.data.frame(t(strsplit(line, ",")[[1]]),stringsAsFactors=F)
  p$V2 <- as.double(p$V2)
  p$V3 <- as.double(p$V3)
  coordinates(p) = ~V3+V2
  proj4string(p) = wgs84
  p = spTransform(p, mercator)
  
  return(as.double(c(p$V1,p@coords)))
}

calculateDisk <- function(pair){
  x = as.double(pair[2]) - as.double(pair[5])
  y = as.double(pair[3]) - as.double(pair[6])
  d2 = x^2 + y^2
  if(d2 != 0){
    root = sqrt(abs(4 * (r2 / d2) - 1))
    h1 = ((x + y * root) / 2) + as.double(pair[5])
    h2 = ((x - y * root) / 2) + as.double(pair[5])
    k1 = ((y - x * root) / 2) + as.double(pair[6])
    k2 = ((y + x * root) / 2) + as.double(pair[6])
    return(c(as.double(pair[1]),as.double(pair[4]), h1, k1, h2, k2))
  }
  return(NULL)
}

transformCenters <- function(centers){
  library(sp)
  c1 = data.frame(lng=c(centers[3]), lat=c(centers[4]))
  c1$lng <- as.double(c1$lng)
  c1$lat <- as.double(c1$lat)
  coordinates(c1) = ~lng+lat
  proj4string(c1) = mercator
  c1 = spTransform(c1, wgs84)
  c2 = data.frame(lng=c(centers[5]), lat=c(centers[6]))
  c2$lng <- as.double(c2$lng)
  c2$lat <- as.double(c2$lat)
  coordinates(c2) = ~lng+lat
  proj4string(c2) = mercator
  c2 = spTransform(c2, wgs84)
  
  return(as.numeric(c(centers[1],centers[2],c1@coords,c2@coords)))
}

transform <- function(points){
  library(sp)

  coordinates(points) = ~lng+lat
  proj4string(points) = mercator
  points = spTransform(points, wgs84)
  
  return(as.numeric(c(points$id,points@coords)))
}
