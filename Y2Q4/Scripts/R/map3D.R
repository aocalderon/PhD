library("OpenStreetMap")
library("rgl")

map3d <- function(map, stc_data = NULL){
  if(length(map$tiles)!=1){stop("multiple tiles not implemented") }
  nx = map$tiles[[1]]$xres
  ny = map$tiles[[1]]$yres
  xmin = map$tiles[[1]]$bbox$p1[1]
  xmax = map$tiles[[1]]$bbox$p2[1]
  ymin = map$tiles[[1]]$bbox$p1[2]
  ymax = map$tiles[[1]]$bbox$p2[2]
  xc = seq(xmin,xmax,len=ny)
  yc = seq(ymin,ymax,len=nx)
  colours = matrix(map$tiles[[1]]$colorData,ny,nx)
  m = matrix(117/EXAGGERATION,ny,nx)
  surface3d(xc,yc,m,col=colours)
  points3d(stc_data$x, stc_data$y, stc_data$t, col=1:)
}

createBaseMap <- function(dataframe, Zoom = NULL, Type = "osm", MergeTiles = TRUE, Title = "Test", proj = NULL) {
  ## Retrieve Upper Left / Lower Right lat and long
  UpperLeft <- c(max(dataframe$y),min(dataframe$x))
  ifelse(UpperLeft[1] <= 80, UpperLeft [1] <- UpperLeft[1],UpperLeft[1] <- 90)
  ifelse(UpperLeft[2] <= 170, UpperLeft[2] <- UpperLeft[2], UpperLeft[2] <- 180) 
  LowerRight <- c(min(dataframe$y),max(dataframe$x))
  ifelse(LowerRight[1] >= -80, LowerRight[1] <- LowerRight[1], LowerRight[1] <- -90) 
  ifelse(LowerRight[2] >= -170, LowerRight[2] <- LowerRight[2], LowerRight[2] <- -180) 
  print("Bounding Box Lat/Long Boundary =")
  print(paste("Upper Left Lat/Long =",UpperLeft[1],",",UpperLeft[2]))
  print(paste("Lower Right Lat/Long =",LowerRight[1],",",LowerRight[2]))
  ## retrieve the open map
  datamap <- openmap(upperLeft = UpperLeft
                       ,lowerRight = LowerRight
                       ,zoom = Zoom
                       ,type = Type
                       , mergeTiles = MergeTiles)
  if ( is.null(proj) == FALSE ) {
    datamap <- openproj(datamap, projection = "+proj=longlat")
    warning("proj must equal a true projection")
  }
  plot(datamap,raster = T,main = paste(Title, "test visualization"))
  return(openproj(datamap))
}